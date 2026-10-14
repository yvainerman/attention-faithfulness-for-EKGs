"""
data/ptbxl.py
-------------
PTB-XL loader for all THREE SCP label schemes -- diagnostic, rhythm, form --
with MULTI-LABEL (multi-hot) targets.

For a chosen `label_scheme`, every in-scheme SCP statement present in a record
is a positive label (diagnostic statements are first mapped to
`diagnostic_level`: "superclass" (default) -> NORM/MI/STTC/CD/HYP, "subclass",
or "statement"). Rare classes (< MIN_CLASS_COUNT records) are dropped when
filtered=True, and a record is dropped only if no label survives.
NOTE: `filtered` refers to this rare-CLASS filtering, not signal filtering.

SIGNAL PREPROCESSING. The PTB-XL .dat files are NOT filtered by the dataset
authors (noise is only flagged in metadata columns such as baseline_drift and
static_noise), so every signal is cleaned here, once, when the cache is built.
Every model and every test then reads the same cleaned signals:
  1. Wavelet denoising of each lead with denoise(lead) from
     data/utils/denoise.py, unchanged (CAT-Net recipe: db5, 9 levels,
     cD1+cD2 zeroed, cD3-cD9 soft-thresholded).                     ECG_DENOISE
  2. Baseline-wander removal: the lead's level-9 db5 approximation (cA9,
     everything < 0.49 Hz at 500 Hz) is rebuilt on its own and subtracted
     (_baseline below). denoise() leaves cA9 untouched, so this equals
     zeroing cA9 inside the same wavelet pass.              ECG_REMOVE_BASELINE
  3. Per-lead z-score over the whole recording.                       ECG_ZSCORE
All three default to ON. Set an env var to 0 to switch a step off. The wavelet
recipe assumes 500 Hz, so denoising requires fs=500.

CACHE FILES (in the PTB-XL folder, one per label scheme):
  ptbxl_rhythm.pkl   ptbxl_form.pkl   ptbxl_diagnostic-superclass.pkl
  (+ "_all" before .pkl when filtered=False)
"IS IT ALREADY CLEAN?" is answered by the cache, never by the raw files: each
cache stores the settings it was built with (CACHE_VERSION, fs, filtered and
the preprocessing steps). A cache whose settings differ from the current ones,
or an empty one, is rebuilt instead of loaded. Bump CACHE_VERSION after
changing how signals are built, to force a rebuild.

Return contract: (data_list, num_classes, label2idx) with
data_list = list of (ecg[12, T] float32, label_multihot[num_classes], patient_id).
The cache also stores `strat_folds` aligned to data_list, used by
ptbxl_split_indices and carried into the beat cache by data/sph_beats.py.
"""
import ast
import os
import pickle
import time

import numpy as np
import pandas as pd
import pywt
import wfdb
from joblib import Parallel, delayed

from data.utils.denoise import denoise      # the original one-argument denoise(lead)


CACHE_VERSION   = 1   # stored INSIDE each cache; bump to force a rebuild
MIN_CLASS_COUNT = 100
SCHEMES         = ("diagnostic", "rhythm", "form")


def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() not in ("0", "false", "no")


DENOISE         = _env_bool("ECG_DENOISE", True)
REMOVE_BASELINE = _env_bool("ECG_REMOVE_BASELINE", True) and DENOISE   # same wavelet pass
ZSCORE          = _env_bool("ECG_ZSCORE", True)


def _prep_tag():
    parts = []
    if DENOISE:
        parts.append("dwt")
    if REMOVE_BASELINE:
        parts.append("bw")
    if ZSCORE:
        parts.append("z")
    return "-".join(parts) if parts else "raw"


PREP_TAG = _prep_tag()     # short label for logs, e.g. "dwt-bw-z" (also used by sph_beats.py)
PREP = {"denoise": DENOISE, "remove_baseline": REMOVE_BASELINE, "zscore": ZSCORE,
        "wavelet": "db5", "level": 9}


def _settings(fs, filtered):
    """Everything a cache must have been built with to be reused as-is."""
    return {"version": CACHE_VERSION, "fs": fs, "filtered": bool(filtered), **PREP}


# =====================================================
# Helpers
# =====================================================
def _parse_scp_codes(s):
    if isinstance(s, dict):
        return s
    if pd.isna(s):
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def _scheme_code_set(scp, scheme):
    """SCP statements belonging to `scheme` (column == 1)."""
    if scheme not in scp.columns:
        raise ValueError(f"scp_statements.csv has no '{scheme}' column; "
                         f"columns: {list(scp.columns)}")
    col = pd.to_numeric(scp[scheme], errors="coerce").fillna(0)
    return set(scp.index[col == 1].astype(str).tolist())


def _scheme_name(scheme, diagnostic_level, filtered):
    """'rhythm', 'form', 'diagnostic-superclass', ... (+ '_all' if unfiltered).
    Shared with data/sph_beats.py so signal and beat caches are named alike."""
    name = f"diagnostic-{diagnostic_level}" if scheme == "diagnostic" else scheme
    return name if filtered else f"{name}_all"


def _cache_path(DATA_PATH, scheme, diagnostic_level, filtered, fs):
    """e.g. <DATA_PATH>/ptbxl_rhythm.pkl. fs and preprocessing are checked from
    the settings stored inside the file, not encoded in the name."""
    return os.path.join(DATA_PATH, f"ptbxl_{_scheme_name(scheme, diagnostic_level, filtered)}.pkl")


def _read_valid_cache(cache_path, fs, filtered):
    """The cache dict if the file exists, was built with the current settings
    and is not empty; otherwise None (after saying why)."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        obj = pickle.load(f)
    if obj.get("settings") != _settings(fs, filtered):
        print(f"[ptbxl] {cache_path}: built with {obj.get('settings')}, "
              f"need {_settings(fs, filtered)}; rebuilding")
        return None
    if not obj.get("data_list"):
        print(f"[ptbxl] {cache_path} is EMPTY (a failed earlier build); rebuilding")
        return None
    return obj


def _baseline(lead):
    """Baseline wander of one lead: its level-9 db5 approximation (cA9) rebuilt
    on its own, i.e. everything below fs / 2**10 (0.49 Hz at 500 Hz). Same
    wavelet and level as denoise(), so denoise(lead) - _baseline(lead) is the
    CAT-Net output with cA9 zeroed."""
    coeffs = pywt.wavedec(lead, "db5", level=9)
    coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, "db5")[: len(lead)]


def preprocess_signal(signals, denoise_on=True, remove_baseline=True, zscore=True):
    """(T, 12) raw PTB-XL signal -> (T, 12) float32 preprocessed signal.
    Denoising runs on each lead independently; z-score is per lead over the
    whole recording (a flat lead becomes all zeros)."""
    x = np.nan_to_num(np.asarray(signals, dtype=np.float64))
    T = x.shape[0]
    if denoise_on:
        leads = []
        for j in range(x.shape[1]):
            y = denoise(x[:, j])[:T]          # [:T]: waverec adds a sample for odd T
            if remove_baseline:
                y = y - _baseline(x[:, j])
            leads.append(y)
        x = np.stack(leads, axis=1)
    if zscore:
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    return x.astype(np.float32)


def check_data_path(DATA_PATH, fs=500):
    """Absolute PTB-XL folder, after confirming the raw PTB-XL files are in it.
    Stops with the full path otherwise, so a wrong ECG_DATA_ROOT (or running
    from the wrong folder) fails at once instead of building from nothing."""
    path = os.path.abspath(DATA_PATH)
    need = ["ptbxl_database.csv", "scp_statements.csv",
            "records500" if fs == 500 else "records100"]
    missing = [n for n in need if not os.path.exists(os.path.join(path, n))]
    if missing:
        raise FileNotFoundError(
            f"PTB-XL not found at {path} (missing: {', '.join(missing)}). "
            f"ECG_DATA_ROOT must be the folder that CONTAINS ptbxl/, e.g. "
            f"export ECG_DATA_ROOT=/home/yv33n/data  (currently "
            f"{os.environ.get('ECG_DATA_ROOT')!r})")
    return path


def _load_one_record(record_path, denoise_on, remove_baseline, zscore):
    """Top-level joblib worker (must be picklable): read + preprocess one record."""
    try:
        signals, _ = wfdb.rdsamp(record_path)
        return preprocess_signal(signals, denoise_on, remove_baseline, zscore)
    except Exception as e:
        return e


# =====================================================
# Public API
# =====================================================
def load_ptbxl_dataset(DATA_PATH, filtered=True, fs=500,
                       label_scheme="rhythm", diagnostic_level="superclass"):
    """Load PTB-XL with MULTI-LABEL targets under `label_scheme`.

    Returns (data_list, num_classes, label2idx).
        data_list : list of (ecg[12,T] float32, label_multihot[num_classes] float32, patient_id)
    """
    if label_scheme not in SCHEMES:
        raise ValueError(f"label_scheme must be one of {SCHEMES}, got {label_scheme!r}")
    if label_scheme == "diagnostic" and diagnostic_level not in ("superclass", "subclass", "statement"):
        raise ValueError(f"diagnostic_level invalid: {diagnostic_level!r}")
    if DENOISE and fs != 500:
        raise ValueError(f"denoising uses a 9-level wavelet recipe set for 500 Hz; got fs={fs}. "
                         f"Use fs=500 or set ECG_DENOISE=0.")
    DATA_PATH = check_data_path(DATA_PATH, fs)      # absolute path; raw PTB-XL confirmed

    cache_path = _cache_path(DATA_PATH, label_scheme, diagnostic_level, filtered, fs)
    tag = f"ptbxl-{label_scheme}"

    obj = _read_valid_cache(cache_path, fs, filtered)
    if obj is not None:
        print(f"[{tag}] Loaded {cache_path}: {len(obj['data_list'])} ECGs, "
              f"{obj['num_classes']} classes, preprocessing={PREP_TAG}, "
              f"labels: {obj['label2idx']}")
        return obj["data_list"], obj["num_classes"], obj["label2idx"]

    print(f"[{tag}] Building cache at {cache_path}  (preprocessing={PREP_TAG})")
    t0 = time.time()

    db = pd.read_csv(os.path.join(DATA_PATH, "ptbxl_database.csv"), index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(_parse_scp_codes)
    scp = pd.read_csv(os.path.join(DATA_PATH, "scp_statements.csv"), index_col=0)

    code_set = _scheme_code_set(scp, label_scheme)
    print(f"[{tag}] {len(code_set)} {label_scheme} SCP codes")

    # diagnostic statement -> level mapping (superclass / subclass / statement)
    super_map = None
    if label_scheme == "diagnostic" and diagnostic_level != "statement":
        col = "diagnostic_class" if diagnostic_level == "superclass" else "diagnostic_subclass"
        if col not in scp.columns:
            raise ValueError(f"scp_statements.csv lacks '{col}' for level={diagnostic_level}")
        super_map = scp[col].astype(str).to_dict()

    def label_set(scp_codes):
        """Set of class strings present in this record (ALL in-scheme codes)."""
        present = [c for c in map(str, scp_codes.keys()) if c in code_set]
        if super_map is not None:
            out = set()
            for c in present:
                v = super_map.get(c)
                if v not in (None, "", "nan"):
                    out.add(v)
            return out
        return set(present)

    db["labelset"] = db["scp_codes"].apply(label_set)

    # Count records per class (a record counts once per class it contains).
    class_counts = {}
    for s in db["labelset"]:
        for c in s:
            class_counts[c] = class_counts.get(c, 0) + 1
    print(f"[{tag}] {len(class_counts)} raw classes")

    keep = set(class_counts)
    if filtered:
        keep = {c for c, n in class_counts.items() if n >= MIN_CLASS_COUNT}
        dropped = set(class_counts) - keep
        if dropped:
            print(f"[{tag}] dropping rare (<{MIN_CLASS_COUNT}): {sorted(dropped)}")

    # label2idx ordered by frequency (most-common = 0), over KEPT classes
    ordered = sorted(keep, key=lambda c: (-class_counts[c], c))
    label2idx = {c: i for i, c in enumerate(ordered)}
    num_classes = len(label2idx)

    # per-record kept index set; drop records with no surviving labels
    db["idxset"] = db["labelset"].apply(
        lambda s: sorted(label2idx[c] for c in s if c in label2idx))
    n_before = len(db)
    db_kept = db[db["idxset"].map(len) > 0].copy()
    print(f"[{tag}] {len(db_kept)} records with >=1 kept label "
          f"({n_before - len(db_kept)} dropped); num_classes={num_classes}")
    print(f"[{tag}] label2idx={label2idx}")
    # positives per class (after record filtering)
    pos = {c: 0 for c in label2idx}
    inv = {i: c for c, i in label2idx.items()}
    for s in db_kept["idxset"]:
        for i in s:
            pos[inv[i]] += 1
    avg_labels = float(np.mean([len(s) for s in db_kept["idxset"]]))
    print(f"[{tag}] positives/class: "
          + ", ".join(f"{c}:{pos[c]}" for c in label2idx))
    print(f"[{tag}] avg labels/record = {avg_labels:.2f}")

    def multihot(idxs):
        v = np.zeros(num_classes, dtype=np.float32)
        if idxs:
            v[idxs] = 1.0
        return v

    fname_col, expected_T = ("filename_hr", 5000) if fs == 500 else ("filename_lr", 1000)
    paths  = [os.path.join(DATA_PATH, r[fname_col]) for _, r in db_kept.iterrows()]
    labels = [multihot(r["idxset"])                 for _, r in db_kept.iterrows()]
    pids   = [int(r["patient_id"])                  for _, r in db_kept.iterrows()]
    folds  = [int(r["strat_fold"])                  for _, r in db_kept.iterrows()]

    print(f"[{tag}] loading + preprocessing {len(paths)} waveforms at {fs} Hz "
          f"(denoise={DENOISE}, remove_baseline={REMOVE_BASELINE}, zscore={ZSCORE})...")
    results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
        delayed(_load_one_record)(p, DENOISE, REMOVE_BASELINE, ZSCORE) for p in paths)

    data_list, strat_folds, skipped, errors = [], [], 0, []
    for signals, lab, pid, fold, p in zip(results, labels, pids, folds, paths):
        if isinstance(signals, Exception):
            errors.append((p, signals))
            skipped += 1
            continue
        if signals.shape != (expected_T, 12):
            skipped += 1
            continue
        data_list.append((np.ascontiguousarray(signals.T, dtype=np.float32), lab, pid))
        strat_folds.append(fold)

    print(f"[{tag}] loaded {len(data_list)} ECGs (skipped {skipped}); "
          f"build {(time.time()-t0)/60:.1f} min")
    if errors:
        print(f"[{tag}] WARNING: {len(errors)} records failed to read/preprocess; "
              f"first: {errors[0][0]} -> {errors[0][1]!r}")
    if not data_list:
        # Never save an empty cache: it would be loaded silently on every later run.
        raise RuntimeError(f"[{tag}] no records loaded; cache NOT saved. "
                           + (f"First error: {errors[0][1]!r}" if errors else
                              f"All records had a shape other than ({expected_T}, 12)."))

    tmp = f"{cache_path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump({"data_list": data_list, "num_classes": num_classes,
                     "label2idx": label2idx, "strat_folds": strat_folds,
                     "label_scheme": label_scheme, "diagnostic_level": diagnostic_level,
                     "settings": _settings(fs, filtered)},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)
    print(f"[{tag}] saved cache -> {cache_path}")
    return data_list, num_classes, label2idx


def load_ptbxl_strat_folds(DATA_PATH, filtered=True, fs=500,
                           label_scheme="rhythm", diagnostic_level="superclass"):
    """Return strat_folds aligned to load_ptbxl_dataset's data_list order.
    Builds (or rebuilds) the cache first if it is missing or out of date."""
    cache_path = _cache_path(DATA_PATH, label_scheme, diagnostic_level, filtered, fs)
    obj = _read_valid_cache(cache_path, fs, filtered)
    if obj is None:
        load_ptbxl_dataset(DATA_PATH, filtered, fs, label_scheme, diagnostic_level)
        obj = _read_valid_cache(cache_path, fs, filtered)
    return obj["strat_folds"]


def ptbxl_split_indices(DATA_PATH, filtered=True, fs=500,
                        label_scheme="rhythm", diagnostic_level="superclass"):
    """Train/val/test positions into the data_list, by PTB-XL folds
    (1-8 train, 9 val, 10 test). NOTE: these index the full data_list; for the
    beat models use data.sph_beats.ptbxl_beat_split_indices, which is robust to
    beat-extraction skips."""
    folds = load_ptbxl_strat_folds(DATA_PATH, filtered, fs, label_scheme, diagnostic_level)
    train = [i for i, f in enumerate(folds) if 1 <= f <= 8]
    val   = [i for i, f in enumerate(folds) if f == 9]
    test  = [i for i, f in enumerate(folds) if f == 10]
    print(f"[ptbxl-{label_scheme}] split: train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test