"""
data/sph_beats.py
-----------------
Beat-level data layer for the HIERARCHICAL models, PTB-XL only. (The file name
is historical; it is kept so the existing imports keep working.)

WHAT IT DOES
  1. Loads the cleaned PTB-XL signals from data/ptbxl.py (wavelet-denoised,
     baseline-removed, per-lead z-scored -- see that file).
  2. Finds R-peaks (data/utils/r_peak_processing.py: Pan-Tompkins on each lead,
     kept when >= 6 of 12 leads agree) and cuts a 300-sample (600 ms) window
     around each one: 99 samples before the peak, 201 after.
  3. Caches the result next to the signal cache, one file per label scheme:
       ptbxl_rhythm_beats.pkl   (+ ptbxl_rhythm_beats.pkl.meta, a small sidecar
                                 with beat counts and folds, used for the split)
     Each cache stores the settings it was built with (BEAT_CACHE_VERSION,
     the signal cache's CACHE_VERSION and preprocessing, fs, filtered, window
     size); a cache built with other settings is rebuilt instead of loaded. Bump BEAT_CACHE_VERSION after
     changing how beats are cut, to force a rebuild.

Each item in beats_list is a 4-tuple: (beats[S,300,12], label_multihot,
patient_id, rr[S]). rr (RR intervals in seconds) is stored but not used by the
current models.

Normalisation happens ONCE, in data/ptbxl.py (per-lead z-score over the whole
recording), so flat and hierarchical models see identically scaled signals.
The beats are not re-standardised here.

FOLDS: each beat record carries its PTB-XL strat_fold, so
ptbxl_beat_split_indices() gives a correct split even when beat extraction skips
some records (folds 1-8 train, 9 val, 10 test).
"""
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from data.utils.r_peak_processing import extract_beats_multi_lead
from data.ptbxl import PREP_TAG as PTBXL_PREP_TAG, CACHE_VERSION as SIGNAL_CACHE_VERSION, _scheme_name


BEAT_CACHE_VERSION = 1   # stored INSIDE each beat cache; bump to force a rebuild


def _atomic_dump(obj, path):
    """Write a pickle atomically (temp file in the same dir, then os.replace), so
    a concurrent reader never sees a half-written file. os.replace is atomic on
    the same filesystem."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _beat_settings(fs, filtered, pre_samples, post_samples):
    """Everything a beat cache must have been built with to be reused as-is."""
    return {"version": BEAT_CACHE_VERSION, "signal_version": SIGNAL_CACHE_VERSION,
            "preprocessing": PTBXL_PREP_TAG, "fs": fs, "filtered": bool(filtered),
            "pre_samples": pre_samples, "post_samples": post_samples}


def _beat_cache_path(DATA_PATH, label_scheme, diagnostic_level, filtered):
    """e.g. <DATA_PATH>/ptbxl_rhythm_beats.pkl (named like the signal cache)."""
    name = _scheme_name(label_scheme, diagnostic_level, filtered)
    return os.path.join(DATA_PATH, f"ptbxl_{name}_beats.pkl")


def _meta_from(beats_list, beat_folds, settings):
    """Small per-record metadata (cheap to load) used to compute the split
    without re-reading the full multi-GB beat cache."""
    return {"num_beats": [item[0].shape[0] for item in beats_list],
            "folds": beat_folds, "settings": settings}


# =====================================================
# Cached beat extraction
# =====================================================
def load_sph_beats(DATA_PATH, filtered=True, fs=500, pre_samples=99,
                   post_samples=201, force_reload=False, dataset="ptbxl",
                   label_scheme="rhythm", diagnostic_level="superclass"):
    """
    Load (and cache) the beat-segmented PTB-XL dataset.

    Parameters
    ----------
    dataset : "ptbxl"  (kept as an argument so existing calls keep working)
    label_scheme : "diagnostic" | "rhythm" | "form"
    diagnostic_level : "superclass" | "subclass" | "statement" (diagnostic only)

    Returns
    -------
    beats_list : list of (beats, label, patient_id, rr)
    num_classes: int
    label2idx  : dict
    """
    if dataset != "ptbxl":
        raise ValueError(f"only dataset='ptbxl' is supported, got {dataset!r}")

    cache_path = _beat_cache_path(DATA_PATH, label_scheme, diagnostic_level, filtered)
    meta_path  = cache_path + ".meta"
    settings   = _beat_settings(fs, filtered, pre_samples, post_samples)
    tag = f"ptbxl:{label_scheme}"

    if os.path.exists(cache_path) and not force_reload:
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        if obj.get("settings") != settings:
            print(f"[beats:{tag}] {cache_path}: built with {obj.get('settings')}, "
                  f"need {settings}; rebuilding")
        elif not obj.get("beats_list"):
            print(f"[beats:{tag}] {cache_path} is EMPTY; rebuilding")
        else:
            print(f"[beats:{tag}] Loaded {cache_path}: {len(obj['beats_list'])} ECGs, "
                  f"{obj['num_classes']} classes, preprocessing={PTBXL_PREP_TAG}")
            # Self-heal: make sure the split sidecar exists and matches.
            meta_ok = False
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    meta_ok = pickle.load(f).get("settings") == settings
            if not meta_ok:
                _atomic_dump(_meta_from(obj["beats_list"], obj["folds"], settings), meta_path)
            return obj["beats_list"], obj["num_classes"], obj["label2idx"]

    # ---- Cleaned signals + per-record folds (from data/ptbxl.py) ----
    from data.ptbxl import load_ptbxl_dataset, load_ptbxl_strat_folds
    data_list, num_classes, label2idx = load_ptbxl_dataset(
        DATA_PATH, filtered=filtered, fs=fs,
        label_scheme=label_scheme, diagnostic_level=diagnostic_level)
    folds_aligned = load_ptbxl_strat_folds(
        DATA_PATH, filtered=filtered, fs=fs,
        label_scheme=label_scheme, diagnostic_level=diagnostic_level)
    if len(folds_aligned) != len(data_list):
        raise RuntimeError(f"[beats:{tag}] folds ({len(folds_aligned)}) and data_list "
                           f"({len(data_list)}) misaligned -- cache mismatch?")

    print(f"[beats:{tag}] Building {cache_path} from {len(data_list)} ECGs "
          f"(fs={fs}, pre={pre_samples}, post={post_samples})...")
    beats_list, beat_folds, skipped = [], [], 0
    for i, (ecg, label, patient_id) in enumerate(data_list):
        if (i + 1) % 2000 == 0:
            print(f"[beats:{tag}]   {i+1}/{len(data_list)}")
        beats, rr = extract_beats_multi_lead(
            ecg, fs=fs, pre_samples=pre_samples, post_samples=post_samples)
        if beats is None:
            skipped += 1
            continue
        beats_list.append((beats, label, patient_id, rr))
        beat_folds.append(folds_aligned[i])      # fold carried alongside this beat record

    print(f"[beats:{tag}] Kept {len(beats_list)} ECGs, skipped {skipped} (too few valid beats).")
    if not beats_list:
        raise RuntimeError(f"[beats:{tag}] no beats extracted from any record; cache NOT saved")

    obj = {"beats_list": beats_list, "num_classes": num_classes,
           "label2idx": label2idx, "label_scheme": label_scheme,
           "folds": beat_folds, "settings": settings}
    _atomic_dump(obj, cache_path)
    _atomic_dump(_meta_from(beats_list, beat_folds, settings), meta_path)
    print(f"[beats:{tag}] Saved {cache_path}")
    return beats_list, num_classes, label2idx


# =====================================================
# Robust PTB-XL split (indices INTO the min_beats-filtered HANBeatsDataset)
# =====================================================
def ptbxl_beat_split_indices(DATA_PATH, filtered=True, fs=500,
                             label_scheme="rhythm", diagnostic_level="superclass",
                             min_beats=2,
                             train_folds=(1, 2, 3, 4, 5, 6, 7, 8),
                             val_folds=(9,), test_folds=(10,),
                             pre_samples=99, post_samples=201):
    """Train/val/test indices into the beat dataset, by PTB-XL fold.

    Reads only the small .meta sidecar (building or rebuilding the beat cache
    first if the sidecar is missing or out of date), applies the SAME
    `>= min_beats` filter HANBeatsDataset uses, and returns positions into the
    resulting dataset. min_beats MUST match the value passed to HANBeatsDataset.
    """
    cache_path = _beat_cache_path(DATA_PATH, label_scheme, diagnostic_level, filtered)
    meta_path  = cache_path + ".meta"
    settings   = _beat_settings(fs, filtered, pre_samples, post_samples)

    meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    if meta is None or meta.get("settings") != settings:
        load_sph_beats(DATA_PATH, filtered=filtered, fs=fs, pre_samples=pre_samples,
                       post_samples=post_samples, dataset="ptbxl",
                       label_scheme=label_scheme, diagnostic_level=diagnostic_level)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    num_beats, folds = meta["num_beats"], meta["folds"]
    if folds is None or len(folds) != len(num_beats):
        raise RuntimeError(f"{meta_path} has no aligned folds; delete the beat cache "
                           f"(+ .meta) so it is rebuilt.")

    # Replicate HANBeatsDataset(min_beats=...) ordering exactly.
    kept_folds = [f for n, f in zip(num_beats, folds) if n >= min_beats]

    train = [k for k, f in enumerate(kept_folds) if f in train_folds]
    val   = [k for k, f in enumerate(kept_folds) if f in val_folds]
    test  = [k for k, f in enumerate(kept_folds) if f in test_folds]
    print(f"[beats:ptbxl:{label_scheme}] split (dataset idx, min_beats={min_beats}): "
          f"train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


# =====================================================
# Dataset (variable number of beats per recording)
# =====================================================
class HANBeatsDataset(Dataset):
    """Keeps ALL beats of every recording with >= min_beats beats; exposes the
    per-recording beat counts for LengthBucketBatchSampler (which batches
    recordings with equal beat counts, so no padding/masking is needed).
    `num_beats` must stay None (kept as an argument so existing calls work)."""
    def __init__(self, beats_list, num_beats=None, min_beats=2):
        if num_beats is not None:
            raise ValueError("fixed num_beats mode is not supported; pass num_beats=None")
        self.data = [item for item in beats_list if item[0].shape[0] >= min_beats]
        self.num_beats = [item[0].shape[0] for item in self.data]

        kept, total = len(self.data), len(beats_list)
        if kept < total:
            print(f"[HANBeatsDataset] kept {kept}/{total} ECGs (min_beats={min_beats})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        beats, label, _patient_id, rr = self.data[idx]
        return {"beats": torch.from_numpy(np.asarray(beats, dtype=np.float32)),   # (S,300,12)
                "rr":    torch.from_numpy(np.asarray(rr, dtype=np.float32)),      # (S,)
                "label": torch.as_tensor(np.asarray(label, dtype=np.float32))}    # (C,)


def han_collate(batch):
    return {"beats": torch.stack([item["beats"] for item in batch], dim=0),   # (B,S,300,12)
            "rr":    torch.stack([item["rr"]    for item in batch], dim=0),   # (B,S)
            "label": torch.stack([item["label"] for item in batch], dim=0)}   # (B,C)


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, drop_last=False, shuffle=True, seed=None):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self._rng = random.Random(seed) if seed is not None else random
        self.buckets = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            self.buckets[length].append(idx)

    def __iter__(self):
        all_batches = []
        for length, indices in self.buckets.items():
            local = list(indices)
            if self.shuffle:
                self._rng.shuffle(local)
            for i in range(0, len(local), self.batch_size):
                batch = local[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            self._rng.shuffle(all_batches)
        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.buckets.values():
            n = len(indices)
            total += n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        return total