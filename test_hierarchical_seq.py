"""
test_hierarchical_seq.py
------------------------
Evaluates the trained MULTI-LABEL HIERARCHICAL models (models/hierarchical_seq.py)
on the PTB-XL TEST fold (fold 10) -- any subset of the 4 variants x 3 label
schemes in one run. The hierarchical counterpart to test_flat_seq.py.

Mirrors train_hierarchical_seq.py exactly (everything below is IMPORTED from it,
so the test can never drift from training):
  * architecture (ARCH), FS, the beat window / min_beats, the data path, and the
    uniform-attention patch (sequence attention only);
  * the same beats cut from the cleaned signals (data/sph_beats.py on top of
    data/ptbxl.py; preprocessing tag printed) and the same fold split (folds 1-8
    train, 9 val, 10 TEST; each beat record carries its own fold);
  * multi-label decoding: per-class sigmoid; macro-AUC (threshold-free, primary),
    plus F1 / exact-match / Hamming loss at THRESHOLD.

Every per-seed checkpoint is evaluated and reported as mean +/- std over seeds
(the trainer saves best_{variant}_{scheme}_filtered_seed{N}.pth for each seed).
Each checkpoint's _meta.json run record is checked first: a checkpoint trained
with different preprocessing or the wrong attention mode is flagged.

SETTINGS (env vars)
  ECG_TEST_UNIFORM   0 (default) -> checkpoints/seq/          (normal models)
                     1           -> checkpoints/seq_uniform/  (W&P uniform models)
  ECG_TEST_SCHEMES   comma list, default diagnostic,rhythm,form
  ECG_TEST_VARIANTS  comma list, default noenc,conv,bilstm_cnn,bilstm_full
  ECG_DIAG_LEVEL, ECG_USE_FILTERED, ECG_DATA_ROOT, ECG_NUM_WORKERS  (as in training)
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss

from data.ptbxl import check_data_path, PREP_TAG
from data.sph_beats import (
    load_sph_beats, HANBeatsDataset, han_collate, LengthBucketBatchSampler,
    ptbxl_beat_split_indices,
)
from models.hierarchical_seq import HierarchicalSeqAttention

# AUTO-SYNC: architecture, beat constants, data path and the uniform patch all
# come from the trainer.
from train_hierarchical_seq import (
    ARCH, FS, PRE_SAMPLES, POST_SAMPLES, MIN_BEATS, SEEDS, resolve_data_path,
    apply_uniform_seq,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =====================================================
# CONFIG
# =====================================================
def _env_list(name, default):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else list(default)

def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() not in ("0", "false", "no")

UNIFORM          = _env_bool("ECG_TEST_UNIFORM", False)
VARIANTS_TO_TEST = _env_list("ECG_TEST_VARIANTS", HierarchicalSeqAttention.VARIANTS)
SCHEMES_TO_TEST  = _env_list("ECG_TEST_SCHEMES", ("diagnostic", "rhythm", "form"))
DIAG_LEVEL       = os.environ.get("ECG_DIAG_LEVEL", "superclass")
USE_FILTERED     = _env_bool("ECG_USE_FILTERED", True)
NUM_WORKERS      = int(os.environ.get("ECG_NUM_WORKERS", "0"))
PRINT_PER_CLASS  = True       # per-class AUC / F1 / support table
THRESHOLD        = 0.5        # decision threshold for F1 / accuracy (not AUC)
BATCH_SIZE       = 64

CKPT_DIR = "checkpoints/seq_uniform" if UNIFORM else "checkpoints/seq"
OUT_DIR  = "test_results"
TAG      = "hier_uniform" if UNIFORM else "hier"


# =====================================================
# Test loader (one scheme at a time -- all variants share a scheme's test set;
# the previous scheme's data is released to keep memory low)
# =====================================================
_loader_cache = {}

def get_test_loader(data_path, scheme):
    if scheme in _loader_cache:
        return _loader_cache[scheme]
    _loader_cache.clear()
    print(f"[loader] PTB-XL test fold for scheme={scheme} (level={DIAG_LEVEL}, "
          f"preprocessing={PREP_TAG})...", flush=True)
    beats_list, num_classes, label2idx = load_sph_beats(
        data_path, filtered=USE_FILTERED, fs=FS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES,
        dataset="ptbxl", label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    dataset = HANBeatsDataset(beats_list, num_beats=None, min_beats=MIN_BEATS)
    _, _, test_idx = ptbxl_beat_split_indices(
        data_path, filtered=USE_FILTERED, fs=FS,
        label_scheme=scheme, diagnostic_level=DIAG_LEVEL, min_beats=MIN_BEATS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES)
    test_lengths = [dataset.num_beats[i] for i in test_idx]
    sampler = LengthBucketBatchSampler(test_lengths, BATCH_SIZE,      # equal beat counts per batch
                                       drop_last=False, shuffle=False)
    loader = DataLoader(Subset(dataset, test_idx), batch_sampler=sampler,
                        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=han_collate)
    print(f"[loader]   {len(dataset.data)} ECGs (min_beats={MIN_BEATS}), test={len(test_idx)}, "
          f"classes={num_classes}", flush=True)
    _loader_cache[scheme] = (loader, label2idx, num_classes)
    return _loader_cache[scheme]


def get_checkpoints(variant, scheme):
    suffix = "_filtered" if USE_FILTERED else ""
    stem = f"{CKPT_DIR}/best_{variant}_{scheme}{suffix}"
    return stem, sorted(glob.glob(f"{stem}_seed*.pth"))


def check_meta(stem):
    """Compare the run record the trainer wrote with what this test expects.
    Returns a list of warnings (empty = all consistent)."""
    path = f"{stem}_meta.json"
    if not os.path.exists(path):
        return [f"no run record {os.path.basename(path)} (trained by an older trainer?)"]
    meta = json.load(open(path))
    warn = []
    if meta.get("preprocessing") != PREP_TAG:
        warn.append(f"trained with preprocessing={meta.get('preprocessing')!r}, "
                    f"testing on {PREP_TAG!r}")
    if bool(meta.get("uniform_attention")) != UNIFORM:
        warn.append(f"run record says uniform_attention={meta.get('uniform_attention')}, "
                    f"but this test expects {UNIFORM}")
    if meta.get("arch") != ARCH[meta.get("variant")]:
        warn.append(f"trained with arch={meta.get('arch')}, trainer now has "
                    f"{ARCH[meta.get('variant')]}")
    return warn


def build_eval_model(variant, num_classes):
    model = HierarchicalSeqAttention(num_classes=num_classes, variant=variant,
                                     **ARCH[variant]).to(DEVICE)
    if UNIFORM:
        model = apply_uniform_seq(model)   # same 1/S sequence attention it was trained with
    return model


# =====================================================
# Multi-label evaluation
# =====================================================
@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses, ys, ps = [], [], []
    for batch in loader:
        beats  = batch["beats"].to(DEVICE)
        labels = batch["label"].to(DEVICE).float()
        logits = model(beats)
        losses.append(criterion(logits, labels).item())
        ys.append(labels.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    pred = (p >= THRESHOLD).astype(int)

    per_class_auc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        yc = y[:, c]
        if yc.min() != yc.max():
            per_class_auc[c] = roc_auc_score(yc, p[:, c])
    macro_auc = float(np.nanmean(per_class_auc)) if np.isfinite(per_class_auc).any() else float("nan")

    return {
        "loss":          float(np.mean(losses)),
        "macro_auc":     macro_auc,
        "f1_macro":      float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_micro":      float(f1_score(y, pred, average="micro", zero_division=0)),
        "f1_weighted":   float(f1_score(y, pred, average="weighted", zero_division=0)),
        "exact_acc":     float((pred == y).all(axis=1).mean()),
        "hamming_loss":  float(hamming_loss(y, pred)),          # 0=best, 1=worst
        "per_class_auc": per_class_auc,
        "per_class_f1":  f1_score(y, pred, average=None, zero_division=0),
        "support":       y.sum(axis=0).astype(int),
    }


SCALAR_KEYS = ("loss", "macro_auc", "f1_macro", "f1_micro", "f1_weighted", "exact_acc", "hamming_loss")


def run_one_combo(data_path, scheme, variant):
    stem, ckpts = get_checkpoints(variant, scheme)
    print(f"\n{'#'*72}\n# {scheme} / {variant}{'  (UNIFORM)' if UNIFORM else ''}\n{'#'*72}",
          flush=True)
    if not ckpts:
        print(f"  !! No checkpoints matched {stem}_seed*.pth. Skipping.")
        return None
    for w in check_meta(stem):
        print(f"  !! WARNING: {w}")
    if len(ckpts) != len(SEEDS):
        print(f"  !! WARNING: found {len(ckpts)} seed checkpoint(s), trainer runs {len(SEEDS)} "
              f"({SEEDS}) -- training may not have finished")
    loader, label2idx, num_classes = get_test_loader(data_path, scheme)
    print(f"  Evaluating {len(ckpts)} checkpoint(s).")

    results = []
    for path in ckpts:
        model = build_eval_model(variant, num_classes)
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        except Exception as e:
            print(f"  !! Could not load {os.path.basename(path)}: {e}")
            continue
        m = evaluate(model, loader, num_classes)
        m["ckpt"] = os.path.basename(path)
        print(f"    {m['ckpt']:<48s}  macroAUC={m['macro_auc']:.4f}  "
              f"macroF1={m['f1_macro']:.4f}  microF1={m['f1_micro']:.4f}  "
              f"weightedF1={m['f1_weighted']:.4f}  "
              f"exactAcc={m['exact_acc']:.4f}  hammingLoss={m['hamming_loss']:.4f}", flush=True)
        results.append(m)
    if not results:
        return None

    idx2label = {v: k for k, v in label2idx.items()}
    pca = np.nanmean(np.stack([r["per_class_auc"] for r in results]), axis=0)
    pcf = np.mean(np.stack([r["per_class_f1"] for r in results]), axis=0)
    sup = results[0]["support"]
    if PRINT_PER_CLASS:
        print(f"\n  Per-class (mean over {len(results)} seeds), sorted by support:")
        print(f"    {'class':>10}  {'support':>7}  {'AUC':>7}  {'F1@'+str(THRESHOLD):>7}")
        for c in sorted(range(num_classes), key=lambda i: -sup[i]):
            auc_s = "nan" if np.isnan(pca[c]) else f"{pca[c]:.3f}"
            print(f"    {idx2label[c]:>10}  {sup[c]:>7d}  {auc_s:>7}  {pcf[c]:>7.3f}")

    agg = {}
    for k in SCALAR_KEYS:
        vals = np.array([r[k] for r in results if not np.isnan(r[k])])
        agg[k] = (float(vals.mean()), float(vals.std())) if len(vals) else (float("nan"), float("nan"))
    print(f"\n  --- Aggregate ({len(results)} seed{'s' if len(results)!=1 else ''}) ---")
    for k, lbl in [("loss","BCE loss "), ("macro_auc","macro-AUC"), ("f1_macro","macro-F1 "),
                   ("f1_micro","micro-F1 "), ("f1_weighted","weightF1 "),
                   ("exact_acc","exactAcc "), ("hamming_loss","hammingLoss")]:
        mu, sd = agg[k]
        print(f"    {lbl}: {mu:.4f} +/- {sd:.4f}")
    return {"scheme": scheme, "variant": variant, "n_seeds": len(results), "agg": agg,
            "per_seed": [{"ckpt": r["ckpt"], **{k: r[k] for k in SCALAR_KEYS}} for r in results],
            "per_class": {idx2label[c]: {"support": int(sup[c]),
                                         "auc": None if np.isnan(pca[c]) else float(pca[c]),
                                         "f1": float(pcf[c])} for c in range(num_classes)}}


def print_summary(summaries):
    """One concise end-of-run grid: rows grouped by scheme and ordered along the
    reach sweep (noenc -> conv -> bilstm_cnn -> bilstm_full); best macro-AUC per
    scheme marked '*'."""
    done = [s for s in summaries if s is not None]
    order = {v: i for i, v in enumerate(HierarchicalSeqAttention.VARIANTS)}
    def fmt(t):
        return f"{'nan':>14}" if np.isnan(t[0]) else f"{t[0]:.4f}+/-{t[1]:.4f}"

    print(f"\n\n{'='*92}")
    print(f"  FINAL SUMMARY  HIER{' UNIFORM' if UNIFORM else ''} (test fold, multi-label, "
          f"preprocessing={PREP_TAG}, mean+/-std over seeds)")
    print(f"{'='*92}")
    print(f"  {'scheme':<11} {'variant':<13} {'N':>2}  {'macroAUC':>14}  "
          f"{'macroF1':>14}  {'weightF1':>14}  {'microF1':>14}  {'hammingLoss':>14}")
    print(f"  {'-'*11} {'-'*13} {'-'*2}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    winners = []
    for scheme in SCHEMES_TO_TEST:
        rows = sorted([s for s in done if s["scheme"] == scheme],
                      key=lambda s: order.get(s["variant"], 99))
        if not rows:
            continue
        valid = [s for s in rows if not np.isnan(s["agg"]["macro_auc"][0])]
        best = max(valid, key=lambda s: s["agg"]["macro_auc"][0]) if valid else None
        for s in rows:
            a = s["agg"]
            star = "*" if s is best else " "
            print(f"  {scheme:<11} {s['variant']:<13}{star}{s['n_seeds']:>1}  "
                  f"{fmt(a['macro_auc']):>14}  {fmt(a['f1_macro']):>14}  "
                  f"{fmt(a['f1_weighted']):>14}  {fmt(a['f1_micro']):>14}  "
                  f"{fmt(a['hamming_loss']):>14}")
        if best is not None:
            winners.append((scheme, best))
        print(f"  {'-'*11} {'-'*13} {'-'*2}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

    if winners:
        print(f"\n  Best variant per scheme (by macro-AUC):")
        for scheme, b in winners:
            mu, sd = b["agg"]["macro_auc"]
            print(f"    {scheme:<11} -> {b['variant']:<13} macroAUC={mu:.4f}+/-{sd:.4f}")

    n_skip = sum(1 for s in summaries if s is None)
    if n_skip:
        print(f"\n  ({n_skip} combo(s) skipped -- no checkpoints found)")


def main():
    bad_v = [v for v in VARIANTS_TO_TEST if v not in HierarchicalSeqAttention.VARIANTS]
    bad_s = [s for s in SCHEMES_TO_TEST if s not in ("diagnostic", "rhythm", "form")]
    if bad_v: raise ValueError(f"bad variants: {bad_v}")
    if bad_s: raise ValueError(f"bad schemes: {bad_s}")
    data_path = check_data_path(resolve_data_path(), FS)

    print(f"{'='*72}")
    print(f"  test_hierarchical_seq.py  (MULTI-LABEL / HAN, test fold)")
    print(f"  checkpoints  : {CKPT_DIR}   (uniform={UNIFORM})")
    print(f"  schemes      : {SCHEMES_TO_TEST}")
    print(f"  variants     : {VARIANTS_TO_TEST}")
    print(f"  diag_lvl     : {DIAG_LEVEL}   filtered: {USE_FILTERED}   seeds expected: {SEEDS}")
    print(f"  preprocessing: {PREP_TAG}")
    print(f"  data_path    : {data_path}")
    print(f"{'='*72}", flush=True)

    summaries = []
    for scheme in SCHEMES_TO_TEST:            # scheme outer -> reuse the loaded test set
        for variant in VARIANTS_TO_TEST:
            summaries.append(run_one_combo(data_path, scheme, variant))
    print_summary(summaries)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{TAG}_test.json")
    with open(out_path, "w") as f:
        json.dump({"family": "hier", "uniform": UNIFORM, "preprocessing": PREP_TAG,
                   "threshold": THRESHOLD, "diag_level": DIAG_LEVEL,
                   "results": [s for s in summaries if s is not None]}, f, indent=2)
    print(f"\n  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()