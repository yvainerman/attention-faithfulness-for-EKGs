"""
prepare_data.py
---------------
STEP 0: build every PTB-XL cache once, before any tuning / training job starts.

For each label scheme it builds, in the PTB-XL folder:
    ptbxl_{scheme}.pkl             cleaned whole signals  (flat models; data/ptbxl.py)
    ptbxl_{scheme}_beats.pkl(.meta) beats cut from them   (hierarchical; data/sph_beats.py)
(diagnostic is named diagnostic-{level}, e.g. ptbxl_diagnostic-superclass.pkl)
"""
import argparse
import os
import time

import numpy as np

from data.ptbxl import load_ptbxl_dataset, ptbxl_split_indices, check_data_path, PREP_TAG
from data.sph_beats import load_sph_beats, ptbxl_beat_split_indices
from train_hierarchical_seq import (
    resolve_data_path, USE_FILTERED, DIAG_LEVEL, FS, PRE_SAMPLES, POST_SAMPLES, MIN_BEATS,
)

SCHEMES = ("diagnostic", "rhythm", "form")


def prepare(scheme, data_path):
    t0 = time.time()
    print(f"\n{'='*64}\n  {scheme}  (preprocessing={PREP_TAG}, filtered={USE_FILTERED}"
          f"{', level=' + DIAG_LEVEL if scheme == 'diagnostic' else ''})\n{'='*64}", flush=True)

    # 1. cleaned whole signals (flat models)
    data_list, num_classes, label2idx = load_ptbxl_dataset(
        data_path, filtered=USE_FILTERED, fs=FS,
        label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    ptbxl_split_indices(data_path, filtered=USE_FILTERED, fs=FS,
                        label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    sig = np.stack([d[0] for d in data_list[:200]])
    print(f"[check] signals {sig.shape[1:]} ; first 200: per-lead mean "
          f"{abs(sig.mean(-1)).max():.3f} (max |.|), std {sig.std(-1).mean():.3f}")
    n_signals = len(data_list)
    del data_list, sig

    # 2. beats cut from those signals (hierarchical models)
    beats_list, _, _ = load_sph_beats(
        data_path, filtered=USE_FILTERED, fs=FS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES, dataset="ptbxl",
        label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    ptbxl_beat_split_indices(data_path, filtered=USE_FILTERED, fs=FS,
                             label_scheme=scheme, diagnostic_level=DIAG_LEVEL,
                             min_beats=MIN_BEATS,
                             pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES)
    counts = np.array([b[0].shape[0] for b in beats_list])
    print(f"[check] beats per recording: median {int(np.median(counts))}, "
          f"min {counts.min()}, max {counts.max()}; "
          f"{(counts < MIN_BEATS).sum()} recordings below min_beats={MIN_BEATS}")
    print(f"[done] {scheme}: {n_signals} recordings, {len(beats_list)} with beats, "
          f"{num_classes} classes, {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    args = ap.parse_args()
    data_path = check_data_path(resolve_data_path(), FS)   # stops here if PTB-XL is not there
    print(f"code folder : {os.getcwd()}")
    print(f"PTB-XL data : {data_path}   (caches are written here too)")
    print(f"schemes     : {args.schemes}")
    for scheme in args.schemes:
        prepare(scheme, data_path)


if __name__ == "__main__":
    main()