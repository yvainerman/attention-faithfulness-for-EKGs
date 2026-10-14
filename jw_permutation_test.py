"""
jw_permutation_test.py
----------------------
Jain & Wallace (2019) attention-permutation test at the FINAL attention layer (paper P4).
For each test instance: shuffle the trained attention weights over the SAME
encoded positions (N_PERM times), re-run attention-sum + classifier, and measure
how much the output moves. Small change at high max-attention = the prediction
does not depend on WHERE the weight sits.

Env: ECG_TEST_SCHEMES, ECG_SEEDS, ECG_FLAT_VARIANTS, ECG_HIER_VARIANTS,
     ECG_ATTENTION_MODE, ECG_N_SAMPLES, ECG_N_PERM (100), ECG_PERM_THRESHOLD (0.1)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

import faithfulness_common as fc

TEST = "jw_permutation"
N_PERM = int(os.environ.get("ECG_N_PERM", "100"))
PERM_THR = float(os.environ.get("ECG_PERM_THRESHOLD", "0.1"))


def instance_row(model, inst, rng):
    row = inst.info()
    N = inst.N
    perms = np.stack([rng.permutation(N) for _ in range(N_PERM)])             # (K,N)
    A = inst.alphas[torch.as_tensor(perms, device=fc.DEVICE)]                   # (K,N)
    cmp = fc.compare_outputs(inst.p, fc.head_probs(model, inst.features, A), inst.top)
    row.update({
        "med_tvd_top": float(cmp["tvd_top"].median()),
        "med_tvd_max": float(cmp["tvd_max"].median()),
        "mean_tvd_top": float(cmp["tvd_top"].mean()),
        "set_change_rate": float(cmp["set_changed"].float().mean()),
        "top_flip_rate": float(cmp["top_flipped"].float().mean()),
    })
    return row


def run_model(data, kind, variant, seed):
    scheme = data["scheme"]
    model, path = fc.load_model(kind, variant, scheme, seed, data["num_classes"])
    if model is None:
        print(f"  !! missing {path} -- skipped")
        return None
    for w in fc.check_run_record(kind, variant, scheme, fc.ATTENTION_MODE == "uniform"):
        print(f"  [warn] {fc.pretty(kind, variant)}: {w}")
    rng = np.random.RandomState(2000 + seed)
    rows = [instance_row(model, inst, rng) for inst in fc.iter_instances(data, model) if inst.N >= 2]
    df = pd.DataFrame(rows)
    out = fc.csv_path(TEST, scheme, kind, variant, seed)
    df.to_csv(out, index=False)
    if len(df):
        print(f"  {fc.pretty(kind, variant):8s} seed {seed}: n={len(df)}  "
              f"median TVD(top)={df.med_tvd_top.median():.4f}  inv_tvd={(df.med_tvd_top < PERM_THR).mean():.3f}  "
              f"inv_set={(df.set_change_rate < 0.5).mean():.3f}  -> {out}", flush=True)
    return out


def compute():
    for scheme in fc.SCHEMES_TO_TEST:
        loaded = fc.load_scheme(scheme)
        for kind, variant in fc.COMBOS:
            for seed in fc.SEEDS_TO_TEST:
                run_model(loaded[kind], kind, variant, seed)
        del loaded


# =====================================================
# Summary
# =====================================================
def model_summary(frames):
    fs = list(frames.values())
    hi = [(f["max_attn"] > 0.5).values for f in fs]
    return {
        "n": len(fs[0]), "n_seeds": len(fs),
        "median_tvd_top": fc.seed_stat([f.med_tvd_top.values for f in fs], stat=np.nanmedian),
        "median_tvd_max": fc.seed_stat([f.med_tvd_max.values for f in fs], stat=np.nanmedian),
        "inv_tvd": fc.seed_stat([(f.med_tvd_top < PERM_THR).astype(float).values for f in fs]),
        "inv_set": fc.seed_stat([(f.set_change_rate < 0.5).astype(float).values for f in fs]),
        "mean_set_change_rate": fc.seed_stat([f.set_change_rate.values for f in fs]),
        "n_maxattn_gt_0.5": float(np.mean([h.sum() for h in hi])),
        "median_tvd_top_maxattn_gt_0.5": fc.seed_stat(
            [np.where(h, f.med_tvd_top.values, np.nan) for f, h in zip(fs, hi)],
            stat=np.nanmedian, n_boot=0)["mean"],
    }


def plot_scheme(scheme, cells, i2l):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10")
    fig, axes, axd = fc.scheme_pair_grid(f"J&W permutation - {scheme}: max attention vs median "
                                         f"TVD of the top label over {N_PERM} permutations")
    for key, ax in axd.items():
        if key in cells:
            df = pd.concat(cells[key].values())
            ax.scatter(df.max_attn, df.med_tvd_top, s=4, alpha=0.4, edgecolors="none",
                       c=[cmap(int(t) % 10) for t in df["top"]])
            present = sorted(int(t) for t in df["top"].unique())[:6]
            ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=cmap(t % 10),
                                      label=i2l.get(t, str(t))) for t in present],
                      fontsize=5, loc="upper right", framealpha=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.grid(alpha=0.3)
        ax.set_xlabel(r"max attn $\hat\alpha$", fontsize=6)
        ax.set_ylabel("median TVD (top label)", fontsize=6)
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"jw_permutation_{scheme}_{fc.ATTENTION_MODE}.png"))


def summarize():
    payload = {"test": "Jain & Wallace attention permutation, multi-label",
               "mode": fc.ATTENTION_MODE, "n_perm": N_PERM, "tvd_threshold": PERM_THR,
               "distance": "Bernoulli TVD of the top label (tvd_top) and max over labels (tvd_max)",
               "stat_format": "mean over seeds, sd_seeds, 95% bootstrap CI over test instances",
               "results": []}
    for scheme in fc.SCHEMES_TO_TEST:
        print(f"\n=== {scheme}")
        cells = {}
        for kind, variant in fc.COMBOS:
            frames = fc.load_seed_frames(TEST, scheme, kind, variant)
            if not frames:
                print(f"  {fc.pretty(kind, variant):8s} (no CSVs)")
                continue
            cells[(kind, variant)] = frames
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                s = model_summary(frames)
            print(f"  {fc.pretty(kind, variant):8s} n={s['n']} seeds={s['n_seeds']} | "
                  f"median TVD(top) {fc.fmt(s['median_tvd_top'])} | inv_tvd {fc.fmt(s['inv_tvd'])} | "
                  f"inv_set {fc.fmt(s['inv_set'])}")
            payload["results"].append({"scheme": scheme, "model": fc.pretty(kind, variant),
                                       "kind": kind, "variant": variant, **s})
        if cells:
            plot_scheme(scheme, cells, fc.load_labels(scheme))
    fc.write_summary(TEST, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    fc.print_settings("jw_permutation_test.py  (Jain & Wallace permutation, multi-label)",
                      f"n_perm={N_PERM}  tvd_threshold={PERM_THR}")
    if not args.summarize:
        compute()
    if not args.no_summary:
        summarize()


if __name__ == "__main__":
    main()

