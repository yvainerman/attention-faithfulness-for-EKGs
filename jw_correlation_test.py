"""
jw_correlation_test.py
----------------------
Jain & Wallace correlation test at the FINAL attention layer.

For each test instance, Kendall's tau measures agreement between the
rankings of the N positions (flat model: 100-ms PatchCNN tokens;
hierarchical model: individual heartbeats):

   attention  Final attention weights (alpha).

   gradient   Magnitude of the gradient of the top label's sigmoid
	probability with respect to the input at each position.
	Attention weights are held fixed during this calculation.

   LOO        Change in the model's output probability when one input
	position is deleted and the entire model is run again,
	allowing attention weights to be recalculated.

J&W's finding to test: gradient and LOO rankings agree more strongly
with each other than either agrees with the attention ranking:
tau(gradient, LOO) > tau(attention, LOO)..

ENVIRONMENT VARIABLES
ECG_TEST_SCHEMES, ECG_SEEDS, ECG_FLAT_VARIANTS, ECG_HIER_VARIANTS,
ECG_ATTENTION_MODE, ECG_N_SAMPLES, ECG_LOO_CHUNK (flat LOO batch size,
default 100), ECG_N_BOOT
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import faithfulness_common as fc

TEST = "jw_correlation"


def _tau(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, p = kendalltau(a, b)
    return float(t), float(p)


def instance_row(kind, model, inst):
    """ONE EKG -> one csv row."""
    row = inst.info()
    alpha = inst.alphas.detach().cpu().numpy()
    g_l2, g_gxi = fc.input_gradient_importance(kind, model, inst)
    cmp = fc.compare_outputs(inst.p, fc.loo_probs(kind, model, inst), inst.top)
    loo_top = cmp["tvd_top"].cpu().numpy()          # |p_top - p'_top| for each deleted position
    loo_max = cmp["tvd_max"].cpu().numpy()          # largest change over all labels

    row["tau_attn_grad"], row["p_attn_grad"] = _tau(alpha, g_l2)
    row["tau_attn_gxi"], _ = _tau(alpha, g_gxi)
    row["tau_attn_loo"], row["p_attn_loo"] = _tau(alpha, loo_top)
    row["tau_grad_loo"], row["p_grad_loo"] = _tau(g_l2, loo_top)
    row["tau_gxi_loo"], _ = _tau(g_gxi, loo_top)
    row["tau_attn_loomax"], _ = _tau(alpha, loo_max)
    row["tau_grad_loomax"], _ = _tau(g_l2, loo_max)
    row["loo_top_mean"] = float(loo_top.mean())
    row["loo_set_changes"] = int(cmp["set_changed"].sum())
    return row


def run_model(data, kind, variant, seed):
    """Load a trained model, evaluate its attention-importance correlations,
    and save the per-instance results to a CSV file."""
    scheme = data["scheme"]
    model, path = fc.load_model(kind, variant, scheme, seed, data["num_classes"])
    if model is None:
        print(f"  !! missing {path} -- skipped")
        return None
    for w in fc.check_run_record(kind, variant, scheme, fc.ATTENTION_MODE == "uniform"):
        print(f"  [warn] {fc.pretty(kind, variant)}: {w}")
    rows = [instance_row(kind, model, inst)
            for inst in fc.iter_instances(data, model) if inst.N >= 2]
    df = pd.DataFrame(rows)
    out = fc.csv_path(TEST, scheme, kind, variant, seed)
    df.to_csv(out, index=False)
    if len(df):
        print(f"  {fc.pretty(kind, variant):8s} seed {seed}: n={len(df)}  "
              f"tau(a,g)={df.tau_attn_grad.mean():+.3f}  tau(a,LOO)={df.tau_attn_loo.mean():+.3f}  "
              f"tau(g,LOO)={df.tau_grad_loo.mean():+.3f}  -> {out}", flush=True)
    return out


def compute():
    """Run the correlation test for every selected dataset scheme, model
    variant, and training seed."""
    for scheme in fc.SCHEMES_TO_TEST:
        loaded = fc.load_scheme(scheme)
        for kind, variant in fc.COMBOS:
            for seed in fc.SEEDS_TO_TEST:
                run_model(loaded[kind], kind, variant, seed)
        del loaded


# SUMMARY
def model_summary(frames):
    """Calculate summary stats for attention, grad, and LOO correlations
    across training seed and test instances."""
    fs = list(frames.values())
    cols = [c for c in fs[0].columns if c.startswith("tau_")]
    out = {"n": len(fs[0]), "n_seeds": len(fs), "mean_tau": {}, "median_tau": {}}
    for c in cols:
        out["mean_tau"][c] = fc.seed_stat([f[c].values for f in fs])
        out["median_tau"][c] = fc.seed_stat([f[c].values for f in fs], stat=np.nanmedian, n_boot=0)["mean"]
    out["delta_gradloo_minus_attnloo"] = fc.seed_stat(
        [(f["tau_grad_loo"] - f["tau_attn_loo"]).values for f in fs])
    out["frac_significant_attn_loo"] = fc.seed_stat(
        [(f["p_attn_loo"] < 0.05).astype(float).values for f in fs], n_boot=0)["mean"]
    out["frac_significant_attn_grad"] = fc.seed_stat(
        [(f["p_attn_grad"] < 0.05).astype(float).values for f in fs], n_boot=0)["mean"]
    out["nan_tau_attn_loo"] = int(sum(f["tau_attn_loo"].isna().sum() for f in fs))
    return out


def plot_scheme(scheme, cells):
    """Plot the distributions of Kendall tau correlations for each variant."""
    fig, axes, axd = fc.scheme_pair_grid(f"J&W correlation - {scheme}: Kendall tau "
                                         f"(a=attention, g=gradient, l=leave-one-out)")
    labels = [r"$\tau$(a,g)", r"$\tau$(a,l)", r"$\tau$(g,l)"]
    cols = ["tau_attn_grad", "tau_attn_loo", "tau_grad_loo"]
    for key, ax in axd.items():
        if key in cells:
            frames = cells[key]
            data = [pd.concat([f[c] for f in frames.values()]).dropna().values for c in cols]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False, tick_labels=labels, widths=0.6)
            for patch, c in zip(bp["boxes"], ["#a3c2db", "#d4e3f0", "#5e8eb5"]):
                patch.set_facecolor(c)
        ax.axhline(0, color="red", lw=0.7, ls="--")
        ax.set_ylim(-1, 1); ax.grid(alpha=0.3, axis="y")
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"jw_correlation_{scheme}_{fc.ATTENTION_MODE}.png"))


def summarize():
    payload = {"test": "Jain & Wallace correlation (final attention), multi-label",
               "mode": fc.ATTENTION_MODE,
               "gradient_target": "sigmoid probability of the top label, all attention frozen",
               "loo": "delete the position (J&W), re-run the whole model",
               "loo_distance": "|p_top - p'_top| (Bernoulli TVD of the top label); 'loomax' = max over labels",
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
            if "tau_attn_loo" not in next(iter(frames.values())).columns:
                print(f"  {fc.pretty(kind, variant):8s} CSVs are from the old multi-baseline version "
                      f"-- re-run the compute step")
                continue
            cells[(kind, variant)] = frames
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                s = model_summary(frames)
            m = s["mean_tau"]
            print(f"  {fc.pretty(kind, variant):8s} n={s['n']} seeds={s['n_seeds']} | "
                  f"tau(a,g) {fc.fmt(m['tau_attn_grad'])} | tau(a,l) {fc.fmt(m['tau_attn_loo'])} | "
                  f"tau(g,l) {fc.fmt(m['tau_grad_loo'])} | "
                  f"delta {fc.fmt(s['delta_gradloo_minus_attnloo'])}")
            payload["results"].append({"scheme": scheme, "model": fc.pretty(kind, variant),
                                       "kind": kind, "variant": variant, **s})
        if cells:
            plot_scheme(scheme, cells)
    fc.write_summary(TEST, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    fc.print_settings("jw_correlation_test.py  (Jain & Wallace correlation, multi-label)",
                      "leave-one-out: delete each position (J&W)")
    if not args.summarize:
        compute()
    if not args.no_summary:
        summarize()


if __name__ == "__main__":
    main()
