"""
ss_erasure_test.py
------------------
Serrano & Smith (2019) erasure tests at the FINAL attention layer.

Env: ECG_TEST_SCHEMES, ECG_SEEDS, ECG_FLAT_VARIANTS, ECG_HIER_VARIANTS,
     ECG_ATTENTION_MODE, ECG_N_SAMPLES, ECG_THRESHOLD, ECG_N_BOOT
"""
import argparse
import os
import warnings
 
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
 
import faithfulness_common as fc
 
TEST = "ss_erasure"
RANKS = ("random", "attention", "gradient", "product")
RANK_LABELS = ["Random", "Attention", "Gradient", "Attention$\\times$Gradient"]
CRITS = ("set", "top") 	# two ways a decision can flip
LINE_COLORS = ["#9e9e9e", "#6baed6", "#2171b5", "#0D4293"]   # Random in grey on line plots
 
 
# PER-INSTANCE COMPUTATION
def removal_outputs(model, inst, order):
    """Input: One recording and an order (ranking of positions).
    	Output: (N, C), Row k is the model's output after the first k+1 positions
		in the order are erased. Last row has everything erased."""
    N = inst.N
    rank_of = torch.empty(N, dtype=torch.long, device=fc.DEVICE) 		# place in erasure order
    rank_of[torch.as_tensor(order, device=fc.DEVICE)] = torch.arange(N, device=fc.DEVICE)
    keep = (rank_of.unsqueeze(0) > torch.arange(N, device=fc.DEVICE).unsqueeze(1)).float()	 # if rank is > k, position j is present after k+1 erasures.
    A = inst.alphas.unsqueeze(0) * keep
    s = A.sum(dim=1, keepdim=True)
    A = torch.where(s > 0, A / s.clamp_min(1e-30), A)		# renorm rows so they sum to 1 after removal
    return fc.head_probs(model, inst.features, A)
 
 
def first_flip(flags):
    """Return index of row that says 'decision flipped here'. Last row is ignored
    so a flip only counts while some attention is left."""
    hit = np.flatnonzero(flags[:-1])
    return int(hit[0]) if len(hit) else None
 
 
def instance_row(model, inst, rng):
    N, top = inst.N, inst.top
    p0 = inst.p
    a = inst.alphas.detach().cpu().numpy()
    row = inst.info()
 
    # TEST 1: top-attention position vs a random other position (erase random)
    i_star = int(np.argmax(a))						# top attended position
    r = int(rng.choice([j for j in range(N) if j != i_star])) 		# random position
    A = inst.alphas.unsqueeze(0).repeat(2, 1)
    A[0, i_star] = 0.0
    A[1, r] = 0.0
    A = A / A.sum(dim=1, keepdim=True)
    cmp1 = fc.compare_outputs(p0, fc.head_probs(model, inst.features, A), top) 		# refer to faithfulness_summary
    row.update({
        "t1_delta_alpha": float(a[i_star] - a[r]),
        "t1_jsd_top_i": float(cmp1["jsd_top"][0]), "t1_jsd_top_r": float(cmp1["jsd_top"][1]),
        "t1_jsd_max_i": float(cmp1["jsd_max"][0]), "t1_jsd_max_r": float(cmp1["jsd_max"][1]),	
        "t1_set_i": int(cmp1["set_changed"][0]), "t1_set_r": int(cmp1["set_changed"][1]),	
        "t1_top_i": int(cmp1["top_flipped"][0]), "t1_top_r": int(cmp1["top_flipped"][1]),
    })
 
    # TEST 2: ranked removal
    grad = fc.attention_gradient(model, inst)
    orders = {
        "random":    rng.permutation(N),
        "attention": np.argsort(-a, kind="stable"),
        "gradient":  np.argsort(-grad, kind="stable"),
        "product":   np.argsort(-(a * grad), kind="stable"),
    }
    for name, order in orders.items():
        P = removal_outputs(model, inst, order)                     # (N, C)
        cmp = fc.compare_outputs(p0, P, top)
        if name == "random":                                        # terminal row is order-independent
            row["zero_set_flip"] = int(cmp["set_changed"][-1])
            row["zero_top_flip"] = int(cmp["top_flipped"][-1])
            row["zero_top_label"] = int(P[-1].argmax())
        cum_mass = np.cumsum(a[order])
        for crit, key in (("set", "set_changed"), ("top", "top_flipped")):
            k = first_flip(cmp[key].cpu().numpy())
            row[f"k_{crit}_{name}"] = (k + 1) if k is not None else np.nan       # positions removed
            row[f"mass_{crit}_{name}"] = float(cum_mass[k]) if k is not None else np.nan
    return row
 
 
def run_model(data, kind, variant, seed):
    scheme = data["scheme"]
    model, path = fc.load_model(kind, variant, scheme, seed, data["num_classes"])
    if model is None:
        print(f"  !! missing {path} -- skipped")
        return None
    for w in fc.check_run_record(kind, variant, scheme, fc.ATTENTION_MODE == "uniform"):
        print(f"  [warn] {fc.pretty(kind, variant)}: {w}")
    rng = np.random.RandomState(1000 + seed)
    rows = []
    for inst in fc.iter_instances(data, model):
        if inst.N < 2:
            continue
        rows.append(instance_row(model, inst, rng))
    df = pd.DataFrame(rows)
    out = fc.csv_path(TEST, scheme, kind, variant, seed)
    df.to_csv(out, index=False)
    if len(df):
        cov_a = df["k_set_attention"].notna().mean()
        cov_r = df["k_set_random"].notna().mean()
        print(f"  {fc.pretty(kind, variant):8s} seed {seed}: n={len(df)}  "
              f"coverage(set) Attn={cov_a:.1%} Rand={cov_r:.1%}  "
              f"S&S-retained={df['zero_set_flip'].mean():.1%}  -> {out}", flush=True)
    return out
 
 
def compute():
    for scheme in fc.SCHEMES_TO_TEST:
        loaded = fc.load_scheme(scheme)
        for kind, variant in fc.COMBOS:
            for seed in fc.SEEDS_TO_TEST:
                run_model(loaded[kind], kind, variant, seed)
        del loaded
 
 
# SUMMARY
def _win(ka, kb):
    """1 if order A flips strictly earlier than order B, 0.5 tie, 0 later
    (never flipping = infinitely late)."""
    a = np.where(np.isnan(ka), np.inf, ka)
    b = np.where(np.isnan(kb), np.inf, kb)
    return np.where(a < b, 1.0, np.where(a == b, 0.5, 0.0))
 
 
def model_summary(frames, i2l):
    """frames: {seed: DataFrame} aligned by idx."""
    fs = list(frames.values())
    n = len(fs[0])
    out = {"n": n, "n_seeds": len(fs), "mean_positions": float(fs[0]["N"].mean()),
           "frac_empty_prediction": fc.seed_stat([f["n_pred"].eq(0).astype(float).values for f in fs],
                                                 n_boot=0)["mean"]}
    for crit in CRITS:
        c = {}
        for r in RANKS:
            k = [f[f"k_{crit}_{r}"].values.astype(float) for f in fs]
            frac = [kk / f["N"].values for kk, f in zip(k, fs)]
            mass = [f[f"mass_{crit}_{r}"].values.astype(float) for f in fs]
            c[r] = {
                "coverage": fc.seed_stat([(~np.isnan(kk)).astype(float) for kk in k]),
                "median_frac_flippers": fc.seed_stat(frac, stat=np.nanmedian),
                "q1_frac_flippers": fc.seed_stat(frac, stat=lambda v: np.nanpercentile(v, 25), n_boot=0)["mean"],
                "q3_frac_flippers": fc.seed_stat(frac, stat=lambda v: np.nanpercentile(v, 75), n_boot=0)["mean"],
                "median_mass_flippers": fc.seed_stat(mass, stat=np.nanmedian, n_boot=0)["mean"],
            }
        ka = [f[f"k_{crit}_attention"].values.astype(float) for f in fs]
        kr = [f[f"k_{crit}_random"].values.astype(float) for f in fs]
        kg = [f[f"k_{crit}_gradient"].values.astype(float) for f in fs]
        c["attention_beats_random"] = fc.seed_stat([_win(x, y) for x, y in zip(ka, kr)])
        c["gradient_beats_attention"] = fc.seed_stat([_win(x, y) for x, y in zip(kg, ka)])
        # S&S-exact (conditional on the zero vector flipping the decision)
        ret = [f[f"zero_{crit}_flip"].values == 1 for f in fs]
        ss = {"retained_frac": fc.seed_stat([x.astype(float) for x in ret], n_boot=0)["mean"]}
        for r in RANKS:
            cond = []
            for f, keep in zip(fs, ret):
                fr = (f[f"k_{crit}_{r}"] / f["N"]).fillna(1.0).values      # flips at the terminal step = 1.0
                cond.append(np.where(keep, fr, np.nan))
            ss[r] = {"median": fc.seed_stat(cond, stat=np.nanmedian, n_boot=0)["mean"],
                     "q1": fc.seed_stat(cond, stat=lambda v: np.nanpercentile(v, 25), n_boot=0)["mean"],
                     "q3": fc.seed_stat(cond, stat=lambda v: np.nanpercentile(v, 75), n_boot=0)["mean"]}
        excl = pd.concat([f.loc[~keep, "top"] for f, keep in zip(fs, ret)])
        ss["excluded_by_top_label"] = {i2l.get(int(k), str(k)): round(v / len(fs), 1)
                                       for k, v in excl.value_counts().items()}
        c["ss_exact"] = ss
        out[crit] = c
    t1 = {"median_delta_jsd_top": fc.seed_stat([(f.t1_jsd_top_i - f.t1_jsd_top_r).values for f in fs],
                                               stat=np.nanmedian, n_boot=0)["mean"],
          "frac_top_beats_random_jsd": fc.seed_stat([(f.t1_jsd_top_i > f.t1_jsd_top_r).astype(float).values
                                                     for f in fs], n_boot=0)["mean"],
          "set_flip_top_only_pct": 100 * fc.seed_stat([((f.t1_set_i == 1) & (f.t1_set_r == 0)).astype(float).values
                                                       for f in fs], n_boot=0)["mean"]}
    out["test1"] = t1
    return out
 
 
def flip_curve(frames, crit, rank, grid):
    """Fraction of ALL instances flipped after removing <= x of the positions,
    averaged over seeds (never-flipping instances count as not flipped)."""
    curves = []
    for f in frames.values():
        fr = (f[f"k_{crit}_{rank}"] / f["N"]).values
        curves.append([(fr <= x).mean() for x in grid])
    return np.nanmean(np.array(curves), axis=0)
 
 
def plot_wide(results, crit, mode):
    schemes = [s for s in fc.SCHEMES_TO_TEST if s in results]
    if not schemes:
        return
    # (a) S&S-style boxplot, among instances that flip; coverage printed on top
    fig, axes, axd = fc.wide_grid(schemes, "frac removed")
    for (scheme, kind, v), ax in axd.items():
        cell = results[scheme].get((kind, v))
        if cell is None:
            ax.set_title(f"{fc.pretty(kind, v)}\n(no data)", fontsize=6, color="#999999")
            continue
        frames, summ = cell
        data = [np.concatenate([(f[f"k_{crit}_{r}"] / f["N"]).dropna().values for f in frames.values()])
                for r in RANKS]
        data = [d if len(d) else np.array([np.nan]) for d in data]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.62,
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(color="#444444", linewidth=0.7),
                        capprops=dict(color="#444444", linewidth=0.7),
                        boxprops=dict(linewidth=0.5, edgecolor="#222222"))
        for patch, col in zip(bp["boxes"], fc.BLUES):
            patch.set_facecolor(col)
        for j, r in enumerate(RANKS):
            ax.text(j + 1, 1.09, f"{100 * summ[crit][r]['coverage']['mean']:.0f}",
                    ha="center", va="bottom", fontsize=4.5)
        ax.set_title(f"{fc.pretty(kind, v)}\n$n$={summ['n']}", fontsize=6, pad=7)
        ax.set_ylim(-0.02, 1.08); ax.set_xticks([]); ax.set_yticks([0, 0.5, 1.0])
    handles = [Patch(facecolor=c, edgecolor="#222222", label=l) for c, l in zip(fc.BLUES, RANK_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, 0.0),
               title=f"ranking (box = instances whose {'label set' if crit == 'set' else 'top label'} flips; "
                     f"number above = % of ALL instances that flip)", title_fontsize=6.5)
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"ss_fig1_wide_{crit}_{mode}.png"), rect=(0, 0.6 / fig.get_figheight(), 1, 1))
 
    # (b) flip curves over the FULL test set (censoring-aware view)
    grid = np.linspace(0, 1, 101)
    fig, axes, axd = fc.wide_grid(schemes, "frac flipped")
    for (scheme, kind, v), ax in axd.items():
        cell = results[scheme].get((kind, v))
        ax.set_title(fc.pretty(kind, v), fontsize=6)
        if cell is None:
            continue
        frames, _ = cell
        for r, col in zip(RANKS, LINE_COLORS):
            ax.plot(grid, flip_curve(frames, crit, r, grid), color=col, lw=1.0)
        ax.set_ylim(0, 1.0); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=5)
    handles = [Line2D([], [], color=c, lw=1.5, label=l) for c, l in zip(LINE_COLORS, RANK_LABELS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=True,
               bbox_to_anchor=(0.5, 0.0),
               title="fraction of ALL test instances flipped vs fraction of positions removed "
                     "(higher / earlier = more faithful)", title_fontsize=6.5)
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"ss_flipcurve_wide_{crit}_{mode}.png"), rect=(0, 0.6 / fig.get_figheight(), 1, 1))
 
 
def summarize():
    results, payload = {}, {"test": "Serrano & Smith erasure (final attention), multi-label",
                            "mode": fc.ATTENTION_MODE, "threshold": fc.THRESH,
                            "stat_format": "mean over seeds, sd_seeds, 95% bootstrap CI over test instances",
                            "results": []}
    for scheme in fc.SCHEMES_TO_TEST:
        results[scheme] = {}
        print(f"\n=== {scheme}")
        i2l = fc.load_labels(scheme)
        for kind, variant in fc.COMBOS:
            frames = fc.load_seed_frames(TEST, scheme, kind, variant)
            if not frames:
                print(f"  {fc.pretty(kind, variant):8s} (no CSVs)")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                summ = model_summary(frames, i2l)
            results[scheme][(kind, variant)] = (frames, summ)
            s = summ["set"]
            print(f"  {fc.pretty(kind, variant):8s} n={summ['n']} seeds={summ['n_seeds']} | "
                  f"coverage Attn {fc.fmt(s['attention']['coverage'])}  Rand {fc.fmt(s['random']['coverage'])} | "
                  f"median frac Attn {fc.fmt(s['attention']['median_frac_flippers'])} | "
                  f"P(Attn<Rand) {fc.fmt(s['attention_beats_random'])} | "
                  f"S&S kept {100 * s['ss_exact']['retained_frac']:.0f}%")
            payload["results"].append({"scheme": scheme, "model": fc.pretty(kind, variant),
                                       "kind": kind, "variant": variant, **summ})
    for crit in CRITS:
        plot_wide(results, crit, fc.ATTENTION_MODE)
    fc.write_summary(TEST, payload)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true", help="only summarise existing CSVs")
    ap.add_argument("--no-summary", action="store_true", help="only compute (SLURM array tasks)")
    args = ap.parse_args()
    fc.print_settings("ss_erasure_test.py  (Serrano & Smith, multi-label)")
    if not args.summarize:
        compute()
    if not args.no_summary:
        summarize()
 
 
if __name__ == "__main__":
    main()