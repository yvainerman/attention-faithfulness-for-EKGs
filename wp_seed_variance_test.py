"""
wp_seed_variance_test.py
------------------------
Wiegreffe & Pinter (2019) seed-variance baseline.
Tests how much attention changes when the same model is trained with
different random seeds.

For each test recording, compare the final attention distributions from
different seeds using Jensen-Shannon divergence (JSD). With 5 seeds, there
are 10 possible pairs of attention distributions.

This provides a baseline for the J&W adversarial attention test:
if adversarial attention is not much more different than attention from
another training seed, the difference may be explained by ordinary
training randomness.

The script also measures how often different seeds predict the same set
of labels, even when their attention distributions differ.

If adversarial-test CSV files are available, the summary compares each
adversarial JSD with the seed-to-seed JSD for the same recording.

(needs ALL seeds' normal checkpoints at once -> one SLURM task per scheme)
Env: ECG_TEST_SCHEMES, ECG_SEEDS, ECG_FLAT_VARIANTS, ECG_HIER_VARIANTS, ECG_N_SAMPLES
"""
import argparse
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import torch

import faithfulness_common as fc

TEST = "wp_seed_variance"
ADV_TEST = "jw_adversarial"
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]


def csv_path(scheme, kind, variant):
    return os.path.join(fc.test_dir(TEST, scheme), f"{kind}_{variant}_normal.csv")


def run_model(data, kind, variant):
    """Compare attention distributions from different training seeds.

    For every seed, load the trained model and collect its final attention
    distribution and predicted label set for each test recording."""
    scheme = data["scheme"]
    per_seed = {}
    for seed in fc.SEEDS_TO_TEST:
        model, path = fc.load_model(kind, variant, scheme, seed, data["num_classes"], uniform_trained=False)
        if model is None:
            print(f"  !! missing {path}")
            continue
        
        # store each recording's attn dist, predicted label set, largest attn weight.
        per_seed[seed] = {inst.idx: (inst.alphas.detach(), inst.p >= fc.THRESH, float(inst.alphas.max()))
                          for inst in fc.iter_instances(data, model)}
        del model
    
    # At least 2 seeds needed to compare attn dists.
    if len(per_seed) < 2:
        print(f"  {fc.pretty(kind, variant)}: need >= 2 seeds, found {len(per_seed)} -- skipped")
        return None
    seeds = sorted(per_seed)
    rows = []
    
    # Compare only instances present for every successfully loaded seed
    for idx in sorted(set.intersection(*[set(d) for d in per_seed.values()])):
        A = torch.stack([per_seed[s][idx][0] for s in seeds])                 # (n_seeds, N)
        sets = torch.stack([per_seed[s][idx][1] for s in seeds])              # (n_seeds, C)
        
        # Generate every unique pair of seeds (10 pairs when 5 seeds).
        pairs = list(itertools.combinations(range(len(seeds)), 2))
        i, j = map(list, zip(*pairs))
        
        # Compare attn dists. for each seed pair
        jsd = fc.jsd_rows(A[i], A[j]).cpu().numpy()
        same = (sets[i] == sets[j]).all(dim=1).float().cpu().numpy()
        rows.append({"idx": idx, "N": int(A.shape[1]), "n_seeds": len(seeds),
                     "seed_jsd_mean": float(jsd.mean()), "seed_jsd_max": float(jsd.max()),
                     "mean_max_attn": float(np.mean([per_seed[s][idx][2] for s in seeds])),
                     "pred_set_agreement": float(same.mean())})
    df = pd.DataFrame(rows)
    out = csv_path(scheme, kind, variant)
    df.to_csv(out, index=False)
    print(f"  {fc.pretty(kind, variant):8s} seeds={seeds}: n={len(df)}  median seed JSD="
          f"{df.seed_jsd_mean.median():.4f}  label-set agreement={df.pred_set_agreement.mean():.3f}  -> {out}",
          flush=True)
    return out


def compute():
    for scheme in fc.SCHEMES_TO_TEST:
        loaded = fc.load_scheme(scheme)
        for kind, variant in fc.COMBOS:
            run_model(loaded[kind], kind, variant)
        del loaded


# SUMMARY
def compare_adversarial(scheme, kind, variant, sv):
    """Measure how often adversarial JSD exceeds normal seed-to-seed JSD."""
    files = fc.seed_csvs(ADV_TEST, scheme, kind, variant, mode="normal")
    if not files:
        return None
    
    # Index baseline by instance ID so each adversarial result is
    # compared against the baseline for same instance.
    base = sv.set_index("idx")["seed_jsd_mean"]
    out = {}
    adv = pd.concat([pd.read_csv(p).assign(seed=s) for s, p in files.items()])
    for eps, d in adv.groupby("eps"):
        fr = []
        for _, ds in d.groupby("seed"):
            ds = ds[ds.idx.isin(base.index)]
            
            # Calculate fraction of matched instances where adversarial
            # attn. differs more than normal seed variance baseline.
            fr.append(float((ds.jsd.values > base.loc[ds.idx].values).mean()))
        out[f"eps={eps}"] = {"frac_adv_jsd_gt_seed_jsd": float(np.mean(fr)), "n_seeds": len(fr)}
    return out


def plot_scheme(scheme, cells):
    fig, axes, axd = fc.scheme_pair_grid(f"W&P seed variance - {scheme}: mean pairwise JSD of the "
                                         f"final attention across seeds, by max attention")
    for key, ax in axd.items():
        df = cells.get(key)
        if df is not None:
            data, pos, labs = [], [], []
            for b, (lo, hi) in enumerate(BINS):
                v = df.loc[(df.mean_max_attn >= lo) & (df.mean_max_attn < hi), "seed_jsd_mean"].values
                if len(v) > 1:
                    data.append(v); pos.append(b); labs.append(f"[{lo:.2f},{min(hi, 1):.2f})\nn={len(v)}")
            if data:
                parts = ax.violinplot(data, positions=pos, vert=False, showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor("#5e8eb5"); pc.set_alpha(0.6)
                ax.set_yticks(pos); ax.set_yticklabels(labs, fontsize=5)
        ax.set_xlim(0, 0.7); ax.grid(alpha=0.3, axis="x")
        ax.set_xlabel("seed JSD", fontsize=6)
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"wp_seed_variance_{scheme}.png"))


def summarize():
    payload = {"test": "Wiegreffe & Pinter seed-variance baseline (final attention)",
               "metric": "mean pairwise JSD of attention across seeds, per instance",
               "comparison": "fraction of instances whose J&W adversarial JSD exceeds the seed JSD",
               "results": []}
    for scheme in fc.SCHEMES_TO_TEST:
        print(f"\n=== {scheme}")
        cells = {}
        for kind, variant in fc.COMBOS:
            p = csv_path(scheme, kind, variant)
            if not os.path.exists(p):
                print(f"  {fc.pretty(kind, variant):8s} (no CSV)")
                continue
            df = pd.read_csv(p)
            cells[(kind, variant)] = df
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                rec = {"n": len(df), "n_seeds": int(df.n_seeds.max()),
                       "median_seed_jsd": fc.seed_stat([df.seed_jsd_mean.values], stat=np.nanmedian),
                       "mean_label_set_agreement": float(df.pred_set_agreement.mean()),
                       "vs_adversarial": compare_adversarial(scheme, kind, variant, df)}
            adv = rec["vs_adversarial"] or {}
            print(f"  {fc.pretty(kind, variant):8s} n={rec['n']} seeds={rec['n_seeds']} | "
                  f"median seed JSD {fc.fmt(rec['median_seed_jsd'])} | label-set agreement "
                  f"{rec['mean_label_set_agreement']:.3f} | adversarial > seed: "
                  + ", ".join(f"{k} {v['frac_adv_jsd_gt_seed_jsd']:.3f}" for k, v in adv.items()))
            payload["results"].append({"scheme": scheme, "model": fc.pretty(kind, variant),
                                       "kind": kind, "variant": variant, **rec})
        if cells:
            plot_scheme(scheme, cells)
    fc.write_summary(TEST, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    fc.print_settings("wp_seed_variance_test.py  (Wiegreffe & Pinter seed variance)")
    if not args.summarize:
        compute()
    if not args.no_summary:
        summarize()


if __name__ == "__main__":
    main()
