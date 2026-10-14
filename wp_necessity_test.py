"""
wp_necessity_test.py
--------------------
Tests whether learned attention is necessary for model performance
(Wiegreffe & Pinter, 2019).

Evaluate three versions of each model on the same PTB-XL test fold:

normal
The normally trained model using learned attention.

uniform_trained
A model trained from scratch without learned attention. Its attention
weights are fixed to 1/N, where N is the number of attention positions.

uniform_at_test
The normally trained model, but with its attention weights replaced
by uniform weights during testing. The model is not retrained.

RUN  python wp_necessity_test.py [--summarize | --no-summary]
Env: ECG_TEST_SCHEMES, ECG_SEEDS, ECG_FLAT_VARIANTS, ECG_HIER_VARIANTS, ECG_N_BOOT
"""
import argparse
import os
import warnings

import numpy as np
from scipy.stats import rankdata, ttest_ind, ttest_rel
from sklearn.metrics import f1_score

import faithfulness_common as fc

TEST = "wp_necessity"
CONDITIONS = ("normal", "uniform_trained", "uniform_at_test")


def npz_path(scheme, kind, variant, seed):
    """Build the filename for one model's saved predictions."""
    return os.path.join(fc.test_dir(TEST, scheme), f"{kind}_{variant}_seed{seed}.npz")


def predict(data, model):
    """Generate predictions and labels for all test instances.
    	idx: identifiers for the test instances.
    	ys: the true labels for each instance.
	ps: the model's predicted probabilities."""
    idx, ys, ps = [], [], []
    for inst in fc.iter_instances(data, model, batch_size=64):
        idx.append(inst.idx); ys.append(inst.y); ps.append(inst.p.cpu().numpy())
    order = np.argsort(idx)
    return np.asarray(idx)[order], np.stack(ys)[order], np.stack(ps)[order]


def run_model(data, kind, variant, seed):
    """Evaluating one model and seed under all three attention conditions. 
    uniform_trained=False loads the normally trained model, 
    uniform_trained=True loads the model trained with uniform attention,
    uniform_at_test=True loads normal model with final attention replaced at eval time."""
    scheme, C = data["scheme"], data["num_classes"]
    out = {}
    specs = {"normal": dict(uniform_trained=False),
             "uniform_trained": dict(uniform_trained=True),
             "uniform_at_test": dict(uniform_trained=False, uniform_at_test=True)}
    for cond, kw in specs.items():
        model, path = fc.load_model(kind, variant, scheme, seed, C, **kw)
        if model is None:
            print(f"  !! {fc.pretty(kind, variant)} seed {seed}: missing {path} ({cond} skipped)")
            continue
        if cond != "uniform_at_test":
            for w in fc.check_run_record(kind, variant, scheme, kw["uniform_trained"]):
                print(f"  [warn] {fc.pretty(kind, variant)} {cond}: {w}")
        idx, y, p = predict(data, model)
        out["idx"], out["y"], out[f"p_{cond}"] = idx, y, p
    
    # Save the predictions. 
    if "idx" not in out:
        return None
    path = npz_path(scheme, kind, variant, seed)
    np.savez_compressed(path, **out)
    msg = "  ".join(f"{c}={macro_auc(out['y'], out['p_' + c]):.4f}" for c in CONDITIONS if f"p_{c}" in out)
    print(f"  {fc.pretty(kind, variant):8s} seed {seed}: macro-AUROC {msg}  -> {path}", flush=True)
    return path


def compute():
    for scheme in fc.SCHEMES_TO_TEST:
        loaded = fc.load_scheme(scheme)
        for kind, variant in fc.COMBOS:
            for seed in fc.SEEDS_TO_TEST:
                run_model(loaded[kind], kind, variant, seed)
        del loaded


# METRICS
def macro_auc(y, p):
    aucs = []
    for c in range(y.shape[1]):
        yc = y[:, c]
        npos = yc.sum(); nneg = len(yc) - npos
        if npos == 0 or nneg == 0:
            continue
        r = rankdata(p[:, c])
        aucs.append((r[yc == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs)) if aucs else float("nan")


def f1s(y, p):
    pred = (p >= fc.THRESH).astype(int)
    return (float(f1_score(y, pred, average="macro", zero_division=0)),
            float(f1_score(y, pred, average="micro", zero_division=0)))


# SUMMARY
def model_summary(runs):
    """Measure how much performance changes when attention is made uniform."""
    seeds = sorted(runs)
    out = {"n": int(len(runs[seeds[0]]["idx"])), "n_seeds": len(seeds)}
    per = {c: [] for c in CONDITIONS}
    for c in CONDITIONS:
        aucs, mac, mic = [], [], []
        for s in seeds:
            if f"p_{c}" not in runs[s]:
                continue
            y, p = runs[s]["y"], runs[s][f"p_{c}"]
            aucs.append(macro_auc(y, p)); a, b = f1s(y, p); mac.append(a); mic.append(b)
        per[c] = aucs
        if aucs:
            out[c] = {"macro_auc": {"mean": np.mean(aucs), "sd_seeds": np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0,
                                    "per_seed": aucs},
                      "macro_f1": float(np.mean(mac)), "micro_f1": float(np.mean(mic)), "n_seeds": len(aucs)}

    # Calculate performance difference between normal condition vs./ alternate
    def delta(cond):
        both = [s for s in seeds if "p_normal" in runs[s] and f"p_{cond}" in runs[s]]
        if not both:
            return None
        n = len(runs[both[0]]["idx"])
        rng = np.random.default_rng(0)
        boots = []
        
        # Bootstrap CI
        for _ in range(fc.N_BOOT):
            ii = rng.integers(0, n, n)
            a = np.mean([macro_auc(runs[s]["y"][ii], runs[s]["p_normal"][ii]) for s in both])
            b = np.mean([macro_auc(runs[s]["y"][ii], runs[s][f"p_{cond}"][ii]) for s in both])
            boots.append(a - b)
        an = [macro_auc(runs[s]["y"], runs[s]["p_normal"]) for s in both]
        bn = [macro_auc(runs[s]["y"], runs[s][f"p_{cond}"]) for s in both]
        res = {"mean": float(np.mean(an) - np.mean(bn)),
               "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))] if boots else None,
               "n_seeds": len(both)}
        if len(both) > 1:
            t = ttest_rel(an, bn) if cond == "uniform_at_test" else ttest_ind(an, bn, equal_var=False)
            res["p_value_over_seeds"] = float(t.pvalue)
        return res
    out["delta_normal_minus_uniform_trained"] = delta("uniform_trained")
    out["delta_normal_minus_uniform_at_test"] = delta("uniform_at_test")
    return out


# PLOTS
def plot_scheme(scheme, cells):
    fig, axes, axd = fc.scheme_pair_grid(f"W&P necessity - {scheme}: macro-AUROC (mean ± SD over seeds)")
    cols = ["#1f77b4", "#bbbbbb", "#ff7f0e"]
    names = ["normal", "uniform\n(retrained)", "uniform\n(at test)"]
    for key, ax in axd.items():
        s = cells.get(key)
        if s:
            vals = [s[c]["macro_auc"]["mean"] if c in s else np.nan for c in CONDITIONS]
            errs = [s[c]["macro_auc"]["sd_seeds"] if c in s else 0 for c in CONDITIONS]
            ax.bar(range(3), vals, yerr=errs, color=cols, width=0.65, capsize=2)
            for xi, v in enumerate(vals):
                if np.isfinite(v):
                    ax.text(xi, min(v + 0.02, 1.0), f"{v:.3f}", ha="center", fontsize=5)
        ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=5)
        lo = 0.5
        if s:
            finite = [s[c]["macro_auc"]["mean"] for c in CONDITIONS if c in s]
            lo = min(0.5, min(finite) - 0.05) if finite else 0.5
        ax.set_ylim(lo, 1.02); ax.grid(alpha=0.3, axis="y")
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"wp_necessity_{scheme}.png"))


def summarize():
    payload = {"test": "Wiegreffe & Pinter uniform-attention necessity, multi-label",
               "conditions": {"normal": "learned attention",
                              "uniform_trained": "retrained from scratch with final attention fixed to 1/N",
                              "uniform_at_test": "normal model, final attention set to 1/N at test time"},
               "delta_ci": "95% bootstrap over test instances (all seeds resampled together)",
               "results": []}
    for scheme in fc.SCHEMES_TO_TEST:
        print(f"\n=== {scheme}")
        cells = {}
        for kind, variant in fc.COMBOS:
            files = sorted(glob_npz(scheme, kind, variant))
            if not files:
                print(f"  {fc.pretty(kind, variant):8s} (no results)")
                continue
            runs = align_runs({seed: dict(np.load(p)) for seed, p in files})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                s = model_summary(runs)
            cells[(kind, variant)] = s
            dt, di = s["delta_normal_minus_uniform_trained"], s["delta_normal_minus_uniform_at_test"]
            line = f"  {fc.pretty(kind, variant):8s} seeds={s['n_seeds']} | "
            for c in CONDITIONS:
                if c in s:
                    line += f"{c} {s[c]['macro_auc']['mean']:.4f}±{s[c]['macro_auc']['sd_seeds']:.4f} | "
            ci = lambda d: f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d.get("ci95") else ""
            if dt:
                line += f"d(train) {dt['mean']:+.4f} {ci(dt)} | "
            if di:
                line += f"d(test) {di['mean']:+.4f} {ci(di)}"
            print(line)
            payload["results"].append({"scheme": scheme, "model": fc.pretty(kind, variant),
                                       "kind": kind, "variant": variant, **s})
        if cells:
            plot_scheme(scheme, cells)
    fc.write_summary(TEST, payload)


def align_runs(runs):
    """Restrict all runs to the test instances shared by every seed."""
    common = set.intersection(*[set(r["idx"].tolist()) for r in runs.values()])
    out = {}
    for s, r in runs.items():
        keep = np.isin(r["idx"], list(common))
        out[s] = {k: v[keep] for k, v in r.items()}
    return out


def glob_npz(scheme, kind, variant):
    import glob
    out = []
    for p in glob.glob(os.path.join(fc.OUT_ROOT, scheme, TEST, f"{kind}_{variant}_seed*.npz")):
        try:
            out.append((int(p.rsplit("_seed", 1)[1][:-4]), p))
        except ValueError:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()
    fc.print_settings("wp_necessity_test.py  (Wiegreffe & Pinter uniform necessity)")
    if not args.summarize:
        compute()
    if not args.no_summary:
        summarize()


if __name__ == "__main__":
    main()
