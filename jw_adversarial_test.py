"""
jw_adversarial_test.py
----------------------
Jain & Wallace (2019) per-instance adversarial attention at the FINAL attention
layer (paper P4). With the trained model frozen, search for an attention
distribution alpha' that is as DIFFERENT as possible from the learned alpha
(max JSD) while the output stays (almost) the same. If a very different alpha'
gives the same output, the learned alpha was not a necessary explanation.

  alpha' = softmax(theta), theta initialised at log(alpha)
  maximise  JSD(alpha', alpha)
  subject to  |p_c(alpha') - p_c(alpha)| <= eps  for EVERY label c

Env: ECG_TEST_SCHEMES, ECG_SEEDS, ECG_FLAT_VARIANTS, ECG_HIER_VARIANTS,
     ECG_N_SAMPLES, ECG_ADV_EPSILONS ("0.05,0.01"), ECG_ADV_STEPS (300),
     ECG_ADV_LR (0.05), ECG_ADV_LAMBDA (10), ECG_ADV_BATCH (256), ECG_ADV_KEEP_SET (1)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D

import faithfulness_common as fc

TEST = "jw_adversarial"
EPSILONS = [float(e) for e in fc.env_list("ECG_ADV_EPSILONS", ["0.05", "0.01"])]
N_STEPS = int(os.environ.get("ECG_ADV_STEPS", "300"))
ADV_LR = float(os.environ.get("ECG_ADV_LR", "0.05"))
LAMBDA = float(os.environ.get("ECG_ADV_LAMBDA", "10.0"))
BATCH = int(os.environ.get("ECG_ADV_BATCH", "256"))
KEEP_SET = fc.env_bool("ECG_ADV_KEEP_SET", True)    # adversary must also keep the predicted label set


def _probs(model, a, feats):
    """attention (B,N) x features (B,N,D) -> sigmoid outputs (B,C)."""
    return torch.sigmoid(model.classifier(torch.einsum("bn,bnd->bd", a, feats)))


def _feasible(p, p0, eps):
    """Every label within eps AND (if KEEP_SET) the predicted label set unchanged."""
    ok = (p - p0).abs().max(dim=1).values <= eps
    if KEEP_SET:
        ok &= ((p >= fc.THRESH) == (p0 >= fc.THRESH)).all(dim=1)
    return ok


def adversary(model, feats, a0, p0, eps):
    """Batched search. Returns best feasible alpha' and diagnostics per row."""
    B = a0.shape[0]
    theta = a0.clamp_min(1e-8).log().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=ADV_LR)
    best = torch.zeros(B, device=fc.DEVICE)
    best_a = a0.clone()
    found = torch.zeros(B, dtype=torch.bool, device=fc.DEVICE)
    with torch.enable_grad():
        for _ in range(N_STEPS):
            a = F.softmax(theta, dim=1)
            p = _probs(model, a, feats)
            dev = (p - p0).abs().max(dim=1).values
            jsd = fc.jsd_rows(a, a0)
            loss = (-jsd + LAMBDA * torch.relu(dev - eps)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():                      # keep the best FEASIBLE point seen
                ok = _feasible(p.detach(), p0, eps) & (jsd > best)
                best = torch.where(ok, jsd.detach(), best)
                best_a[ok] = a.detach()[ok]
                found |= ok
    with torch.no_grad():
        nf = ~found                                    # line search alpha -> optimiser's last point
        if nf.any():
            a_end = F.softmax(theta, dim=1)[nf]
            a_start, f_nf, p_nf = a0[nf], feats[nf], p0[nf]
            lo = torch.zeros(a_end.shape[0], 1, device=fc.DEVICE)
            hi = torch.ones_like(lo)
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                a_mid = (1 - mid) * a_start + mid * a_end
                ok = _feasible(_probs(model, a_mid, f_nf), p_nf, eps)
                lo = torch.where(ok[:, None], mid, lo)
                hi = torch.where(ok[:, None], hi, mid)
            best_a[nf] = (1 - lo) * a_start + lo * a_end
        best = fc.jsd_rows(best_a, a0)
        p_best = _probs(model, best_a, feats)
        dev_best = (p_best - p0).abs().max(dim=1).values
        set_changed = ((p_best >= fc.THRESH) != (p0 >= fc.THRESH)).any(dim=1)
    return best, dev_best, found, set_changed


def run_model(data, kind, variant, seed):
    scheme = data["scheme"]
    model, path = fc.load_model(kind, variant, scheme, seed, data["num_classes"], uniform_trained=False)
    if model is None:
        print(f"  !! missing {path} -- skipped")
        return None
    for w in fc.check_run_record(kind, variant, scheme, False):
        print(f"  [warn] {fc.pretty(kind, variant)}: {w}")
    groups = {}                                        # N -> list of (info, feats, alpha, p)
    for inst in fc.iter_instances(data, model):
        if inst.N >= 2:
            groups.setdefault(inst.N, []).append((inst.info(), inst.features[0], inst.alphas, inst.p))
    rows = []
    for N, items in groups.items():
        for s in range(0, len(items), BATCH):
            chunk = items[s:s + BATCH]
            feats = torch.stack([c[1] for c in chunk])
            a0 = torch.stack([c[2] for c in chunk])
            p0 = torch.stack([c[3] for c in chunk])
            jmax = fc.jsd_max(a0)
            for eps in EPSILONS:
                jsd, dev, found, setch = adversary(model, feats, a0, p0, eps)
                for j, c in enumerate(chunk):
                    rows.append({**c[0], "eps": eps, "jsd": float(jsd[j]), "jsd_max": float(jmax[j]),
                                 "jsd_norm": float(jsd[j] / jmax[j]) if jmax[j] > 0 else np.nan,
                                 "max_dev": float(dev[j]), "opt_found": int(found[j]),
                                 "set_changed": int(setch[j])})
    df = pd.DataFrame(rows)
    out = fc.csv_path(TEST, scheme, kind, variant, seed, mode="normal")
    df.to_csv(out, index=False)
    for eps in EPSILONS:
        d = df[df.eps == eps]
        if len(d):
            print(f"  {fc.pretty(kind, variant):8s} seed {seed} eps={eps}: n={len(d)}  "
                  f"median JSD={d.jsd.median():.3f}  median JSD/max={d.jsd_norm.median():.3f}  "
                  f"optimiser-found={d.opt_found.mean():.2f}  set changed={d.set_changed.mean():.3f}", flush=True)
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
    out = {}
    for eps in EPSILONS:
        fs = [f[f.eps == eps].sort_values("idx").reset_index(drop=True) for f in frames.values()]
        fs = [f for f in fs if len(f)]
        if not fs:
            continue
        common = set.intersection(*[set(f.idx) for f in fs])
        fs = [f[f.idx.isin(common)].reset_index(drop=True) for f in fs]
        hi = [(f.max_attn > 0.5).values for f in fs]
        out[f"eps={eps}"] = {
            "n": len(fs[0]), "n_seeds": len(fs),
            "median_jsd": fc.seed_stat([f.jsd.values for f in fs], stat=np.nanmedian),
            "median_jsd_norm": fc.seed_stat([f.jsd_norm.values for f in fs], stat=np.nanmedian),
            "mean_jsd_norm": fc.seed_stat([f.jsd_norm.values for f in fs]),
            "median_jsd_max": fc.seed_stat([f.jsd_max.values for f in fs], stat=np.nanmedian, n_boot=0)["mean"],
            "frac_optimizer_found": fc.seed_stat([f.opt_found.values.astype(float) for f in fs], n_boot=0)["mean"],
            "frac_set_changed": fc.seed_stat([f.set_changed.values.astype(float) for f in fs], n_boot=0)["mean"],
            "n_maxattn_gt_0.5": float(np.mean([h.sum() for h in hi])),
            "median_jsd_maxattn_gt_0.5": fc.seed_stat([np.where(h, f.jsd.values, np.nan) for f, h in zip(fs, hi)],
                                                      stat=np.nanmedian, n_boot=0)["mean"],
        }
    return out


def plot_scheme(scheme, cells, i2l):
    import matplotlib.pyplot as plt
    eps = EPSILONS[0]
    cmap = plt.get_cmap("tab10")
    fig, axes, axd = fc.scheme_pair_grid(f"J&W adversarial - {scheme}: max attention vs max JSD "
                                         f"with every label within eps={eps}")
    for key, ax in axd.items():
        if key in cells:
            df = pd.concat(cells[key].values())
            df = df[df.eps == eps]
            ax.scatter(df.max_attn, df.jsd, s=4, alpha=0.4, edgecolors="none",
                       c=[cmap(int(t) % 10) for t in df["top"]])
            present = sorted(int(t) for t in df["top"].unique())[:6]
            ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=cmap(t % 10),
                                      label=i2l.get(t, str(t))) for t in present],
                      fontsize=5, loc="upper right", framealpha=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 0.72); ax.grid(alpha=0.3)
        ax.set_xlabel(r"max attn $\hat\alpha$", fontsize=6)
        ax.set_ylabel("max JSD", fontsize=6)
    fc.save_fig(fig, os.path.join(fc.OUT_ROOT, f"jw_adversarial_{scheme}.png"))


def summarize():
    payload = {"test": "Jain & Wallace per-instance adversarial attention, multi-label",
               "constraint": "every label's |p_c - p'_c| <= eps (Bernoulli TVD)"
                             + (" and predicted label set unchanged" if KEEP_SET else ""),
               "epsilons": EPSILONS, "steps": N_STEPS, "lr": ADV_LR, "lambda": LAMBDA,
               "jsd_norm": "JSD / JSD_max(alpha), JSD_max = JSD(one-hot at argmin alpha, alpha)",
               "stat_format": "mean over seeds, sd_seeds, 95% bootstrap CI over test instances",
               "results": []}
    for scheme in fc.SCHEMES_TO_TEST:
        print(f"\n=== {scheme}")
        cells = {}
        for kind, variant in fc.COMBOS:
            frames = {s: pd.read_csv(p) for s, p in fc.seed_csvs(TEST, scheme, kind, variant, mode="normal").items()}
            frames = {s: f for s, f in frames.items() if len(f)}
            if not frames:
                print(f"  {fc.pretty(kind, variant):8s} (no CSVs)")
                continue
            cells[(kind, variant)] = frames
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                s = model_summary(frames)
            for key, v in s.items():
                print(f"  {fc.pretty(kind, variant):8s} {key}: n={v['n']} seeds={v['n_seeds']} | "
                      f"median JSD {fc.fmt(v['median_jsd'])} | median JSD/max {fc.fmt(v['median_jsd_norm'])} | "
                      f"set changed {v['frac_set_changed']:.3f}")
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
    fc.print_settings("jw_adversarial_test.py  (Jain & Wallace adversarial, multi-label)",
                      f"eps={EPSILONS} steps={N_STEPS} lr={ADV_LR} lambda={LAMBDA} batch={BATCH}")
    if not args.summarize:
        compute()
    if not args.no_summary:
        summarize()


if __name__ == "__main__":
    main()

