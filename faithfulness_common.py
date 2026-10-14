"""
faithfulness_common.py
----------------------
Shared code for every attention-faithfulness test in the paper:

    ss_erasure_test.py        Serrano & Smith (2019)  erasure / decision flips   (P1)
    jw_correlation_test.py    Jain & Wallace (2019)   attention vs grad vs LOO   (P2)
    jw_permutation_test.py    Jain & Wallace (2019)   permuted attention         (P4)
    jw_adversarial_test.py    Jain & Wallace (2019)   adversarial attention      (P4)
    wp_necessity_test.py      Wiegreffe & Pinter (2019) uniform attention        (P3)
    wp_seed_variance_test.py  Wiegreffe & Pinter (2019) seed-variance baseline   (P4)

Every test is run at the model's FINAL attention layer (flat = the one attention
over 100-ms tokens; hierarchical = the sequence attention over beats), the S&S
convention for hierarchical models.

The models are trained with BCEWithLogitsLoss: every label c gets its OWN
probability p_c = sigmoid(logit_c), and the p_c do NOT sum to 1. So the model
does not output one distribution over classes -- it outputs one yes/no
(Bernoulli) distribution PER LABEL. The tests therefore compare outputs per
label, exactly as J&W and W&P do for their binary tasks:

  * per-label TVD   |p_c - p'_c|      (TVD of two Bernoullis; always in [0, 1])
  * per-label JSD   binary JSD        (always in [0, ln 2])
  summarised two ways:
  * "top"  -> the label the model scores highest (fixed from the ORIGINAL output)
  * "max"  -> the largest change over all labels ("did ANY label move?")
Neither summary grows with the number of labels, so one threshold means the
same thing for 5, 7 and 15 labels.

DECISIONS. The model's decision is its predicted LABEL SET {c : p_c >= 0.5}.
  * "set" flip -> the predicted label set changes (any label switches on/off)
  * "top" flip -> the top label's own yes/no decision changes
------------------------------------------------------------------------------

Per-instance CSVs from every test carry `idx` (position in the test dataset)
and the true / predicted label vectors, so results can be paired across seeds
and variants, bootstrapped, and broken down per class.
"""
import contextlib
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DATA_PATHS
from models.flat_seq import FlatSeqAttention
from models.hierarchical_seq import HierarchicalSeqAttention, SeqBiLSTM

# Architecture + data constants come from the trainers, so the tests can never
# drift from the checkpoints they load.
from train_hierarchical_seq import (
    ARCH as H_ARCH, FS as H_FS, PRE_SAMPLES, POST_SAMPLES, MIN_BEATS,
    SEEDS as TRAIN_SEEDS,
)
from train_flat_seq import ARCH as F_ARCH, FS as F_FS, FlatECGDataset
from data.sph_beats import load_sph_beats, HANBeatsDataset, ptbxl_beat_split_indices
from data.ptbxl import load_ptbxl_dataset, ptbxl_split_indices, PREP_TAG


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# SETTINGS (for SLURM)
def env_list(name, default):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(",") if x.strip()] if v else list(default)


def env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() not in ("0", "false", "no")


SCHEMES         = ("diagnostic", "rhythm", "form")
SCHEMES_TO_TEST = tuple(env_list("ECG_TEST_SCHEMES", SCHEMES))
SEEDS_TO_TEST   = list(TRAIN_SEEDS)               # ECG_SEEDS (same variable as the trainers)
DIAG_LEVEL      = os.environ.get("ECG_DIAG_LEVEL", "superclass")
USE_FILTERED    = env_bool("ECG_USE_FILTERED", True)
ATTENTION_MODE  = os.environ.get("ECG_ATTENTION_MODE", "normal")   # normal | uniform (uniform-TRAINED ckpts)
FLAT_VARIANTS   = tuple(env_list("ECG_FLAT_VARIANTS", FlatSeqAttention.VARIANTS))
HIER_VARIANTS   = tuple(env_list("ECG_HIER_VARIANTS", HierarchicalSeqAttention.VARIANTS))
COMBOS          = [("flat", v) for v in FLAT_VARIANTS] + [("hier", v) for v in HIER_VARIANTS]
THRESH          = float(os.environ.get("ECG_THRESHOLD", "0.5"))    # label is "on" if p >= THRESH
_NS             = os.environ.get("ECG_N_SAMPLES")
N_SAMPLES       = int(_NS) if _NS not in (None, "", "0") else None  # None = full test fold
N_BOOT          = int(os.environ.get("ECG_N_BOOT", "1000"))
OUT_ROOT        = "attention_tests"

if ATTENTION_MODE not in ("normal", "uniform"):
    raise ValueError(f"ECG_ATTENTION_MODE={ATTENTION_MODE!r} (use normal or uniform)")

# Paper names for the seven models.
PRETTY = {
    ("flat", "noenc"): "f-pt", ("flat", "conv"): "f-lc", ("flat", "bilstm"): "f-gl",
    ("hier", "noenc"): "h-lc-pt", ("hier", "conv"): "h-lc-lc",
    ("hier", "bilstm_cnn"): "h-lc-gl", ("hier", "bilstm_full"): "h-gl-gl",
}


# Map code names to paper names (flat conv -> f-lc, so on)
def pretty(kind, variant):
    return PRETTY.get((kind, variant), f"{kind}-{variant}")


def resolve_data_path():
    root = os.environ.get("ECG_DATA_ROOT")
    return DATA_PATHS["ptbxl"] if not root else os.path.join(root, "ptbxl")


# put a header on top of every log
def print_settings(test_name, extra=""):
    print(f"{'=' * 72}\n  {test_name}\n{'=' * 72}")
    print(f"  schemes={list(SCHEMES_TO_TEST)}  seeds={SEEDS_TO_TEST}  mode={ATTENTION_MODE}")
    print(f"  flat={list(FLAT_VARIANTS)}  hier={list(HIER_VARIANTS)}")
    print(f"  threshold={THRESH}  n_samples={N_SAMPLES or 'FULL test fold'}  "
          f"preprocessing={PREP_TAG}  device={DEVICE}")
    print(f"  data={resolve_data_path()}")
    if extra:
        print(f"  {extra}")
    print("=" * 72, flush=True)


# DATA (PTB-XL test fold 10)
def subsample(test_idx, n=None):
    """Optional cap on the number of test instances. The subset is FIXED (same
    for every variant and every seed) so results stay paired."""
    n = N_SAMPLES if n is None else n
    idxs = list(test_idx)
    if n is None or len(idxs) <= n:
        return idxs
    rng = np.random.RandomState(12345)
    return sorted(idxs[i] for i in rng.choice(len(idxs), size=n, replace=False))


def load_flat(scheme, data_path=None):
    data_path = data_path or resolve_data_path()
    data_list, num_classes, label2idx = load_ptbxl_dataset(
        data_path, filtered=USE_FILTERED, fs=F_FS,
        label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    _, _, test_idx = ptbxl_split_indices(
        data_path, filtered=USE_FILTERED, fs=F_FS,
        label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    return {"kind": "flat", "scheme": scheme, "dataset": FlatECGDataset(data_list),
            "num_classes": num_classes, "label2idx": label2idx,
            "test_idx": subsample(test_idx)}


def load_hier(scheme, data_path=None):
    data_path = data_path or resolve_data_path()
    beats_list, num_classes, label2idx = load_sph_beats(
        data_path, filtered=USE_FILTERED, fs=H_FS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES,
        dataset="ptbxl", label_scheme=scheme, diagnostic_level=DIAG_LEVEL)
    dataset = HANBeatsDataset(beats_list, num_beats=None, min_beats=MIN_BEATS)
    _, _, test_idx = ptbxl_beat_split_indices(
        data_path, filtered=USE_FILTERED, fs=H_FS,
        label_scheme=scheme, diagnostic_level=DIAG_LEVEL, min_beats=MIN_BEATS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES)
    return {"kind": "hier", "scheme": scheme, "dataset": dataset,
            "num_classes": num_classes, "label2idx": label2idx,
            "test_idx": subsample(test_idx)}


def load_scheme(scheme, kinds=("flat", "hier")):
    """Load the data each family needs for one scheme (flat signals and/or beats)."""
    out = {}
    if "flat" in kinds and FLAT_VARIANTS:
        out["flat"] = load_flat(scheme)
    if "hier" in kinds and HIER_VARIANTS:
        out["hier"] = load_hier(scheme)
    for d in out.values():
        save_labels(scheme, d["label2idx"])
    return out


def idx2label(data):
    return {v: k for k, v in data["label2idx"].items()}


# write a {index: name} file so summary job knows label names w/out loading signal
def save_labels(scheme, label2idx):
    """Small {index: name} file so the CPU summary step never loads the signals."""
    path = os.path.join(OUT_ROOT, scheme, "labels.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(v): k for k, v in label2idx.items()}, f, indent=1)


def load_labels(scheme):
    path = os.path.join(OUT_ROOT, scheme, "labels.json")
    if not os.path.exists(path):
        return {}
    return {int(k): v for k, v in json.load(open(path)).items()}


# MODELS + CHECKPOINTS
def build(kind, variant, num_classes):
    if kind == "hier":
        m = HierarchicalSeqAttention(num_classes=num_classes, variant=variant, **H_ARCH[variant])
    else:
        m = FlatSeqAttention(num_classes=num_classes, variant=variant, **F_ARCH[variant])
    return m.to(DEVICE)


def ckpt_stem(kind, variant, scheme, uniform=False):
    if kind == "hier":
        sub = "seq_uniform" if uniform else "seq"
    else:
        sub = "flat_seq_uniform" if uniform else "flat_seq"
    suffix = "_filtered" if USE_FILTERED else ""
    return f"checkpoints/{sub}/best_{variant}_{scheme}{suffix}"


def ckpt_path(kind, variant, scheme, seed, uniform=False):
    """The checkpoint of ONE training seed."""
    return f"{ckpt_stem(kind, variant, scheme, uniform)}_seed{seed}.pth"


def check_run_record(kind, variant, scheme, uniform=False):
    """Read trainer's _meta.json and warns if there's errors. Returns
    a list of warnings (empty = consistent)."""
    path = f"{ckpt_stem(kind, variant, scheme, uniform)}_meta.json"
    if not os.path.exists(path):
        return [f"no run record {path}"]
    meta = json.load(open(path))
    arch = H_ARCH if kind == "hier" else F_ARCH
    warn = []
    if meta.get("preprocessing") != PREP_TAG:
        warn.append(f"trained with preprocessing={meta.get('preprocessing')!r}, testing on {PREP_TAG!r}")
    if bool(meta.get("uniform_attention")) != bool(uniform):
        warn.append(f"run record says uniform_attention={meta.get('uniform_attention')}, expected {uniform}")
    if meta.get("arch") != arch.get(variant):
        warn.append(f"trained with arch={meta.get('arch')}, trainer now has {arch.get(variant)}")
    return warn


def _uniform_forward_3d(features):
    B, T, _ = features.shape
    w = torch.full((B, T, 1), 1.0 / T, device=features.device, dtype=features.dtype)
    return torch.sum(w * features, dim=1), w


def apply_uniform(kind, model):
    """Freeze the FINAL attention to 1/N. Same patch the trainers use."""
    att = model.seq_attention if kind == "hier" else model.attention
    for p in att.parameters():
        p.requires_grad_(False)
    att.forward = _uniform_forward_3d
    return model


def load_model(kind, variant, scheme, seed, num_classes, uniform_trained=None, uniform_at_test=False):
    """Build + load ONE seed's checkpoint, eval mode, weights frozen.
    uniform_trained : load the uniform-TRAINED checkpoint (default: ECG_ATTENTION_MODE)
    uniform_at_test : load the NORMAL checkpoint but set its final attention to 1/N
                      at test time (no retraining) -- R4 asked for this distinction!"""
    if uniform_trained is None:
        uniform_trained = (ATTENTION_MODE == "uniform")
    path = ckpt_path(kind, variant, scheme, seed, uniform=uniform_trained)
    if not os.path.exists(path):
        return None, path
    model = build(kind, variant, num_classes)
    if uniform_trained or uniform_at_test:
        model = apply_uniform(kind, model)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, path


def rnn_in_path(kind, model):
    if kind == "flat":
        return isinstance(model.seq_encoder, SeqBiLSTM)
    return (not model._beat_is_cnn) or isinstance(model.seq_encoder, SeqBiLSTM)


# workaround for GPU limitations.
def _rnn_guard(kind, model):
    if rnn_in_path(kind, model) and DEVICE.type == "cuda":
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


# DECOMP: run models, keep what tests need
@torch.no_grad()
def decompose_batch(kind, model, x):
    """Mirror the model's forward pass and expose the FINAL attention's input
    features and weights."""
    if kind == "flat":
        feats = model.seq_encoder(model.encoder(x))                 # (B,N,D)
        _, w = model.attention(feats)                               # (B,N,1)
        logits = model.classifier((w * feats).sum(dim=1))
        return {"features": feats, "alphas": w.squeeze(-1), "logits": logits}
    B, S, T, C = x.shape
    flat = x.reshape(B * S, T, C)
    if model._beat_is_cnn:
        flat = flat.permute(0, 2, 1)
    beat_feats = model.beat_encoder(flat)
    beat_vecs, beat_alphas = model.beat_attention(beat_feats)
    beat_vecs = beat_vecs.view(B, S, -1)
    feats = model.seq_encoder(beat_vecs)                            # (B,S,D)
    _, w = model.seq_attention(feats)                               # (B,S,1)
    logits = model.classifier((w * feats).sum(dim=1))
    return {"features": feats, "alphas": w.squeeze(-1), "logits": logits,
            "beat_vectors": beat_vecs, "beat_alphas": beat_alphas.squeeze(-1).view(B, S, -1)}


class Instance:
    """One test EKG, already run through the model.
    top = highest-scoring label -- idx = test-set index."""
    def __init__(self, idx, x, dec, label):
        self.idx = int(idx)
        self.x = x
        self.dec = dec
        self.features = dec["features"]
        self.alphas = dec["alphas"][0]
        self.logits = dec["logits"][0]
        self.p = torch.sigmoid(self.logits)
        self.y = np.asarray(label, dtype=int)
        self.top = int(self.p.argmax())
        self.N = int(self.alphas.shape[0])

    def info(self):
        """Columns every per-instance CSV starts with."""
        pred = (self.p >= THRESH).int().cpu().numpy()
        return {"idx": self.idx, "N": self.N, "top": self.top,
                "p_top": float(self.p[self.top]), "top_on": int(self.p[self.top] >= THRESH), 	# is top label actually predicted?
                "n_pred": int(pred.sum()), "max_attn": float(self.alphas.max()),			# how many labels predicted & max_attn
                "y": "".join(map(str, self.y)), "pred": "".join(map(str, pred))}				# y and pred are binary strings	


def iter_instances(data, model, batch_size=32):
    """Batching. Hierarchical batches only group recordings with the same beat count (from HAN-EKG)."""
    kind, ds, idxs = data["kind"], data["dataset"], data["test_idx"]
    if kind == "flat":
        groups = [idxs[i:i + batch_size] for i in range(0, len(idxs), batch_size)]
    else:
        by_len = {}
        for i in idxs:
            by_len.setdefault(ds.num_beats[i], []).append(i)
        groups = [g[j:j + batch_size] for g in by_len.values() for j in range(0, len(g), batch_size)]
    key = "beats" if kind == "hier" else "signal"
    for g in groups:
        items = [ds[i] for i in g]
        x = torch.stack([it[key] for it in items]).to(DEVICE)
        dec = decompose_batch(kind, model, x)
        for j, i in enumerate(g):
            yield Instance(i, x[j:j + 1], {k: v[j:j + 1] for k, v in dec.items()},
                           items[j]["label"].numpy())


@torch.no_grad()
def head_probs(model, features, alphas):
    """Keep features fixed, plug in new attention weights, and recompute output.
    An all-zero row gives the zero context vector (terminal state). Re-running with
    the original weights reproduces model's output to float32 rounding."""
    ctx = (alphas.unsqueeze(-1) * features).sum(dim=1)              # (K,D)
    return torch.sigmoid(model.classifier(ctx))


# MULTI-LABEL OUTPUT COMPARISON (instead of softmax)
def bernoulli_jsd(p, q, eps=1e-7):
    """JSD between Bernoulli(p) and Bernoulli(q)."""
    p = p.clamp(eps, 1 - eps); q = q.clamp(eps, 1 - eps) 
    m = 0.5 * (p + q)
    kl = lambda a, b: a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))		# KL div between two Bernoulli dists
    # calculate final JSD by averaging the KL divergence from each original dist to their midpoint.
    # 0 if two probabilities are identical. 
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_outputs(p0, p1, top):
    """p0 original output, p1 perturbed outputs, top = original top label.
    Returns (K,) tensors: tvd_top, tvd_max, jsd_top, jsd_max, set_changed, top_flipped."""
    p1 = p1.reshape(-1, p0.shape[0])
    d = (p1 - p0).abs()			# calculate absolute probability differences
    on0, on1 = p0 >= THRESH, p1 >= THRESH		# determine positives (using THRESH 0.5)
    return {
        "tvd_top": d[:, top],						# TVD for ONE label, original top label.
        "tvd_max": d.max(dim=1).values,			# TVD for ALL labels, which label changed most.
        "jsd_top": bernoulli_jsd(p1[:, top], p0[top].expand(p1.shape[0])),			# JSD for ONE label, original top label.
        "jsd_max": bernoulli_jsd(p1, p0.expand_as(p1)).max(dim=1).values,		# JSD for ALL labels, maximum of all JSDs.
        "set_changed": (on1 != on0).any(dim=1),	# Did any label flip? compare original/perturbed decisions for every label.
        "top_flipped": on1[:, top] != on0[top],		# Did original top label flip? 
    }


def decision_prob(p, top, top_on):
    """Probability the model gives to its ORIGINAL decision on the top label."""
    pt = p[..., top]
    return pt if top_on else 1.0 - pt


# GRADIENTS
def attention_gradient(model, inst):
    """S&S: d(decision prob of the top label) / d(final attention weights),
    features held fixed. Used to rank positions for the Gradient and
    Attention x Gradient orders."""
    top_on = bool(inst.p[inst.top] >= THRESH)
    a = inst.alphas.detach().clone().unsqueeze(0).requires_grad_(True)
    with torch.enable_grad():
        p = torch.sigmoid(model.classifier((a.unsqueeze(-1) * inst.features.detach()).sum(dim=1)))[0]
        dp = decision_prob(p, inst.top, top_on)
        (g,) = torch.autograd.grad(dp, a)
    return g[0].detach().cpu().numpy()


def _forward_fixed_alpha(kind, model, x, dec):
    """Full forward from the RAW input with every attention weight frozen at its
    original value (J&W: gradient flows through the encoders only)."""
    if kind == "flat":
        feats = model.seq_encoder(model.encoder(x))
        return model.classifier((dec["alphas"].detach().unsqueeze(-1) * feats).sum(dim=1))
    B, S, T, C = x.shape
    flat = x.reshape(B * S, T, C)
    if model._beat_is_cnn:
        flat = flat.permute(0, 2, 1)
    bf = model.beat_encoder(flat)
    bf = bf.view(B, S, *bf.shape[1:])
    beat_vecs = (dec["beat_alphas"].detach().unsqueeze(-1) * bf).sum(dim=2)
    feats = model.seq_encoder(beat_vecs)
    return model.classifier((dec["alphas"].detach().unsqueeze(-1) * feats).sum(dim=1))


def input_gradient_importance(kind, model, inst):
    """J&W gradient importance per final-attention position.
    Target = predicted probability of original top label. 
    Returns two arrays, one score per position:
      l2  = size of the prediction gradient across all samples + leads belonging
      	to that position.
      gxi = absolute value of the sum of input * gradient across those samples +
      	leads.
    Flat models:
    	A position is one non-overlapping PatchCNN token, T/N consecutive
	time samples (50 samples x 12 leads). 
    Hierarchical models:
    	A position is one beat (300 samples x 12 leads)."""
    x = inst.x.detach().clone().requires_grad_(True)
    with torch.enable_grad(), _rnn_guard(kind, model):
        p = torch.sigmoid(_forward_fixed_alpha(kind, model, x, inst.dec))[0, inst.top]
        (g,) = torch.autograd.grad(p, x)
    g, xv = g[0], x.detach()[0]
    if kind == "hier":                                   # (S, 300, 12)
        return g.norm(dim=(1, 2)).cpu().numpy(), (g * xv).sum(dim=(1, 2)).abs().cpu().numpy()
    C, T = g.shape                                       # (12, T)
    L = T // inst.N
    if L * inst.N != T:
        raise ValueError(f"signal length {T} is not N={inst.N} tokens x {L} samples")
    g, xv = g.view(C, inst.N, L), xv.view(C, inst.N, L)
    return g.norm(dim=(0, 2)).cpu().numpy(), (g * xv).sum(dim=(0, 2)).abs().cpu().numpy()


# LEAVE-ONE-OUT (J&W): DELETE one input position, re-run the WHOLE model
# delete: cut the frame out and splice. signal becomes 50 samples shorter, or beat is dropped.
# nothing is put in its place, so the input is only the patient's own signal.


@torch.no_grad()
def _seq_head_probs(model, beat_vecs):
    """Last three stages of hierarchical model. Runs reach encoder across beats
    then recomputes attention over remaining beats. Weights are calculated fresh
    and re-normed to sum to 1."""
    feats = model.seq_encoder(beat_vecs)
    ctx, _ = model.seq_attention(feats)
    return torch.sigmoid(model.classifier(ctx))


@torch.no_grad()
def loo_probs(kind, model, inst, chunk=None):
    """Returns an (N,C) table. Row t is the model's output with position t
    deleted."""
    if kind == "hier":
        bv = inst.dec["beat_vectors"][0]                    # (S,Db) one vector per beat
        S = bv.shape[0]
        # row s = every beat except beat s
        batch = torch.stack([torch.cat([bv[:s], bv[s + 1:]], dim=0) for s in range(S)])
        return _seq_head_probs(model, batch)

    x = inst.x[0]                                           # (12,T)
    T = x.shape[1]
    N = inst.N
    L = T // N                                              # samples per token (50)
    chunk = chunk or int(os.environ.get("ECG_LOO_CHUNK", "100"))   # tokens per GPU batch
    out = []
    for start in range(0, N, chunk):
        ts = range(start, min(N, start + chunk))
        # row t = the signal with token t's samples cut out and the two sides joined
        xb = torch.stack([torch.cat([x[:, :t * L], x[:, (t + 1) * L:]], dim=1) for t in ts])
        out.append(torch.sigmoid(model(xb)))
    return torch.cat(out, dim=0)


# ATTN DISTRIBUTIONS
def jsd_rows(p, q, eps=1e-12):
    """JSD between rows of two (B,N) attention distributions."""
    p = p.clamp_min(eps); q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(-1) + 0.5 * (q * (q / m).log()).sum(-1)


def jsd_max(alphas):
    """Put all the weight on a's least attended position. For each row b,
    puts a 1 at that row's smallest-attention position and gives one col
    per row, together they index one cell per row."""
    one_hot = torch.zeros_like(alphas)
    one_hot[torch.arange(alphas.shape[0]), alphas.argmin(dim=1)] = 1.0
    return jsd_rows(one_hot, alphas)


# RESULTS FILES
def test_dir(test, scheme):
    d = os.path.join(OUT_ROOT, scheme, test)
    os.makedirs(d, exist_ok=True)
    return d


def csv_path(test, scheme, kind, variant, seed, mode=None):
    mode = mode or ATTENTION_MODE
    return os.path.join(test_dir(test, scheme), f"{kind}_{variant}_{mode}_seed{seed}.csv")


def seed_csvs(test, scheme, kind, variant, mode=None):
    """{seed: path} for every per-seed CSV of one model."""
    mode = mode or ATTENTION_MODE
    pat = os.path.join(OUT_ROOT, scheme, test, f"{kind}_{variant}_{mode}_seed*.csv")
    out = {}
    for p in sorted(glob.glob(pat)):
        try:
            out[int(p.rsplit("_seed", 1)[1][:-4])] = p
        except ValueError:
            pass
    return out


def load_seed_frames(test, scheme, kind, variant, mode=None):
    """Load every seeds table and line them up."""
    import pandas as pd
    frames = {s: pd.read_csv(p) for s, p in seed_csvs(test, scheme, kind, variant, mode).items()}
    frames = {s: f for s, f in frames.items() if len(f) and "idx" in f.columns}
    if not frames:
        return {}
    common = set.intersection(*[set(f["idx"]) for f in frames.values()])
    return {s: f[f["idx"].isin(common)].drop_duplicates("idx").sort_values("idx").reset_index(drop=True)
            for s, f in frames.items()}


# UNCERTAINTY: seeds + bootstrap over test instances
def seed_stat(per_seed_arrays, stat=np.nanmean, n_boot=None, rng_seed=0):
    """Point estimate: average the metric across training seeds.
    sd_seeds: STD across seeds, measuring training variability.
    ci95: 95% bootstrap CI from resampling test instances with replacement.
    	The same samples instances are used for every seed.
    per_seed_arrays: one aligned array of test-instance values per seed."""
    arrs = [np.asarray(a, dtype=float) for a in per_seed_arrays if len(a)]
    if not arrs:
        return {"mean": float("nan"), "sd_seeds": float("nan"), "ci95": [float("nan")] * 2,
                "per_seed": [], "n_seeds": 0}
    vals = [float(stat(a)) for a in arrs]
    n = len(arrs[0])
    n_boot = N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ii = rng.integers(0, n, n)
        boots[b] = np.nanmean([stat(a[ii]) for a in arrs])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if n_boot else (float("nan"),) * 2
    finite = [v for v in vals if np.isfinite(v)]          # a seed can lack a value (e.g. no flips)
    return {"mean": float(np.mean(finite)) if finite else float("nan"),
            "sd_seeds": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "ci95": [float(lo), float(hi)], "per_seed": [round(v, 6) for v in vals],
            "n_seeds": len(finite)}


def fmt(st, digits=3):
    """Format a metric as a readable string:
    mean plus/minus STD across seeds."""
    if st is None or not np.isfinite(st.get("mean", np.nan)):
        return "n/a"
    f = f"{{:.{digits}f}}"
    s = f.format(st["mean"])
    if st.get("n_seeds", 0) > 1:
        s += " ± " + f.format(st["sd_seeds"])
    lo, hi = st.get("ci95", [np.nan, np.nan])
    if np.isfinite(lo):
        s += f" [{f.format(lo)}, {f.format(hi)}]"
    return s


def round_stats(d, digits=4):
    if isinstance(d, dict):
        return {k: round_stats(v, digits) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [round_stats(v, digits) for v in d]
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else round(float(d), digits)
    if isinstance(d, np.integer):
        return int(d)
    return d


def write_summary(test, payload):
    """Save the summary JSON."""
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, f"{test}_summary_{ATTENTION_MODE}.json")
    payload = round_stats(payload)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    tag = test.upper()
    print(f"\n<<<{tag}_SUMMARY_BEGIN>>>")
    print(json.dumps(payload, indent=2))
    print(f"<<<{tag}_SUMMARY_END>>>")
    print(f"  summary -> {path}", flush=True)
    return path


# FIGURE HELPERS
IEEE_COL_WIDTH  = 3.5
IEEE_FULL_WIDTH = 7.16
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.titlesize": 9, "font.family": "serif",
    "axes.linewidth": 0.6, "lines.linewidth": 0.9,
})
BLUES = ["#c6dbef", "#6baed6", "#2171b5", "#0D4293"]


def scheme_pair_grid(title):
    """One full-width figure: top row = flat variants, bottom row = hierarchical."""
    ncols = max(len(FLAT_VARIANTS), len(HIER_VARIANTS), 1)
    fig, axes = plt.subplots(2, ncols, figsize=(IEEE_FULL_WIDTH, 0.6 * IEEE_FULL_WIDTH),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    axd = {}
    for row, (kind, variants) in enumerate((("flat", FLAT_VARIANTS), ("hier", HIER_VARIANTS))):
        for col, v in enumerate(variants):
            ax = axes[row][col]
            ax.set_visible(True)
            ax.set_title(pretty(kind, v))
            axd[(kind, v)] = ax
    fig.suptitle(title)
    return fig, axes, axd


def wide_grid(schemes, ylabel):
    """Paper-style figure: rows = schemes, columns = flat block | gap | hier block."""
    cols = [("flat", v) for v in FLAT_VARIANTS] + [("gap", None)] + [("hier", v) for v in HIER_VARIANTS]
    wr = [0.25 if k == "gap" else 1.0 for k, _ in cols]
    fig, axes = plt.subplots(len(schemes), len(cols),
                             figsize=(IEEE_FULL_WIDTH, 1.15 * len(schemes) + 0.7),
                             squeeze=False, sharey="row", gridspec_kw={"width_ratios": wr})
    axd = {}
    for r, scheme in enumerate(schemes):
        first = True
        for c, (kind, v) in enumerate(cols):
            ax = axes[r][c]
            if kind == "gap":
                ax.set_visible(False)
                continue
            axd[(scheme, kind, v)] = ax
            ax.grid(alpha=0.25, axis="y", linewidth=0.4)
            if first:
                ax.set_ylabel(f"{scheme}\n{ylabel}", fontsize=7)
                first = False
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
    return fig, axes, axd


def save_fig(fig, path, rect=(0, 0, 1, 0.96)):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout(rect=list(rect))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}", flush=True)
    return path

