"""
train_flat_seq.py
-----------------
ONE trainer for the FLAT (Serrano & Smith FLAN) models in models/flat_seq.py on
PTB-XL, MULTI-LABEL, under any of the three SCP label schemes (diagnostic /
rhythm / form). The flat counterpart to train_hierarchical_seq.py.

WHAT ONE RECORDING GOES THROUGH
  12 x 5000 signal (10 s @ 500 Hz; denoised, baseline-removed and z-scored by
  data/ptbxl.py)
    -> PatchCNNEncoder : 100 NON-overlapping 50-sample (100 ms) tokens x 256
    -> reach encoder   : noenc (none) | conv (+/-2 tokens) | bilstm (all) -> x 512
    -> ONE attention over the 100 tokens -> 512 -> linear -> num_classes logits

OPTIMISATION: shared defaults (OPT; bilstm gets a lower learning rate),
overridden per (scheme, variant) by the TUNED cells that tune_flat_seq.py
prints. With TUNED empty, every variant uses the shared defaults.

SETTINGS (env vars; same names as the hierarchical trainer)
  ECG_MODEL_VARIANT      noenc | conv | bilstm
  ECG_LABEL_SCHEME       diagnostic | rhythm | form
  ECG_DIAG_LEVEL         superclass | subclass | statement
  ECG_UNIFORM_ATTENTION  1 -> W&P baseline: attention frozen to 1/100, retrained
  ECG_EPOCHS, ECG_SEEDS, ECG_NUM_WORKERS, ECG_DATA_ROOT, ECG_USE_FILTERED
  (preprocessing flags ECG_DENOISE / ECG_REMOVE_BASELINE / ECG_ZSCORE are read
   by data/ptbxl.py, not here)
"""
import json
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score

from data.ptbxl import load_ptbxl_dataset, ptbxl_split_indices, PREP_TAG
from models.flat_seq import FlatSeqAttention, PATCH

from config import DATA_PATHS


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FLAGS  (defaults; each overridable by the matching env var)
def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default

def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() not in ("0", "false", "no")

MODEL_VARIANT = _env("ECG_MODEL_VARIANT", "noenc")     # noenc|conv|bilstm
LABEL_SCHEME  = _env("ECG_LABEL_SCHEME",  "rhythm")    # diagnostic|rhythm|form
DIAG_LEVEL    = _env("ECG_DIAG_LEVEL", "superclass")   # superclass|subclass|statement
USE_FILTERED  = _env_bool("ECG_USE_FILTERED", True)
UNIFORM       = _env_bool("ECG_UNIFORM_ATTENTION", False)   # W&P: freeze attn to 1/T, retrain
EPOCHS        = int(_env("ECG_EPOCHS", "50"))
SEEDS         = [int(s) for s in _env("ECG_SEEDS", "42,0,1,2,3").split(",")]
NUM_WORKERS   = int(_env("ECG_NUM_WORKERS", "0"))
DATA_ROOT_ENV = _env("ECG_DATA_ROOT", None)

DATASET = "ptbxl"
FS = 500            # 500 samples a second

# Token params, read from the model file so the two can never disagree.
TOKEN_LEN = PATCH[0] * PATCH[1]          # samples per token (50)
TOKEN_MS  = TOKEN_LEN * 1000.0 / FS      # 100 ms

# SHARED optimisation config
OPT = {
    "weight_decay":  1e-4,
    "batch_size":    32,
    "pos_weight":    "sqrt",      # none | inv | sqrt
    "dropout":       0.4,
}
BASE_LR        = 1e-3
PER_VARIANT_LR = {"bilstm": 5e-4}     # lower for the recurrent encoder
OPTIMIZER      = "AdamW"     # fixed for every model (not tuned)
SCHEDULER      = "cosine"    # CosineAnnealingLR over EPOCHS (not tuned)

# Widths: every variant hands 512 numbers per token to the attention
# (noenc/conv: seq_dim=512; bilstm: 2 x seq_hidden = 512). The extractor
# always outputs conv_channels=256 per token.
ARCH = {
    "noenc":  dict(conv_channels=256, seq_dim=512),
    "conv":   dict(conv_channels=256, seq_dim=512),
    "bilstm": dict(conv_channels=256, seq_hidden=256),
}


def lr_for(variant):
    return PER_VARIANT_LR.get(variant, BASE_LR)


# TUNED overrides, filled from tune_flat_seq.py
TUNED = {
    ("diagnostic", "bilstm"): dict(lr=0.0019391672607974029, dropout=0.3947067020558175, weight_decay=0.0001358466077805578, pos_weight='inv', batch_size=16),
    ("form", "bilstm"): dict(lr=0.0015527760085742605, dropout=0.4135392697191156, weight_decay=2.2731423531248504e-05, pos_weight='inv', batch_size=16),
    ("rhythm", "bilstm"): dict(lr=0.0006228910145246475, dropout=0.21645754395557798, weight_decay=7.027344354253187e-05, pos_weight='sqrt', batch_size=32),
    ("diagnostic", "conv"): dict(lr=0.0016682175869003535, dropout=0.2753954850612579, weight_decay=3.265819189638202e-06, pos_weight='sqrt', batch_size=64),
    ("form", "conv"): dict(lr=0.0028246966604966732, dropout=0.28948709462176414, weight_decay=0.0009015583351720496, pos_weight='sqrt', batch_size=32),
    ("rhythm", "conv"): dict(lr=0.0025766592518596323, dropout=0.3806295057082611, weight_decay=0.0002989125842538722, pos_weight='none', batch_size=16),
    ("diagnostic", "noenc"): dict(lr=0.0013999778375813584, dropout=0.2282572457368231, weight_decay=1.2104799314730345e-06, pos_weight='sqrt', batch_size=32),
    ("form", "noenc"): dict(lr=0.001834555855587948, dropout=0.21912155682953316, weight_decay=0.00013699155313810334, pos_weight='sqrt', batch_size=64),
    ("rhythm", "noenc"): dict(lr=0.0008671461623228557, dropout=0.20185957251124836, weight_decay=2.8642782499372967e-05, pos_weight='none', batch_size=32),
}


def run_config(scheme, variant):
    """Effective config for one (scheme, variant): shared defaults overridden by
    the TUNED cell if present."""
    cfg = {
        "lr":           lr_for(variant),
        "dropout":      OPT["dropout"],
        "weight_decay": OPT["weight_decay"],
        "pos_weight":   OPT["pos_weight"],
        "batch_size":   OPT["batch_size"],
        "tuned":        False,
    }
    cell = TUNED.get((scheme, variant))
    if cell:
        cfg.update(cell)
        cfg["tuned"] = True
    return cfg


# look up config for run, and overwrite all settings if found
CFG = run_config(LABEL_SCHEME, MODEL_VARIANT)

# set data path
def resolve_data_path():
    root = DATA_ROOT_ENV
    return DATA_PATHS["ptbxl"] if root is None else os.path.join(root, "ptbxl")


# Whole-signal dataset (fixed length -> no padding / bucketing)
class FlatECGDataset(Dataset):
    """Wraps a multi-label data_list of (signal[12,T], multihot[C], pid)."""
    def __init__(self, data_list):
        self.data = data_list

    # amount of recordings
    def __len__(self): 
        return len(self.data)

    # return recording i as two tensors: signal (shape), label (multi-hot). pid ignored.
    def __getitem__(self, i):
        sig, label, _pid = self.data[i]
        return {
            "signal": torch.as_tensor(np.asarray(sig), dtype=torch.float32),   # (12,T)
            "label":  torch.as_tensor(np.asarray(label), dtype=torch.float32), # (C,)
        }

# batching
def flat_collate(batch):
    return {
        "signal": torch.stack([b["signal"] for b in batch], dim=0),   # (B,12,T)
        "label":  torch.stack([b["label"]  for b in batch], dim=0),   # (B,C)
    }


# HELPERS
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# increase weighting for rarer classes
def make_pos_weight(scheme, train_label_matrix):
    pos = train_label_matrix.sum(axis=0)
    neg = train_label_matrix.shape[0] - pos
    pos = np.clip(pos, 1.0, None)
    ratio = neg / pos
    if scheme == "none":
        w = np.ones_like(ratio)
    elif scheme == "inv":
        w = ratio
    elif scheme == "sqrt":
        w = np.sqrt(ratio)
    else:
        raise ValueError(f"Unknown pos_weight scheme: {scheme}")
    w = np.clip(w, 0.0, 50.0)       # cap at 50, so ultra-rare classes don't dominate
    return torch.tensor(w, dtype=torch.float32)     # tensor single weight per class


def _uniform_forward_3d(features):
    """Replace AdditiveAttention.forward with fixed 1/T weights. Freezes
    the attention layer for the W&P uniform-necessity baseline. """
    B, T, _ = features.shape
    w = torch.full((B, T, 1), 1.0 / T, device=features.device, dtype=features.dtype)
    return torch.sum(w * features, dim=1), w


def apply_uniform(model):
    for p in model.attention.parameters():
        p.requires_grad_(False)
    model.attention.forward = _uniform_forward_3d
    return model


def build_model(variant, num_classes):
    model = FlatSeqAttention(
        num_classes=num_classes, variant=variant,
        dropout=CFG["dropout"], **ARCH[variant],
    ).to(DEVICE)
    if UNIFORM:
        model = apply_uniform(model)   # before optimizer, frozen params excluded
    return model

def build_optimizer(model):
    """AdamW over the TRAINABLE weights only (in uniform mode the frozen
    attention weights are left out). Same optimizer as the tuner."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=CFG["weight_decay"])

# TODO: step/plateau removal?
def build_scheduler(optimizer, epochs):
    """Cosine annealing: the learning rate glides from CFG["lr"] down to ~0 over
    `epochs` along half a cosine wave. Same schedule as the tuner."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def macro_auc(y_true, y_score):
    aucs = []
    for c in range(y_true.shape[1]):
        yc = y_true[:, c]
        if yc.min() == yc.max():
            continue
        aucs.append(roc_auc_score(yc, y_score[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def check_setup(dataset, train_idx, num_classes):
    """Confirm all is in agreement. Check shapes, tokens, run one
    real recording for testing. Check output for classes and attn
    weights."""
    lengths = {np.asarray(d[0]).shape for d in dataset.data}
    if len(lengths) != 1:
        raise ValueError(f"recordings have different shapes {sorted(lengths)}; "
                         f"the flat model needs one fixed length")
    C, T = next(iter(lengths))
    if C != 12:
        raise ValueError(f"expected 12 leads in shape (12, T), got {(C, T)}")
    if T % TOKEN_LEN != 0:
        raise ValueError(f"T={T} is not a multiple of the {TOKEN_LEN}-sample token; "
                         f"the last {T % TOKEN_LEN} samples would be silently dropped")
    n_tokens = T // TOKEN_LEN

    model = build_model(MODEL_VARIANT, num_classes).eval()
    x = dataset[train_idx[0]]["signal"].unsqueeze(0).to(DEVICE)       # (1,12,T)
    with torch.no_grad():
        logits, alphas = model.forward_with_attention(x)
    if tuple(logits.shape) != (1, num_classes) or tuple(alphas.shape) != (1, n_tokens):
        raise RuntimeError(f"shape check failed: logits {tuple(logits.shape)}, "
                           f"alphas {tuple(alphas.shape)}; expected (1,{num_classes}), "
                           f"(1,{n_tokens})")
    params = {
        "total":     sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "extractor": sum(p.numel() for p in model.encoder.parameters()),
        "reach_enc": sum(p.numel() for p in model.seq_encoder.parameters()),
    }
    del model
    return T, n_tokens, params


# TRAIN / EVAL
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        signal = batch["signal"].to(DEVICE)
        labels = batch["label"].to(DEVICE).float()
        optimizer.zero_grad()
        loss = criterion(model(signal), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * labels.size(0); n += labels.size(0)
    return total / max(1, n)


@torch.no_grad()        # don't track gradients
def validate(model, loader, criterion):
    model.eval()
    losses, ys, ps = [], [], []
    for batch in loader:
        signal = batch["signal"].to(DEVICE)
        labels = batch["label"].to(DEVICE).float()
        logits = model(signal)
        losses.append(criterion(logits, labels).item())
        ys.append(labels.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    pred = (p >= 0.5).astype(int)
    auc = macro_auc(y, p)
    # prevent crash with zero_division for classes that are never predicted
    f1_macro = f1_score(y, pred, average="macro", zero_division=0)
    f1_micro = f1_score(y, pred, average="micro", zero_division=0)
    return float(np.mean(losses)), auc, f1_macro, f1_micro


def build_loaders(dataset, train_idx, val_idx, batch_size, seed, num_workers):
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True, generator=g, drop_last=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=flat_collate)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True,
                            collate_fn=flat_collate)
    return train_loader, val_loader


def train_one_seed(seed, dataset, train_idx, val_idx, num_classes, pos_weight,
                   epochs, num_workers, ckpt_path_seed):
    set_seed(seed)
    train_loader, val_loader = build_loaders(
        dataset, train_idx, val_idx, CFG["batch_size"], seed, num_workers)
    model = build_model(MODEL_VARIANT, num_classes)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, epochs)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    best_auc, best_f1, best_micro, best_epoch = -1.0, 0.0, 0.0, 0
    for epoch in range(epochs):
        tr = train_epoch(model, train_loader, criterion, optimizer)
        vl, auc, f1m, f1mi = validate(model, val_loader, criterion)
        scheduler.step()
        print(f"  Epoch {epoch+1:>2}/{epochs}: train_loss={tr:.4f}  val_loss={vl:.4f}  "
              f"macroAUC={auc:.4f}  macroF1={f1m:.4f}  microF1={f1mi:.4f}", flush=True)
        if auc > best_auc:
            best_auc, best_f1, best_micro, best_epoch = auc, f1m, f1mi, epoch + 1
            save = model.module if hasattr(model, "module") else model
            torch.save(save.state_dict(), ckpt_path_seed)
    return best_auc, best_f1, best_micro, best_epoch


# DATA LOAD & SPLIT
def load_and_split(data_path):
    print(f"[load] PTB-XL whole-signal  scheme={LABEL_SCHEME}  level={DIAG_LEVEL}  "
          f"filtered={USE_FILTERED}  preprocessing={PREP_TAG}  (MULTI-LABEL, FLAT)",
          flush=True)
    data_list, num_classes, label2idx = load_ptbxl_dataset(
        data_path, filtered=USE_FILTERED, fs=FS,
        label_scheme=LABEL_SCHEME, diagnostic_level=DIAG_LEVEL)
    dataset = FlatECGDataset(data_list)
    train_idx, val_idx, _ = ptbxl_split_indices(
        data_path, filtered=USE_FILTERED, fs=FS,
        label_scheme=LABEL_SCHEME, diagnostic_level=DIAG_LEVEL)
    print(f"[load] {len(dataset)} ECGs; classes={num_classes}; labels={label2idx}",
          flush=True)
    return dataset, num_classes, label2idx, train_idx, val_idx


# MAIN
def main():
    if MODEL_VARIANT not in FlatSeqAttention.VARIANTS:
        raise ValueError(f"ECG_MODEL_VARIANT={MODEL_VARIANT!r} not in "
                         f"{FlatSeqAttention.VARIANTS}")
    if LABEL_SCHEME not in ("diagnostic", "rhythm", "form"):
        raise ValueError(f"ECG_LABEL_SCHEME={LABEL_SCHEME!r} invalid")

    torch.backends.cudnn.benchmark = True
    data_path = resolve_data_path()

    print(f"{'='*64}")
    print(f"  train_flat_seq.py  (MULTI-LABEL / BCE / FLAN)")
    print(f"  dataset=ptbxl  variant={MODEL_VARIANT}  label_scheme={LABEL_SCHEME}"
          f"{('/'+DIAG_LEVEL) if LABEL_SCHEME=='diagnostic' else ''}")
    print(f"  extractor=PatchCNNEncoder  patch={PATCH} -> {TOKEN_LEN}-sample "
          f"({TOKEN_MS:.0f} ms) tokens, no overlap")
    print(f"  preprocessing={PREP_TAG}")
    print(f"  arch={ARCH[MODEL_VARIANT]}")
    print(f"  cfg={ {k: CFG[k] for k in ('lr','dropout','weight_decay','pos_weight','batch_size')} }  "
        f"optimizer={OPTIMIZER}  scheduler={SCHEDULER}")
    print(f"  tuned={CFG['tuned']}  (scheme={LABEL_SCHEME}, variant={MODEL_VARIANT})")
    print(f"  UNIFORM_ATTENTION={UNIFORM}" + ("  (attention frozen to 1/T)" if UNIFORM else ""))
    print(f"  filtered={USE_FILTERED}  epochs={EPOCHS}  seeds={SEEDS}  workers={NUM_WORKERS}")
    print(f"  data_path={data_path}")
    print(f"{'='*64}", flush=True)

    dataset, num_classes, label2idx, train_idx, val_idx = load_and_split(data_path)

    T, n_tokens, params = check_setup(dataset, train_idx, num_classes)
    print(f"[check] T={T} samples -> {n_tokens} tokens of {TOKEN_LEN} samples; "
          f"logits/attention shapes OK")
    print(f"[check] params total={params['total']:,}  trainable={params['trainable']:,}  "
          f"extractor={params['extractor']:,}  reach_enc={params['reach_enc']:,}",
          flush=True)

    train_labels = np.stack([np.asarray(dataset.data[i][1], dtype=np.float32)
                             for i in train_idx], axis=0)
    pos_weight = make_pos_weight(CFG["pos_weight"], train_labels)
    pos_counts = train_labels.sum(axis=0).astype(int)
    print(f"num_classes={num_classes}  train={len(train_idx)}  val={len(val_idx)}")
    print(f"train positives/class: {pos_counts.tolist()}")
    print(f"pos_weight: {np.round(pos_weight.numpy(), 3).tolist()}\n", flush=True)

    CKPT_DIR = "checkpoints/flat_seq_uniform" if UNIFORM else "checkpoints/flat_seq"
    os.makedirs(CKPT_DIR, exist_ok=True)
    suffix = "_filtered" if USE_FILTERED else ""
    stem = f"{CKPT_DIR}/best_{MODEL_VARIANT}_{LABEL_SCHEME}{suffix}"

    results = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n=== seed {seed} ({i}/{len(SEEDS)})  "
              f"variant={MODEL_VARIANT} scheme={LABEL_SCHEME} ===", flush=True)
        ckpt_seed = f"{stem}_seed{seed}.pth"
        auc, f1m, f1mi, ep = train_one_seed(seed, dataset, train_idx, val_idx, num_classes,
                                            pos_weight, EPOCHS, NUM_WORKERS, ckpt_seed)
        results.append({"seed": seed, "auc": auc, "f1": f1m, "micro": f1mi,
                        "best_epoch": ep, "ckpt": ckpt_seed})
        print(f"  seed {seed} best (epoch {ep}): macroAUC={auc:.4f}  macroF1={f1m:.4f}  "
              f"microF1={f1mi:.4f}", flush=True)

    aucs = np.array([r["auc"] for r in results])
    f1s  = np.array([r["f1"]  for r in results])
    mics = np.array([r["micro"] for r in results])
    print(f"\n{'='*64}")
    print(f"  Summary  ptbxl / FLAT-{MODEL_VARIANT} / {LABEL_SCHEME}  (multi-label, val fold)")
    for r in results:
        print(f"    seed {r['seed']:>3}: macroAUC={r['auc']:.4f}  "
              f"macroF1={r['f1']:.4f}  microF1={r['micro']:.4f}  (epoch {r['best_epoch']})")
    print(f"    macro-AUC: {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"    macro-F1 : {f1s.mean():.4f} +/- {f1s.std():.4f}")
    print(f"    micro-F1 : {mics.mean():.4f} +/- {mics.std():.4f}")

    best = max(results, key=lambda r: r["auc"])
    shutil.copy(best["ckpt"], f"{stem}.pth")
    print(f"  best seed: {best['seed']} (macroAUC={best['auc']:.4f})  ->  {stem}.pth")

    meta = {
        "script": "train_flat_seq.py",
        "variant": MODEL_VARIANT, "scheme": LABEL_SCHEME,
        "diag_level": DIAG_LEVEL if LABEL_SCHEME == "diagnostic" else None,
        "filtered": USE_FILTERED, "uniform_attention": UNIFORM,
        "extractor": {"name": "PatchCNNEncoder", "patch": list(PATCH),
                      "token_samples": TOKEN_LEN, "token_ms": TOKEN_MS,
                      "overlap_samples": 0, "tokens_per_recording": n_tokens},
        "preprocessing": PREP_TAG,
        "arch": ARCH[MODEL_VARIANT], "cfg": CFG, "epochs": EPOCHS,
        "params": params,
        "num_classes": num_classes, "label2idx": label2idx,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "val_results": [{k: r[k] for k in ("seed", "auc", "f1", "micro", "best_epoch")}
                        for r in results],
        "best_seed": best["seed"],
        "optimizer": OPTIMIZER, "scheduler": SCHEDULER,
    }
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  run record -> {stem}_meta.json")
    print(f"{'='*64}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()