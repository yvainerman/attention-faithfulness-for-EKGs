"""
train_hierarchical_seq.py
-------------------------
ONE trainer for the HIERARCHICAL models in models/hierarchical_seq.py on PTB-XL,
MULTI-LABEL, under any of the three SCP label schemes (diagnostic / rhythm /
form). The hierarchical counterpart to train_flat_seq.py.

WHAT ONE RECORDING GOES THROUGH
  12 x 5000 signal (denoised, baseline-removed and z-scored by data/ptbxl.py --
  the log prints which as `preprocessing=`)
    -> beats: 300-sample (600 ms) windows around each R-peak (data/sph_beats.py)
    -> beat encoder (CNN, or BiLSTM for bilstm_full) -> beat attention -> 1 vector/beat
    -> reach encoder over beats: noenc (none) | conv (+/-2 beats) | bilstm (all) -> x 512
    -> sequence attention over beats -> 512 -> linear -> num_classes logits

MULTI-LABEL: per-class sigmoid + BCEWithLogitsLoss with per-class pos_weight.
Threshold-free macro-AUC on the validation fold (fold 9) picks the checkpoint;
macro/micro-F1 at 0.5 are logged. The test fold (10) is NOT used here -- that is
test_hierarchical_seq.py.

SETTINGS (env vars; same names as the flat trainer)
  ECG_MODEL_VARIANT      noenc | conv | bilstm_cnn | bilstm_full
  ECG_LABEL_SCHEME       diagnostic | rhythm | form
  ECG_DIAG_LEVEL         superclass | subclass | statement
  ECG_UNIFORM_ATTENTION  1 -> W&P baseline: SEQUENCE attention frozen to 1/S, retrained
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
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score

from data.ptbxl import PREP_TAG
from data.sph_beats import (
    load_sph_beats, HANBeatsDataset, han_collate, LengthBucketBatchSampler,
    ptbxl_beat_split_indices,
)
from models.hierarchical_seq import HierarchicalSeqAttention

from config import DATA_PATHS


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# FLAGS
def _env(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default

def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return str(v).strip().lower() not in ("0", "false", "no")

MODEL_VARIANT = _env("ECG_MODEL_VARIANT", "noenc")     # noenc|conv|bilstm_cnn|bilstm_full
LABEL_SCHEME  = _env("ECG_LABEL_SCHEME",  "rhythm")    # diagnostic|rhythm|form
DIAG_LEVEL    = _env("ECG_DIAG_LEVEL", "superclass")   # superclass|subclass|statement
USE_FILTERED  = _env_bool("ECG_USE_FILTERED", True)
UNIFORM       = _env_bool("ECG_UNIFORM_ATTENTION", False)   # W&P: freeze SEQ attn to 1/S, retrain
EPOCHS        = int(_env("ECG_EPOCHS", "50"))
SEEDS         = [int(s) for s in _env("ECG_SEEDS", "42,0,1,2,3").split(",")]
NUM_WORKERS   = int(_env("ECG_NUM_WORKERS", "0"))
DATA_ROOT_ENV = _env("ECG_DATA_ROOT", None)

DATASET = "ptbxl"
FS, PRE_SAMPLES, POST_SAMPLES, MIN_BEATS = 500, 99, 201, 2   # beat = 99 + 201 = 300 samples


# SHARED optimization config
OPT = {
    "weight_decay":  1e-4,
    "batch_size":    32,
    "pos_weight":    "sqrt",      # none | inv | sqrt
    "dropout":       0.4,
}
BASE_LR        = 1e-3
PER_VARIANT_LR = {"bilstm_cnn": 5e-4, "bilstm_full": 3e-4}   # lower for recurrent
OPTIMIZER      = "AdamW"
SCHEDULER      = "cosine"

# Widths: every variant hands 512 numbers per beat to the sequence attention
# (noenc/conv: seq_dim=512; BiLSTM variants: 2 x seq_hidden = 512).
ARCH = {
    "noenc":       dict(conv_channels=256, seq_dim=512),
    "conv":        dict(conv_channels=256, seq_dim=512),
    "bilstm_cnn":  dict(conv_channels=256, seq_hidden=256),
    "bilstm_full": dict(beat_hidden=128,  seq_hidden=256),
}


def lr_for(variant):
    return PER_VARIANT_LR.get(variant, BASE_LR)


# TUNED overrides
TUNED = {
    ("diagnostic", "bilstm_cnn"): dict(lr=0.0011114989443094978, dropout=0.20823379771832098, weight_decay=0.0008123245085588687, pos_weight='none', batch_size=64),
    ("form", "bilstm_cnn"): dict(lr=0.0009682938464887533, dropout=0.20368763693190525, weight_decay=2.8642782499372967e-05, pos_weight='inv', batch_size=64),
    ("rhythm", "bilstm_cnn"): dict(lr=0.00039445002074481116, dropout=0.2567751504251358, weight_decay=0.0005266970094621577, pos_weight='inv', batch_size=16),
    ("diagnostic", "bilstm_full"): dict(lr=0.0024481736103195615, dropout=0.4083080068048108, weight_decay=8.880981938197938e-06, pos_weight='inv', batch_size=16),
    ("form", "bilstm_full"): dict(lr=0.002708525962917861, dropout=0.48137807687165135, weight_decay=8.217143602141803e-06, pos_weight='sqrt', batch_size=16),
    ("rhythm", "bilstm_full"): dict(lr=0.00297429392865237, dropout=0.3871290231193866, weight_decay=1.0402289956775619e-05, pos_weight='sqrt', batch_size=16),
    ("diagnostic", "conv"): dict(lr=0.0010463001379244791, dropout=0.3828822441444662, weight_decay=9.649552721645166e-06, pos_weight='sqrt', batch_size=16),
    ("form", "conv"): dict(lr=0.0010449774094005496, dropout=0.22000408316972264, weight_decay=5.953251686451287e-06, pos_weight='sqrt', batch_size=32),
    ("rhythm", "conv"): dict(lr=0.0013154418927742784, dropout=0.22743757481987442, weight_decay=0.0005332925387379329, pos_weight='sqrt', batch_size=32),
    ("diagnostic", "noenc"): dict(lr=0.0018596250322140985, dropout=0.25505760858598997, weight_decay=0.0002961017511740937, pos_weight='inv', batch_size=16),
    ("form", "noenc"): dict(lr=0.0021410081444863524, dropout=0.2333309393780616, weight_decay=1.2286502924646356e-05, pos_weight='sqrt', batch_size=64),
    ("rhythm", "noenc"): dict(lr=0.00032384612969363005, dropout=0.22752604667189388, weight_decay=0.00011648329722964577, pos_weight='inv', batch_size=16),
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


CFG = run_config(LABEL_SCHEME, MODEL_VARIANT)


def resolve_data_path():
    root = DATA_ROOT_ENV
    return DATA_PATHS["ptbxl"] if root is None else os.path.join(root, "ptbxl")


# HELPERS
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_pos_weight(scheme, train_label_matrix):
    """Per-class BCE pos_weight from train positives. (N,C) float -> (C,) tensor."""
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
    w = np.clip(w, 0.0, 50.0)
    return torch.tensor(w, dtype=torch.float32)


def _uniform_forward_3d(features):
    """Replace AdditiveAttention.forward with fixed 1/T weights."""
    B, T, _ = features.shape
    w = torch.full((B, T, 1), 1.0 / T, device=features.device, dtype=features.dtype)
    return torch.sum(w * features, dim=1), w


def apply_uniform_seq(model):
    """Freeze ONLY the sequence (final) attention layer to 1/S (S&S)."""
    for p in model.seq_attention.parameters():
        p.requires_grad_(False)
    model.seq_attention.forward = _uniform_forward_3d
    return model


def build_model(variant, num_classes):
    model = HierarchicalSeqAttention(
        num_classes=num_classes, variant=variant,
        dropout=CFG["dropout"], **ARCH[variant],
    ).to(DEVICE)
    if UNIFORM:
        model = apply_uniform_seq(model)   # before optimizer, frozen params excluded
    return model


def build_optimizer(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=CFG["weight_decay"])


def build_scheduler(optimizer, epochs):
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
    """Check setup."""
    item = dataset[train_idx[0]]
    S, T, C = item["beats"].shape
    if (T, C) != (PRE_SAMPLES + POST_SAMPLES, 12):
        raise ValueError(f"beats have shape (S, {T}, {C}); expected "
                         f"(S, {PRE_SAMPLES + POST_SAMPLES}, 12)")
    model = build_model(MODEL_VARIANT, num_classes).eval()
    with torch.no_grad():
        logits, seq_a, beat_a = model.forward_with_attention(item["beats"].unsqueeze(0).to(DEVICE))
    if tuple(logits.shape) != (1, num_classes) or tuple(seq_a.shape) != (1, S) \
            or tuple(beat_a.shape[:2]) != (1, S):
        raise RuntimeError(f"shape check failed: logits {tuple(logits.shape)}, "
                           f"seq_alphas {tuple(seq_a.shape)}, beat_alphas {tuple(beat_a.shape)}")
    params = {
        "total":     sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "beat_enc":  sum(p.numel() for p in model.beat_encoder.parameters()),
        "reach_enc": sum(p.numel() for p in model.seq_encoder.parameters()),
    }
    t_red = int(beat_a.shape[2])
    del model
    return t_red, params


# TRAIN / EVAL
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        beats  = batch["beats"].to(DEVICE)
        labels = batch["label"].to(DEVICE).float()         # (B, C) multi-hot
        optimizer.zero_grad()
        loss = criterion(model(beats), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * labels.size(0); n += labels.size(0)
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
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
    pred = (p >= 0.5).astype(int)
    auc = macro_auc(y, p)
    f1_macro = f1_score(y, pred, average="macro", zero_division=0)
    f1_micro = f1_score(y, pred, average="micro", zero_division=0)
    return float(np.mean(losses)), auc, f1_macro, f1_micro


def build_loaders(dataset, train_idx, val_idx, batch_size, seed, num_workers):
    train_lengths = [dataset.num_beats[i] for i in train_idx]
    val_lengths   = [dataset.num_beats[i] for i in val_idx]
    train_sampler = LengthBucketBatchSampler(lengths=train_lengths, batch_size=batch_size,
                                             drop_last=True, shuffle=True, seed=seed)
    val_sampler = LengthBucketBatchSampler(lengths=val_lengths, batch_size=batch_size,
                                           drop_last=False, shuffle=False)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_sampler=train_sampler,
                              num_workers=num_workers, pin_memory=True, collate_fn=han_collate)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True, collate_fn=han_collate)
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
        scheduler.step()                            # cosine: lower the LR for the next epoch
        print(f"  Epoch {epoch+1:>2}/{epochs}: train_loss={tr:.4f}  val_loss={vl:.4f}  "
              f"macroAUC={auc:.4f}  macroF1={f1m:.4f}  microF1={f1mi:.4f}", flush=True)
        if auc > best_auc:                          # select on threshold-free macro-AUC
            best_auc, best_f1, best_micro, best_epoch = auc, f1m, f1mi, epoch + 1
            save = model.module if hasattr(model, "module") else model
            torch.save(save.state_dict(), ckpt_path_seed)
    return best_auc, best_f1, best_micro, best_epoch


# DATA LOAD / SPLIT
def load_and_split(data_path):
    print(f"[load] PTB-XL beats  scheme={LABEL_SCHEME}  level={DIAG_LEVEL}  "
          f"filtered={USE_FILTERED}  preprocessing={PREP_TAG}  (MULTI-LABEL)", flush=True)
    beats_list, num_classes, label2idx = load_sph_beats(
        data_path, filtered=USE_FILTERED, fs=FS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES,
        dataset="ptbxl", label_scheme=LABEL_SCHEME, diagnostic_level=DIAG_LEVEL)
    dataset = HANBeatsDataset(beats_list, num_beats=None, min_beats=MIN_BEATS)
    print(f"[load] {len(dataset.data)} ECGs kept (min_beats={MIN_BEATS}); "
          f"classes={num_classes}; labels={label2idx}", flush=True)
    train_idx, val_idx, _ = ptbxl_beat_split_indices(
        data_path, filtered=USE_FILTERED, fs=FS,
        label_scheme=LABEL_SCHEME, diagnostic_level=DIAG_LEVEL, min_beats=MIN_BEATS,
        pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES)
    return dataset, num_classes, train_idx, val_idx


# MAIN
def main():
    if MODEL_VARIANT not in HierarchicalSeqAttention.VARIANTS:
        raise ValueError(f"ECG_MODEL_VARIANT={MODEL_VARIANT!r} not in "
                         f"{HierarchicalSeqAttention.VARIANTS}")
    if LABEL_SCHEME not in ("diagnostic", "rhythm", "form"):
        raise ValueError(f"ECG_LABEL_SCHEME={LABEL_SCHEME!r} invalid")

    torch.backends.cudnn.benchmark = True
    data_path = resolve_data_path()

    print(f"{'='*64}")
    print(f"  train_hierarchical_seq.py  (MULTI-LABEL / BCE / HAN)")
    print(f"  dataset=ptbxl  variant={MODEL_VARIANT}  label_scheme={LABEL_SCHEME}"
          f"{('/'+DIAG_LEVEL) if LABEL_SCHEME=='diagnostic' else ''}")
    print(f"  beats={PRE_SAMPLES}+{POST_SAMPLES} samples around each R-peak, "
          f"min_beats={MIN_BEATS}")
    print(f"  preprocessing={PREP_TAG}")
    print(f"  arch={ARCH[MODEL_VARIANT]}")
    print(f"  cfg={ {k: CFG[k] for k in ('lr','dropout','weight_decay','pos_weight','batch_size')} }  "
          f"optimizer={OPTIMIZER}  scheduler={SCHEDULER}")
    print(f"  tuned={CFG['tuned']}  (scheme={LABEL_SCHEME}, variant={MODEL_VARIANT})")
    print(f"  UNIFORM_ATTENTION={UNIFORM}" + ("  (seq attention frozen to 1/S)" if UNIFORM else ""))
    print(f"  filtered={USE_FILTERED}  epochs={EPOCHS}  seeds={SEEDS}  workers={NUM_WORKERS}")
    print(f"  data_path={data_path}")
    print(f"{'='*64}", flush=True)

    dataset, num_classes, train_idx, val_idx = load_and_split(data_path)

    t_red, params = check_setup(dataset, train_idx, num_classes)
    print(f"[check] beats (S, {PRE_SAMPLES + POST_SAMPLES}, 12) -> {t_red} positions per beat; "
          f"logits/attention shapes OK")
    print(f"[check] params total={params['total']:,}  trainable={params['trainable']:,}  "
          f"beat_enc={params['beat_enc']:,}  reach_enc={params['reach_enc']:,}", flush=True)

    train_labels = np.stack([np.asarray(dataset.data[i][1], dtype=np.float32)
                             for i in train_idx], axis=0)     # (N_train, C)
    pos_weight = make_pos_weight(CFG["pos_weight"], train_labels)
    pos_counts = train_labels.sum(axis=0).astype(int)
    print(f"num_classes={num_classes}  train={len(train_idx)}  val={len(val_idx)}")
    print(f"train positives/class: {pos_counts.tolist()}")
    print(f"pos_weight: {np.round(pos_weight.numpy(), 3).tolist()}\n", flush=True)

    CKPT_DIR = "checkpoints/seq_uniform" if UNIFORM else "checkpoints/seq"
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
    print(f"  Summary  ptbxl / HIER-{MODEL_VARIANT} / {LABEL_SCHEME}  (multi-label, val fold)")
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
        "script": "train_hierarchical_seq.py",
        "variant": MODEL_VARIANT, "scheme": LABEL_SCHEME,
        "diag_level": DIAG_LEVEL if LABEL_SCHEME == "diagnostic" else None,
        "filtered": USE_FILTERED, "uniform_attention": UNIFORM,
        "beats": {"pre_samples": PRE_SAMPLES, "post_samples": POST_SAMPLES,
                  "min_beats": MIN_BEATS, "positions_per_beat": t_red},
        "preprocessing": PREP_TAG,
        "arch": ARCH[MODEL_VARIANT], "cfg": CFG, "epochs": EPOCHS,
        "optimizer": OPTIMIZER, "scheduler": SCHEDULER,
        "params": params,
        "num_classes": num_classes,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "val_results": [{k: r[k] for k in ("seed", "auc", "f1", "micro", "best_epoch")}
                        for r in results],
        "best_seed": best["seed"],
    }
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  run record -> {stem}_meta.json")
    print(f"{'='*64}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()