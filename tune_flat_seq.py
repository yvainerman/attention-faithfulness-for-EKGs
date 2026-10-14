"""
tune_flat_seq.py
----------------
Optuna hyperparameter search for ONE (variant, scheme) of the multi-label FLAT
pipeline. Mirrors tune_hierarchical_seq.py exactly, but imports the FLAT trainer
(train_flat_seq.py) so it reuses that pipeline's data loading (cleaned PTB-XL
signals), model construction inputs (ARCH), loss, training step and multi-label
`validate` (search optimises the SAME metric the paper reports -- threshold-free
macro-AUC on the validation fold).

Axes selected by env (same names as the trainer):
    ECG_MODEL_VARIANT  noenc | conv | bilstm
    ECG_LABEL_SCHEME   diagnostic | rhythm | form
    ECG_DIAG_LEVEL     superclass | subclass | statement

WHAT IS TUNED (optimisation only): lr, dropout, weight_decay, pos_weight, batch_size
-- exactly the five keys of a TUNED cell in train_flat_seq.py.
Architecture WIDTH (ARCH) is held FIXED -- the study compares reach with capacity
constant, so tuning width would reintroduce the confound the design removes.
Only the normal (learned-attention) model is tuned; the uniform-attention
baseline reuses its settings.

Search config (env-overridable):
    ECG_TUNE_TRIALS   (default 50)
    ECG_TUNE_EPOCHS   (default 15)
    ECG_TUNE_SEED     (default 42)
    ECG_TUNE_STORAGE  (optional, e.g. sqlite:///tune_results/flat_conv_form.db)
    ECG_NUM_WORKERS   (default 4)
"""
import os
import json

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# IMPORTANT: env (ECG_MODEL_VARIANT / ECG_LABEL_SCHEME / ...) must be set BEFORE
# this import so the trainer's module-level globals match this run.
from train_flat_seq import (
    MODEL_VARIANT, LABEL_SCHEME, DIAG_LEVEL, USE_FILTERED, UNIFORM, ARCH, DEVICE,
    PREP_TAG, TOKEN_LEN,
    resolve_data_path, load_and_split, build_loaders, make_pos_weight,
    train_epoch, validate, set_seed, check_setup,
)
from models.flat_seq import FlatSeqAttention


def _envint(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default

TUNE_TRIALS  = _envint("ECG_TUNE_TRIALS", 50)
TUNE_EPOCHS  = _envint("ECG_TUNE_EPOCHS", 15)
TUNE_SEED    = _envint("ECG_TUNE_SEED", 42)
NUM_WORKERS  = _envint("ECG_NUM_WORKERS", 4)
STORAGE      = os.environ.get("ECG_TUNE_STORAGE") or None

OUT_DIR = "tune_results"


def make_objective(dataset, num_classes, train_idx, val_idx, train_labels):
    def objective(trial):
        lr           = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        dropout      = trial.suggest_float("dropout", 0.2, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        pw_scheme    = trial.suggest_categorical("pos_weight", ["none", "inv", "sqrt"])
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

        set_seed(TUNE_SEED)
        train_loader, val_loader = build_loaders(
            dataset, train_idx, val_idx, batch_size, TUNE_SEED, NUM_WORKERS)

        model = FlatSeqAttention(
            num_classes=num_classes, variant=MODEL_VARIANT,
            dropout=dropout, **ARCH[MODEL_VARIANT]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TUNE_EPOCHS)
        pos_weight = make_pos_weight(pw_scheme, train_labels).to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auc, best_epoch = -1.0, 0
        for epoch in range(TUNE_EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer)
            vl, auc, *_ = validate(model, val_loader, criterion)   # macro-AUC
            scheduler.step()
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch + 1
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_epoch", best_epoch)
        return best_auc
    return objective


def main():
    if MODEL_VARIANT not in FlatSeqAttention.VARIANTS:
        raise ValueError(f"ECG_MODEL_VARIANT={MODEL_VARIANT!r} invalid "
                         f"(flat variants: {FlatSeqAttention.VARIANTS})")
    if LABEL_SCHEME not in ("diagnostic", "rhythm", "form"):
        raise ValueError(f"ECG_LABEL_SCHEME={LABEL_SCHEME!r} invalid")
    if UNIFORM:
        raise ValueError("ECG_UNIFORM_ATTENTION is set: only the normal "
                         "(learned-attention) model is tuned; unset it.")

    torch.backends.cudnn.benchmark = True
    data_path = resolve_data_path()
    scheme_tag = f"diagnostic-{DIAG_LEVEL}" if LABEL_SCHEME == "diagnostic" else LABEL_SCHEME
    tag = f"flat_{MODEL_VARIANT}_{scheme_tag}" + ("" if USE_FILTERED else "_all")

    print(f"{'='*64}")
    print(f"  tune_flat_seq.py  (MULTI-LABEL / macro-AUC)")
    print(f"  study={tag}")
    print(f"  preprocessing={PREP_TAG}  tokens={TOKEN_LEN} samples")
    print(f"  arch(FIXED)={ARCH[MODEL_VARIANT]}")
    print(f"  trials={TUNE_TRIALS}  epochs/trial={TUNE_EPOCHS}  seed={TUNE_SEED}")
    print(f"  storage={STORAGE or 'in-memory'}")
    print(f"{'='*64}", flush=True)

    dataset, num_classes, _label2idx, train_idx, val_idx = load_and_split(data_path)
    T, n_tokens, params = check_setup(dataset, train_idx, num_classes)   # same check as the trainer
    print(f"[check] T={T} -> {n_tokens} tokens; params={params['total']:,}")
    train_labels = np.stack([np.asarray(dataset.data[i][1], dtype=np.float32)
                             for i in train_idx], axis=0)
    print(f"num_classes={num_classes}  train={len(train_idx)}  val={len(val_idx)}\n",
          flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)          # also holds the SQLite file, if used
    study_kwargs = dict(
        direction="maximize",
        sampler=TPESampler(seed=TUNE_SEED),
        pruner=MedianPruner(n_warmup_steps=5),
    )
    if STORAGE:
        study = optuna.create_study(study_name=tag, storage=STORAGE,
                                    load_if_exists=True, **study_kwargs)
    else:
        study = optuna.create_study(study_name=tag, **study_kwargs)

    objective = make_objective(dataset, num_classes, train_idx, val_idx, train_labels)
    study.optimize(objective, n_trials=TUNE_TRIALS, gc_after_trial=True)

    states = [t.state for t in study.trials]
    n_complete = states.count(optuna.trial.TrialState.COMPLETE)
    n_pruned   = states.count(optuna.trial.TrialState.PRUNED)
    if n_complete == 0:
        raise RuntimeError(f"no trial completed ({n_pruned} pruned); nothing to report")
    best = study.best_trial
    cell = ", ".join(f"{k}={v!r}" for k, v in best.params.items())
    paste = f'("{LABEL_SCHEME}", "{MODEL_VARIANT}"): dict({cell}),'

    print(f"\n{'='*64}")
    print(f"  BEST  {tag}")
    print(f"  macro-AUC = {best.value:.4f}  (trial {best.number}, "
          f"epoch {best.user_attrs.get('best_epoch')}/{TUNE_EPOCHS})")
    for k, v in best.params.items():
        print(f"    {k:14s}: {v}")
    print(f"  trials: {len(study.trials)} total, {n_complete} complete, {n_pruned} pruned")
    print(f"\n  Paste into TUNED in train_flat_seq.py:\n    {paste}")
    print(f"{'='*64}", flush=True)

    out = {
        "family": "flat",
        "variant": MODEL_VARIANT, "scheme": LABEL_SCHEME,
        "diagnostic_level": DIAG_LEVEL if LABEL_SCHEME == "diagnostic" else None,
        "preprocessing": PREP_TAG, "token_samples": TOKEN_LEN,
        "best_macro_auc": float(best.value),
        "best_epoch": best.user_attrs.get("best_epoch"),
        "best_params": best.params,
        "tuned_cell": paste,
        "n_trials": len(study.trials), "n_complete": n_complete, "n_pruned": n_pruned,
        "tune_epochs": TUNE_EPOCHS, "seed": TUNE_SEED,
        "arch_fixed": ARCH[MODEL_VARIANT],
    }
    out_path = os.path.join(OUT_DIR, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()