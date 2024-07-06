"""
Pretrain + fine-tune trainers for the Transfer-Learning ablation.

We use the *same* `FastPropNet` that is in
`sc3_bench/models/descriptor_models.py` so the fine-tuned trunk is
literally the same architecture as the FastProp baseline reported in
the SC3 benchmark.  The only differences vs the stock FastProp pipeline
are:

  - we expose `pretrain_combisolv()` to train the trunk on CombiSolv-QM
    ΔG_solv (or CombiSolv-Exp), and
  - we expose `finetune_on_sc3()` to train on a fraction of the SC3
    train split, optionally starting from a pretrained checkpoint, with
    three fine-tune variants (`full`, `head_only`, `last_two`).

A single shared `train_loop()` covers both pretraining and fine-tuning.
The trunk is `FastPropNet`-style (Linear → BN → ReLU → Dropout) × 3;
the final regression head is a single `Linear(hidden_dims[-1], 1)`.

When a pretrained model is loaded for fine-tuning we
(a) carry over the exact same input normalisation that was used during
    pretraining (so the trunk sees inputs in the right scale), and
(b) re-calibrate BN running statistics with one no-grad forward pass on
    the fine-tune training data before the first eval — otherwise stale
    BN running means corrupt the val-loss signal that drives early
    stopping.

The same BN re-calibration is applied to scratch runs for an
apples-to-apples comparison (otherwise scratch sees BN initialised
from pure randn forward passes, which is also stale, just for a
different reason).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
VANSH_ROOT = ABLATIONS_TRANSFER_DIR.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.models.descriptor_models import FastPropNet  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (kept consistent with the FastProp main-paper baseline)
# ---------------------------------------------------------------------------

HIDDEN_DIMS = (512, 256, 128)
DROPOUT = 0.1
WEIGHT_DECAY = 1e-5

PRETRAIN_BATCH = 1024
PRETRAIN_LR = 5e-4
PRETRAIN_EPOCHS = 60        # CombiSolv-QM is huge, ~60 epochs ≈ 1M*60 examples seen
PRETRAIN_PATIENCE = 6
PRETRAIN_LR_PATIENCE = 3
PRETRAIN_VAL_FRAC = 0.05    # 5% × 1M = 50K val samples is plenty

FINETUNE_BATCH = 256
FINETUNE_LR = 5e-4          # same LR as the FastProp baseline (`fastprop` config)
FINETUNE_EPOCHS = 300
FINETUNE_PATIENCE = 40      # match the `fastprop` HP from configs/best_hps.json
FINETUNE_LR_PATIENCE = 15

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"      [{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Reproducible row subsampling (identical contract to Q4 Data_Scaling)
# ---------------------------------------------------------------------------

def subsample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.RandomState(seed)
    k = max(32, int(round(n * fraction)))
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


# ---------------------------------------------------------------------------
# Stratified subsampling (preserves per-solvent distribution)
# ---------------------------------------------------------------------------

def stratified_subsample_indices(solvent_names: np.ndarray, fraction: float,
                                 seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(solvent_names))
    rng = np.random.RandomState(seed)
    out: list[int] = []
    unique = np.unique(solvent_names)
    for s in unique:
        mask = np.where(solvent_names == s)[0]
        k = max(1, int(round(len(mask) * fraction)))
        k = min(k, len(mask))
        out.extend(rng.choice(mask, size=k, replace=False).tolist())
    return np.array(sorted(out), dtype=np.int64)


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

# FastPropNet stacks: [Linear, BN, ReLU, Dropout] × N then a final Linear(_,1).
# So `model.net` is an nn.Sequential of length 4*N + 1.  We treat each chunk
# of 4 as one "block".  `head_only` freezes everything but the last layer;
# `last_two` keeps the last two blocks + head trainable; `full` trains
# everything.

def _block_slices(model: FastPropNet) -> list[tuple[int, int]]:
    """Return [(start, end)) slices into model.net for each (Linear, BN, ReLU,
    Dropout) block."""
    n_layers = len(model.net)
    slices = []
    for i in range(0, n_layers - 1, 4):
        slices.append((i, i + 4))
    return slices


def set_finetune_mode(model: FastPropNet, variant: str) -> int:
    """Freeze parameters according to `variant`.  Returns # trainable params.

    variant ∈ {"full", "head_only", "last_two"}
    """
    blocks = _block_slices(model)
    n_blocks = len(blocks)

    for p in model.parameters():
        p.requires_grad = True

    if variant == "full":
        pass
    elif variant == "head_only":
        for s, e in blocks:
            for layer in model.net[s:e]:
                for p in layer.parameters():
                    p.requires_grad = False
    elif variant == "last_two":
        # Freeze every block except the last two; the final Linear(head) is
        # already trainable from the loop above.
        for s, e in blocks[:max(0, n_blocks - 2)]:
            for layer in model.net[s:e]:
                for p in layer.parameters():
                    p.requires_grad = False
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_bn_running_stats(model: FastPropNet) -> None:
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()


def calibrate_bn(model: FastPropNet, X_t: torch.Tensor, batch_size: int = 1024,
                 device=None) -> None:
    """Run one no-grad forward pass in train mode to seed BN running stats."""
    model.train()
    n = len(X_t)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            _ = model(X_t[i:i + batch_size])


def replace_head(model: FastPropNet) -> None:
    """Reset just the final Linear layer (regression head) to fresh init."""
    last = model.net[-1]
    assert isinstance(last, nn.Linear) and last.out_features == 1
    new = nn.Linear(last.in_features, 1).to(last.weight.device)
    model.net[-1] = new


# ---------------------------------------------------------------------------
# Shared training loop (pretrain or fine-tune)
# ---------------------------------------------------------------------------

def train_loop(
    model: FastPropNet,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    device,
    *,
    norm_stats: Optional[tuple[np.ndarray, np.ndarray]] = None,
    lr: float = 1e-3,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = 256,
    epochs: int = 300,
    patience: int = 20,
    lr_patience: int = 10,
    desc: str = "train",
    verbose_every: int = 5,
) -> tuple[FastPropNet, tuple[np.ndarray, np.ndarray], float, dict]:
    """Train (or fine-tune) `model` with early stopping on (X_val, y_val).

    Returns (model_with_best_state, (mean, std), best_val_loss, info).
    `norm_stats=None` means: fit (mean, std) on X_train.
    """

    if norm_stats is None:
        mean = X_train.mean(0)
        std  = X_train.std(0) + 1e-8
    else:
        mean, std = norm_stats

    # Defensive: any feature column whose std is essentially zero in the
    # reference normalisation set is effectively a constant.  If we keep
    # the tiny std, then a non-zero value in `X_val` (or any held-out
    # split) standardises to 1e+8+, blowing up the BN forward pass.  We
    # rescale those columns to identity (std=1, mean=0) so the network
    # sees the raw value, which is well-behaved.
    std = std.copy()
    near_const = std < 1e-3
    if near_const.any():
        std[near_const] = 1.0
        mean = mean.copy()
        mean[near_const] = 0.0

    X_tr_n = ((X_train - mean) / std).astype(np.float32)
    X_va_n = ((X_val   - mean) / std).astype(np.float32)
    X_t = torch.tensor(X_tr_n, device=device)
    y_t = torch.tensor(y_train.astype(np.float32), device=device)
    X_v = torch.tensor(X_va_n, device=device)
    y_v = torch.tensor(y_val.astype(np.float32),   device=device)

    # Calibrate BN running stats on the new training data before any eval.
    # (Important when transferring a pretrained trunk.)
    calibrate_bn(model, X_t, batch_size=max(1024, batch_size), device=device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, factor=0.5)

    n = len(y_t)
    bs = min(batch_size, max(32, n // 4)) if n < 4096 else batch_size

    best_vl, best_state, best_ep, wait = float("inf"), None, -1, 0
    train_rmse_at_best = float("nan")
    t0 = time.time()
    pbar = tqdm(range(epochs), desc=desc, ncols=100, leave=False)
    for ep in pbar:
        model.train()
        perm = torch.randperm(n, device=device)
        loss_sum, n_batches = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            ib = perm[i:i + bs]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(X_t[ib]), y_t[ib])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()); n_batches += 1
        train_rmse = float(np.sqrt(loss_sum / max(n_batches, 1)))

        model.eval()
        with torch.no_grad():
            vp = model(X_v).cpu().numpy()
        vl = float(np.mean((vp - y_val.astype(np.float64)) ** 2))
        vrmse = float(np.sqrt(vl))
        scheduler.step(vl)

        improved = vl < best_vl
        if improved:
            best_vl, best_ep, wait = vl, ep, 0
            train_rmse_at_best = train_rmse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1

        pbar.set_postfix(train=f"{train_rmse:.4f}", val=f"{vrmse:.4f}",
                         best=f"{np.sqrt(best_vl):.4f}", wait=wait)
        if (ep + 1) % verbose_every == 0 or improved:
            _log(f"{desc} ep {ep+1:3d}  train={train_rmse:.4f}  val={vrmse:.4f}  "
                 f"best={np.sqrt(best_vl):.4f}  ({'*best' if improved else f'wait={wait}'})")
        if wait >= patience:
            _log(f"{desc} early stop at epoch {ep+1}  (best_ep={best_ep+1})")
            break
    pbar.close()
    _log(f"{desc} trained in {time.time()-t0:.1f}s  best_val_RMSE={np.sqrt(best_vl):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    info = {
        "best_val_RMSE": float(np.sqrt(best_vl)),
        "best_epoch":    int(best_ep + 1),
        "train_RMSE_at_best": float(train_rmse_at_best),
        "elapsed_s":     float(time.time() - t0),
    }
    return model, (mean, std), best_vl, info


# ---------------------------------------------------------------------------
# Pretraining on CombiSolv
# ---------------------------------------------------------------------------

def pretrain_combisolv(
    cache_path: Path,
    seed: int,
    device,
    *,
    in_dim: Optional[int] = None,
    epochs: int = PRETRAIN_EPOCHS,
    batch_size: int = PRETRAIN_BATCH,
    lr: float = PRETRAIN_LR,
    patience: int = PRETRAIN_PATIENCE,
    lr_patience: int = PRETRAIN_LR_PATIENCE,
) -> dict:
    """Pretrain a fresh FastPropNet on a CombiSolv cache.

    Returns a dict with the trained model state, normalisation stats,
    feature dim and pretraining metrics.
    """
    cached = np.load(cache_path)
    X = cached["X"]
    y = cached["y"]

    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_val = max(2048, int(round(n * PRETRAIN_VAL_FRAC)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    _log(f"pretrain: {len(X_tr):,} train / {len(X_va):,} val rows from {cache_path.name}")

    if in_dim is None:
        in_dim = X.shape[1]
    torch.manual_seed(seed); np.random.seed(seed)

    model = FastPropNet(in_dim=in_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"pretrain: model has {n_params:,} parameters  hidden={HIDDEN_DIMS}")

    model, (mu, sd), vl, info = train_loop(
        model, X_tr, y_tr, X_va, y_va, device,
        lr=lr, batch_size=batch_size,
        epochs=epochs, patience=patience, lr_patience=lr_patience,
        desc=f"pretrain[{cache_path.stem}/seed={seed}]",
    )

    return {
        "state_dict":  {k: v.cpu() for k, v in model.state_dict().items()},
        "norm_mean":   mu,
        "norm_std":    sd,
        "in_dim":      int(in_dim),
        "val_RMSE":    info["best_val_RMSE"],
        "best_epoch":  info["best_epoch"],
        "elapsed_s":   info["elapsed_s"],
        "n_train":     int(len(X_tr)),
        "n_val":       int(len(X_va)),
        "seed":        int(seed),
        "src":         str(cache_path.name),
    }


# ---------------------------------------------------------------------------
# Fine-tuning on SC3 (called once per (protocol, variant, fraction, seed))
# ---------------------------------------------------------------------------

def finetune_on_sc3(
    *,
    protocol: str,                      # "scratch" | "qm"
    variant:  str,                      # "full" | "head_only" | "last_two"
    fraction: float,
    seed: int,
    cached_sc3: dict,                   # dict with X_*/y_* (from feature_cache/rdkit.npz)
    splits: dict,                       # SC3 DataFrames (for solvent_names / uncertainties)
    device,
    pretrained: Optional[dict] = None,  # output of pretrain_combisolv (for protocol="qm")
    eval_splits: tuple[str, ...] = ("eval", "ood", "sc3_gold"),
    epochs: int = FINETUNE_EPOCHS,
    batch_size: int = FINETUNE_BATCH,
    patience: int = FINETUNE_PATIENCE,
    lr_patience: int = FINETUNE_LR_PATIENCE,
    finetune_lr: float = FINETUNE_LR,
) -> dict:
    """Fine-tune (or train from scratch) on a fraction of the SC3 train split.

    Returns a dict of per-split metrics (RMSE/MAE/R2/PS_RMSE/Z_RMSE), plus
    book-keeping fields prefixed with "_".
    """
    from sc3_bench.evaluate import compute_metrics

    X_full = cached_sc3["X_train"]
    y_full = cached_sc3["y_train"]
    X_eval = cached_sc3["X_eval"]
    y_eval = cached_sc3["y_eval"]

    # Stratified subsample by solvent (Solvent_Name in SC3 train)
    solvent_names = splits["train"]["Solvent_Name"].values
    if fraction < 1.0:
        sub_idx = stratified_subsample_indices(solvent_names, fraction, seed)
    else:
        sub_idx = np.arange(len(y_full))
    X_tr, y_tr = X_full[sub_idx], y_full[sub_idx]
    _log(f"finetune: protocol={protocol} variant={variant} fraction={fraction:.3f} "
         f"seed={seed}  n_train={len(X_tr):,}")

    torch.manual_seed(seed); np.random.seed(seed)

    in_dim = X_tr.shape[1]

    # 1) Build the model
    model = FastPropNet(in_dim=in_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)

    if protocol == "qm":
        if pretrained is None:
            raise ValueError("protocol='qm' requires pretrained=...")
        if pretrained["in_dim"] != in_dim:
            raise ValueError(
                f"in_dim mismatch: pretrained={pretrained['in_dim']} vs SC3={in_dim}"
            )
        model.load_state_dict({k: v.to(device) for k, v in pretrained["state_dict"].items()})
        replace_head(model)
        reset_bn_running_stats(model)
        norm_stats = (pretrained["norm_mean"], pretrained["norm_std"])
    else:  # scratch
        # Fit normalisation on the *full* SC3 train set, not the random
        # fraction.  At small fractions some RDKit columns can be all-zero
        # in the subsample but non-zero in eval/ood/sc3_gold, in which case
        # standardising with the subsample's near-zero std blows the eval
        # forward pass to ~1e8.  Using the full train mean/std also keeps
        # the input scale identical to the QM-pretrained branch so the
        # comparison is apples-to-apples.
        norm_stats = (X_full.mean(0), X_full.std(0) + 1e-8)

    n_trainable = set_finetune_mode(model, variant)
    n_total = sum(p.numel() for p in model.parameters())
    _log(f"finetune: {n_trainable:,} / {n_total:,} trainable params  variant={variant}")

    # 2) Train
    model, (mu, sd), vl, info = train_loop(
        model, X_tr, y_tr, X_eval, y_eval, device,
        norm_stats=norm_stats,
        lr=finetune_lr, batch_size=batch_size,
        epochs=epochs, patience=patience, lr_patience=lr_patience,
        desc=f"ft[{protocol}/{variant}/f={fraction:.2f}/s={seed}]",
    )

    # 3) Evaluate on all eval_splits with the fixed normalisation
    def predict(X_raw):
        Xn = ((X_raw - mu) / sd).astype(np.float32)
        with torch.no_grad():
            return model(torch.tensor(Xn, device=device)).cpu().numpy()

    metrics: dict[str, dict] = {}
    for sname in eval_splits:
        df = splits[sname]
        sv  = df["Solvent_Name"].values if "Solvent_Name" in df.columns else None
        unc = df["Uncertainty"].values  if "Uncertainty"  in df.columns else None
        preds = predict(cached_sc3[f"X_{sname}"])
        metrics[sname] = compute_metrics(
            cached_sc3[f"y_{sname}"], preds, sv, unc)
        _log(f"  eval[{sname:9s}]  RMSE={metrics[sname]['RMSE']:.4f}  "
             f"PS_RMSE={metrics[sname].get('PS_RMSE', float('nan')):.4f}  "
             f"N={metrics[sname]['N']}")

    metrics["_n_train"]      = int(len(X_tr))
    metrics["_n_trainable"]  = int(n_trainable)
    metrics["_best_epoch"]   = info["best_epoch"]
    metrics["_val_RMSE"]     = info["best_val_RMSE"]
    metrics["_elapsed_s"]    = info["elapsed_s"]
    metrics["_protocol"]     = protocol
    metrics["_variant"]      = variant
    metrics["_fraction"]     = float(fraction)
    metrics["_seed"]         = int(seed)
    return metrics
