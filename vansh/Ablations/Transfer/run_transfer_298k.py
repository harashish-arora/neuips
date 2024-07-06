"""
Q3 Transfer-Learning ablation, **298 K-locked variant**.

This is the temperature-isolated version of `run_transfer.py`: both the
pretraining task (CombiSolv-QM ΔG_solv at 298 K) and the fine-tuning task
(SC3 logS, restricted to / interpolated to 298.15 K) live on the same
temperature axis.  Hence any transfer benefit must come purely from the
chemistry signal and not from the model "filling in" temperature-related
inductive bias.

Two SC3 subsets are supported (selected via `--approach`):
  * filter   — real measurements with 295.15 ≤ T ≤ 301.15 K
  * interp   — every (solute, solvent) pair evaluated at exactly
               298.15 K via the Apelblat / Van't Hoff fit

Build the caches once with:
    python build_298k_data.py

Then run:
    python run_transfer_298k.py --approach filter --gpu 0
    python run_transfer_298k.py --approach interp --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
VANSH_ROOT = ABLATIONS_TRANSFER_DIR.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.evaluate import compute_metrics  # noqa: E402

from transfer_trainers import (  # noqa: E402
    set_finetune_mode, replace_head, reset_bn_running_stats, train_loop,
    HIDDEN_DIMS, DROPOUT,
    FINETUNE_BATCH, FINETUNE_LR, FINETUNE_EPOCHS, FINETUNE_PATIENCE, FINETUNE_LR_PATIENCE,
)
from sc3_bench.models.descriptor_models import FastPropNet  # noqa: E402

DEFAULT_PROTOCOLS = ["scratch", "qm"]
DEFAULT_VARIANTS  = ["full", "head_only"]
DEFAULT_FRACTIONS = [0.05, 0.25, 1.0]
DEFAULT_SEEDS     = [42, 101, 123, 456, 789]
EVAL_SPLITS = ("eval", "ood", "sc3_gold")

CACHE_DIR  = ABLATIONS_TRANSFER_DIR / "cache"
MODEL_DIR  = ABLATIONS_TRANSFER_DIR / "models"
LOG_DIR    = ABLATIONS_TRANSFER_DIR / "logs"
for d in (CACHE_DIR, MODEL_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _load_298k(approach: str) -> dict:
    p = CACHE_DIR / f"298k_{approach}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.  Run `python build_298k_data.py` first.")
    arrs = np.load(p, allow_pickle=True)
    return {k: arrs[k] for k in arrs.files}


def _stratified_subsample_indices(solvent_names: np.ndarray, fraction: float,
                                  seed: int) -> np.ndarray:
    if fraction >= 1.0:
        return np.arange(len(solvent_names))
    rng = np.random.RandomState(seed)
    out: list[int] = []
    for s in np.unique(solvent_names):
        mask = np.where(solvent_names == s)[0]
        k = max(1, int(round(len(mask) * fraction)))
        k = min(k, len(mask))
        out.extend(rng.choice(mask, size=k, replace=False).tolist())
    return np.array(sorted(out), dtype=np.int64)


def _load_pretrained(ckpt: Path, device) -> dict:
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    obj["state_dict"] = {k: v.to(device) for k, v in obj["state_dict"].items()}
    return obj


def _result_path(approach: str, protocol: str, variant: str,
                 fraction: float, seed: int) -> Path:
    return (ABLATIONS_TRANSFER_DIR / "results_298k" / approach /
            f"{protocol}__{variant}" / f"frac_{fraction:g}" /
            f"seed_{seed}.json")


def _evaluate(model, mu, sd, cached: dict, device,
              eval_splits=EVAL_SPLITS) -> dict:
    out: dict[str, dict] = {}
    for sname in eval_splits:
        X = cached[f"X_{sname}"]
        y = cached[f"y_{sname}"]
        sv = cached.get(f"solv_{sname}")
        unc = cached.get(f"unc_{sname}")
        Xn = ((X - mu) / sd).astype(np.float32)
        with torch.no_grad():
            preds = model(torch.tensor(Xn, device=device)).cpu().numpy()
        out[sname] = compute_metrics(y, preds, sv, unc)
        _log(f"  eval[{sname:9s}]  RMSE={out[sname]['RMSE']:.4f}  "
             f"PS_RMSE={out[sname].get('PS_RMSE', float('nan')):.4f}  "
             f"N={out[sname]['N']}")
    return out


def _finetune_one(*, approach: str, protocol: str, variant: str,
                  fraction: float, seed: int, cached: dict, device,
                  pretrained: Optional[dict]) -> dict:
    X_full = cached["X_train"]
    y_full = cached["y_train"]
    X_eval = cached["X_eval"]
    y_eval = cached["y_eval"]
    solv_train = cached["solv_train"]

    if fraction < 1.0:
        sub = _stratified_subsample_indices(solv_train, fraction, seed)
    else:
        sub = np.arange(len(y_full))
    X_tr, y_tr = X_full[sub], y_full[sub]
    _log(f"finetune[{approach}]: protocol={protocol} variant={variant} "
         f"fraction={fraction:.3f} seed={seed}  n_train={len(X_tr):,}")

    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = X_tr.shape[1]
    model = FastPropNet(in_dim=in_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(device)

    if protocol == "qm":
        if pretrained is None:
            raise ValueError("protocol='qm' requires pretrained=...")
        if pretrained["in_dim"] != in_dim:
            raise ValueError(
                f"in_dim mismatch: pretrained={pretrained['in_dim']} vs SC3={in_dim}")
        model.load_state_dict({k: v.to(device) for k, v in pretrained["state_dict"].items()})
        replace_head(model)
        reset_bn_running_stats(model)
        norm_stats = (pretrained["norm_mean"], pretrained["norm_std"])
    else:
        # Important: use the *full* SC3 train set's statistics (not the
        # subsample's).  At fraction=0.05 the subsample has ~70 RDKit
        # columns that happen to be constant, making the per-column std
        # ~1e-8.  Standardising the *eval* split with that std then
        # produces values around 1e+8, which blows up the first BN forward
        # pass and traps best_epoch=1 with a nonsense initial loss.  Using
        # the full-train statistics also keeps `scratch` and `qm` on the
        # same input scale so the comparison is honest.
        norm_stats = (X_full.mean(0), X_full.std(0) + 1e-8)

    n_trainable = set_finetune_mode(model, variant)

    model, (mu, sd), vl, info = train_loop(
        model, X_tr, y_tr, X_eval, y_eval, device,
        norm_stats=norm_stats,
        lr=FINETUNE_LR, batch_size=FINETUNE_BATCH,
        epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
        lr_patience=FINETUNE_LR_PATIENCE,
        desc=f"ft298k[{approach}/{protocol}/{variant}/f={fraction:.2f}/s={seed}]",
    )
    metrics = _evaluate(model, mu, sd, cached, device)
    metrics["_n_train"] = int(len(X_tr))
    metrics["_n_trainable"] = int(n_trainable)
    metrics["_best_epoch"] = info["best_epoch"]
    metrics["_val_RMSE"] = info["best_val_RMSE"]
    metrics["_elapsed_s"] = info["elapsed_s"]
    metrics["_protocol"] = protocol
    metrics["_variant"] = variant
    metrics["_fraction"] = float(fraction)
    metrics["_seed"] = int(seed)
    metrics["_approach"] = approach
    return metrics


def _flatten_metrics(metrics: dict) -> dict:
    row: dict[str, object] = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            row[k.lstrip("_")] = v
        elif isinstance(v, dict):
            for mk, mv in v.items():
                row[f"{k}__{mk}"] = mv
    return row


def _rebuild_table(approach: str) -> Path:
    base = ABLATIONS_TRANSFER_DIR / "results_298k" / approach
    rows = []
    for jp in sorted(base.rglob("seed_*.json")):
        with open(jp) as f:
            rows.append(_flatten_metrics(json.load(f)))
    out = base / "transfer_table.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out, index=False)
    return out


def run(*, approach: str, protocols=None, variants=None,
        fractions=None, seeds=None, force: bool = False,
        gpu: Optional[int] = None) -> None:
    protocols = protocols or DEFAULT_PROTOCOLS
    variants = variants or DEFAULT_VARIANTS
    fractions = fractions or DEFAULT_FRACTIONS
    seeds = seeds or DEFAULT_SEEDS

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device={device}  approach={approach}  CUDA={torch.cuda.is_available()}  "
         f"({torch.cuda.device_count()} devices visible)")

    cached = _load_298k(approach)
    in_dim = cached["X_train"].shape[1]
    _log(f"298k/{approach}: train={len(cached['y_train']):,}  "
         f"eval={len(cached['y_eval']):,}  ood={len(cached['y_ood']):,}  "
         f"sc3_gold={len(cached['y_sc3_gold']):,}")

    # Pretrained checkpoints from the main experiment (per-seed).
    pretrained_by_seed: dict[int, dict] = {}
    if "qm" in protocols:
        for seed in seeds:
            ckpt = MODEL_DIR / f"pretrained_qm_seed{seed}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Missing pretrained QM checkpoint for seed {seed}.  "
                    f"Run `python run_transfer.py --pretrain-only --seeds {seed}` first.")
            pretrained_by_seed[seed] = _load_pretrained(ckpt, device)
            if pretrained_by_seed[seed]["in_dim"] != in_dim:
                raise ValueError(
                    f"in_dim mismatch: pretrained={pretrained_by_seed[seed]['in_dim']} "
                    f"vs 298k cache={in_dim}")

    total = len(protocols) * len(variants) * len(fractions) * len(seeds)
    run_idx = 0
    new_runs = 0

    for protocol in protocols:
        for variant in variants:
            for fraction in fractions:
                for seed in seeds:
                    run_idx += 1
                    out_json = _result_path(approach, protocol, variant, fraction, seed)
                    if out_json.exists() and not force:
                        _log(f"[{run_idx}/{total}] SKIP {out_json.relative_to(ABLATIONS_TRANSFER_DIR)}")
                        continue
                    _log(f"[{run_idx}/{total}] approach={approach} proto={protocol} "
                         f"variant={variant} fraction={fraction} seed={seed}")
                    pre = pretrained_by_seed.get(seed) if protocol == "qm" else None
                    metrics = _finetune_one(
                        approach=approach, protocol=protocol, variant=variant,
                        fraction=fraction, seed=seed, cached=cached,
                        device=device, pretrained=pre,
                    )
                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_json, "w") as f:
                        json.dump(metrics, f, indent=2, default=str)
                    new_runs += 1
                    if new_runs % 5 == 0:
                        _rebuild_table(approach)

    out_csv = _rebuild_table(approach)
    _log("=" * 70)
    _log(f"Done.  {new_runs} new run(s); table at {out_csv}")
    _log("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--approach", choices=["filter", "interp"], required=True)
    p.add_argument("--protocols", nargs="+", default=None,
                   choices=["scratch", "qm"])
    p.add_argument("--variants", nargs="+", default=None,
                   choices=["full", "head_only", "last_two"])
    p.add_argument("--fractions", nargs="+", type=float, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(
        approach=args.approach,
        protocols=args.protocols, variants=args.variants,
        fractions=args.fractions, seeds=args.seeds,
        gpu=args.gpu, force=args.force,
    )
