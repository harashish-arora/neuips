#!/usr/bin/env python3
"""
Run descriptor-based baselines (FastProp, FastSolv, MLP) for SC3 benchmark.

These models operate on RDKit molecular descriptors + temperature features.
Uses dual-descriptor featurization: solute descriptors + solvent descriptors,
concatenated with temperature features, fed into an MLP head.

Usage:
    python scripts/run_descriptor_baselines.py                          # all models, 5 seeds
    python scripts/run_descriptor_baselines.py --model_type fastprop    # single model
    python scripts/run_descriptor_baselines.py --quick                  # 1 seed, 30 epochs
    python scripts/run_descriptor_baselines.py --gpu                    # force GPU
"""

import sys
import os
import json
import time
import argparse
import warnings
import pickle
from pathlib import Path

# Limit to 1 thread to avoid hogging the machine
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
# DataLoader/TensorDataset replaced by manual GPU-side batching (faster for GPU-resident data)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics
from src.benchmarks.featurizers import get_featurizer
from src.benchmarks.methods.descriptor_models import (
    FastPropNet,
    FastSolvNet,
    SimpleMLP,
    build_feature_cache,
    featurize_split,
    compute_sobolev_targets,
    SOBOLEV_SCALE,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_CLASS_MAP = {
    "fastprop": FastPropNet,
    "fastsolv": FastSolvNet,
    "mlp": SimpleMLP,
}


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _normalise_features(X_train, X_val=None):
    """Z-score normalise features. Returns (X_train_norm, X_val_norm, mean, std)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_tn = (X_train - mean) / std
    X_vn = None if X_val is None else (X_val - mean) / std
    return X_tn, X_vn, mean, std


def run_model(
    model_type,
    splits,
    feature_cache,
    feature_names,
    seed,
    device,
    epochs=100,
    lr=1e-3,
    batch_size=256,
    patience=20,
    hidden_dims=(512, 256, 128),
    dropout=0.1,
    method_name=None,
):
    """Train and evaluate a single descriptor-based model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method_name is None:
        method_name = model_type

    # Featurize
    X_train_raw, y_train = featurize_split(splits["train"], feature_cache, feature_names)
    X_eval_raw, y_eval = featurize_split(splits["eval"], feature_cache, feature_names)

    # ---- FastSolv: separate desc vs temp, standardise, compute Sobolev targets ----
    is_fastsolv = model_type == "fastsolv"

    if is_fastsolv:
        n_desc_cols = X_train_raw.shape[1] - 4
        desc_train = X_train_raw[:, :n_desc_cols]
        t_raw_train = X_train_raw[:, n_desc_cols].copy()  # T/300

        # Standardise descriptors
        desc_mean = desc_train.mean(axis=0)
        desc_std = desc_train.std(axis=0) + 1e-8
        desc_train_norm = ((desc_train - desc_mean) / desc_std).astype(np.float32)

        # Standardise temperature
        t_mean = float(t_raw_train.mean())
        t_std_val = float(t_raw_train.std()) + 1e-8
        t_std_arr = ((t_raw_train - t_mean) / t_std_val).astype(np.float32)

        # Standardise targets
        y_mean = float(y_train.mean())
        y_std_val = float(y_train.std()) + 1e-8
        y_train_std = ((y_train - y_mean) / y_std_val).astype(np.float32)

        # Temperature feature stats for in-network reconstruction
        tf_raw = np.column_stack([
            t_raw_train,
            (10.0 / 3.0) / np.clip(t_raw_train, 1e-6, None),
            t_raw_train ** 2,
            np.log(np.clip(t_raw_train, 1e-6, None)),
        ])
        tf_mean = tf_raw.mean(axis=0).astype(np.float32)
        tf_std = (tf_raw.std(axis=0) + 1e-8).astype(np.float32)

        # Sobolev gradient targets
        grad_targets = compute_sobolev_targets(desc_train, t_std_arr, y_train_std)
        n_valid = int(np.isfinite(grad_targets).sum())
        pct = 100.0 * n_valid / len(grad_targets)
        print(f"  Sobolev: {n_valid}/{len(grad_targets)} valid gradient targets ({pct:.1f}%)")

        # Eval tensors
        desc_eval = X_eval_raw[:, :n_desc_cols]
        t_raw_eval = X_eval_raw[:, n_desc_cols].copy()
        desc_eval_norm = ((desc_eval - desc_mean) / desc_std).astype(np.float32)
        t_std_eval = ((t_raw_eval - t_mean) / t_std_val).astype(np.float32)
        y_eval_std = ((y_eval - y_mean) / y_std_val).astype(np.float32)

        # Determine input dim
        in_dim = n_desc_cols + 4

        # Store stats for later prediction
        _stats = {
            "desc_mean": desc_mean, "desc_std": desc_std,
            "t_mean": t_mean, "t_std": t_std_val,
            "y_mean": y_mean, "y_std": y_std_val,
            "tf_mean": tf_mean, "tf_std": tf_std,
            "n_desc_cols": n_desc_cols,
        }
    else:
        # Standard normalisation for FastProp / MLP
        X_train_norm, X_eval_norm, feat_mean, feat_std = _normalise_features(
            X_train_raw, X_eval_raw
        )
        in_dim = X_train_norm.shape[1]
        _stats = {"feat_mean": feat_mean, "feat_std": feat_std}

    # Build model
    ModelClass = MODEL_CLASS_MAP[model_type]
    model = ModelClass(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Setup output directories
    seed_dir = RESULTS_DIR / method_name / f"seed_{seed}"
    checkpoints_dir = seed_dir / "checkpoints"
    predictions_dir = seed_dir / "predictions"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "method_name": method_name,
        "model_type": model_type,
        "seed": seed,
        "device": str(device),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "patience": patience,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "in_dim": in_dim,
        "n_train": len(y_train),
        "n_eval": len(y_eval),
    }
    save_json(seed_dir / "run_config.json", metadata)

    # Prepare DataLoader
    if is_fastsolv:
        desc_t = torch.tensor(desc_train_norm, dtype=torch.float32, device=device)
        t_t = torch.tensor(t_std_arr, dtype=torch.float32, device=device).unsqueeze(1)
        y_t = torch.tensor(y_train_std, dtype=torch.float32, device=device)
        g_t = torch.tensor(grad_targets, dtype=torch.float32, device=device)

        # Stat tensors on device
        tm = torch.tensor([t_mean], dtype=torch.float32, device=device)
        ts = torch.tensor([t_std_val], dtype=torch.float32, device=device)
        tfm = torch.tensor(tf_mean, dtype=torch.float32, device=device)
        tfs = torch.tensor(tf_std, dtype=torch.float32, device=device)

        def _make_features(desc, t_standardised):
            """Reconstruct 4 temp features from standardised temperature for autograd."""
            t_raw = t_standardised * ts + tm
            t_inv = (10.0 / 3.0) / t_raw.clamp(min=1e-6)
            t_sq = t_raw ** 2
            t_log = torch.log(t_raw.clamp(min=1e-6))
            tf = torch.cat([t_raw, t_inv, t_sq, t_log], dim=-1)
            tf_norm = (tf - tfm) / tfs
            return torch.cat([desc, tf_norm], dim=-1)

        n_train = desc_t.size(0)

        # Eval tensors
        desc_v = torch.tensor(desc_eval_norm, dtype=torch.float32, device=device)
        t_v = torch.tensor(t_std_eval, dtype=torch.float32, device=device).unsqueeze(1)
        y_v = torch.tensor(y_eval_std, dtype=torch.float32, device=device)
    else:
        X_t = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_v = torch.tensor(X_eval_norm, dtype=torch.float32, device=device)
        y_v_tensor = torch.tensor(y_eval, dtype=torch.float32, device=device)
        n_train = X_t.size(0)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    t0 = time.time()

    for epoch in range(epochs):
        model.train()

        # GPU-side shuffle (no CPU↔GPU sync)
        perm = torch.randperm(n_train, device=device)

        if is_fastsolv:
            epoch_mse = 0.0
            epoch_sob = 0.0
            n_batches = 0
            for i in range(0, n_train - batch_size + 1, batch_size):
                idx = perm[i:i + batch_size]
                db = desc_t[idx]
                tb = t_t[idx].detach().requires_grad_(True)
                yb = y_t[idx]
                gb = g_t[idx]

                optimizer.zero_grad()
                x = _make_features(db, tb)
                y_hat = model(x)

                mse = nn.functional.mse_loss(y_hat, yb)

                # Sobolev: d(y_hat)/d(t_std) via autograd with create_graph
                dy_dt = torch.autograd.grad(
                    y_hat, tb,
                    grad_outputs=torch.ones_like(y_hat),
                    create_graph=True,
                    retain_graph=True,
                )[0].squeeze(-1)

                valid_mask = ~torch.isnan(gb)
                if valid_mask.any():
                    diff_sq = (dy_dt[valid_mask] - gb[valid_mask]) ** 2
                    sobolev = SOBOLEV_SCALE * diff_sq.mean()
                else:
                    sobolev = torch.zeros(1, device=device, requires_grad=False)[0]

                (mse + sobolev).backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_mse += mse.item()
                epoch_sob += sobolev.item()
                n_batches += 1

            train_loss = epoch_mse / max(n_batches, 1)
        else:
            total_loss = 0.0
            n_seen = 0
            for i in range(0, n_train - batch_size + 1, batch_size):
                idx = perm[i:i + batch_size]
                xb = X_t[idx]
                yb = y_t[idx]

                optimizer.zero_grad()
                preds = model(xb)
                loss = nn.functional.mse_loss(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size
                n_seen += batch_size
            train_loss = total_loss / max(n_seen, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            if is_fastsolv:
                val_pred = model(_make_features(desc_v, t_v))
                val_loss = nn.functional.mse_loss(val_pred, y_v).item()
                val_preds_np = val_pred.cpu().numpy() * y_std_val + y_mean
                val_targets_np = y_eval
            else:
                val_pred = model(X_v)
                val_loss = nn.functional.mse_loss(val_pred, y_v_tensor).item()
                val_preds_np = val_pred.cpu().numpy()
                val_targets_np = y_eval

        val_rmse = float(np.sqrt(np.mean((val_preds_np - val_targets_np) ** 2)))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse": val_rmse,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "is_best": wait == 0,
            "wait": int(wait),
            "elapsed_s": float(time.time() - t0),
        }
        history.append(epoch_record)
        save_json(seed_dir / "epoch_history.json", history)

        ckpt = {
            "epoch": epoch + 1,
            "seed": seed,
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_val_loss_so_far": float(best_val_loss),
            "wait": int(wait),
            "config": metadata,
        }
        save_pickle(checkpoints_dir / f"epoch_{epoch+1:03d}.pkl", ckpt)

        np.savez(
            predictions_dir / f"epoch_{epoch+1:03d}_eval_preds.npz",
            y_pred=val_preds_np,
            y_true=val_targets_np,
        )

        if (epoch + 1) % 10 == 0 or wait == 0:
            extra = ""
            if is_fastsolv:
                extra = f"  sob={epoch_sob / max(n_batches, 1):.4f}"
            print(
                f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  val_rmse={val_rmse:.4f}"
                + extra
                + (f"  *best" if wait == 0 else f"  (wait={wait})")
            )

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    save_pickle(seed_dir / "best_model.pkl", {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "seed": seed,
        "model_type": model_type,
        "config": metadata,
        "model_state_dict": best_state,
    })

    # Evaluate on all splits
    results = {
        "train_time_s": train_time,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }

    for split_name in ["eval", "test_hard", "test_medium", "test_easy"]:
        df = splits[split_name]
        X_split_raw, y_split = featurize_split(df, feature_cache, feature_names)

        with torch.no_grad():
            if is_fastsolv:
                n_dc = _stats["n_desc_cols"]
                desc_s = ((X_split_raw[:, :n_dc] - _stats["desc_mean"]) / _stats["desc_std"]).astype(np.float32)
                t_raw_s = X_split_raw[:, n_dc].copy()
                t_std_s = ((t_raw_s - _stats["t_mean"]) / _stats["t_std"]).astype(np.float32)

                desc_st = torch.tensor(desc_s, dtype=torch.float32, device=device)
                t_st = torch.tensor(t_std_s, dtype=torch.float32, device=device).unsqueeze(1)
                preds_std = model(_make_features(desc_st, t_st)).cpu().numpy()
                preds = preds_std * _stats["y_std"] + _stats["y_mean"]
            else:
                X_split_norm = (X_split_raw - _stats["feat_mean"]) / _stats["feat_std"]
                X_st = torch.tensor(X_split_norm, dtype=torch.float32, device=device)
                preds = model(X_st).cpu().numpy()

        targets = y_split

        solvent_col = "Solvent_Name" if "Solvent_Name" in df.columns else None
        unc_col = "Uncertainty" if "Uncertainty" in df.columns else None

        metrics = compute_metrics(
            y_true=targets,
            y_pred=preds,
            solvent_names=df[solvent_col].values if solvent_col else None,
            uncertainties=df[unc_col].values if unc_col else None,
        )
        results[split_name] = metrics

        np.savez(
            predictions_dir / f"final_{split_name}_preds.npz",
            y_pred=preds,
            y_true=targets,
        )
        df_out = df.copy()
        df_out["pred_LogS"] = preds
        df_out.to_csv(predictions_dir / f"final_{split_name}_predictions.csv", index=False)

        print(
            f"  {split_name:15s}: RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}"
            + (f"  PS-RMSE={metrics.get('PS_RMSE', float('nan')):.4f}" if "PS_RMSE" in metrics else "")
        )

    results["artifacts"] = {
        "seed_dir": str(seed_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "predictions_dir": str(predictions_dir),
        "epoch_history_json": str(seed_dir / "epoch_history.json"),
        "best_model_pkl": str(seed_dir / "best_model.pkl"),
        "run_config_json": str(seed_dir / "run_config.json"),
    }
    save_json(seed_dir / "seed_summary.json", results)
    return results


def aggregate_results(all_results, method_name):
    """Aggregate results across seeds into mean +/- std."""
    seeds = list(all_results.keys())
    split_names = ["eval", "test_hard", "test_medium", "test_easy"]
    splits_present = [k for k in split_names if k in all_results[seeds[0]]]

    summary = {"method": method_name, "n_seeds": len(seeds)}

    for split in splits_present:
        for metric in ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE"]:
            values = []
            for seed in seeds:
                v = all_results[seed].get(split, {}).get(metric)
                if v is not None and not np.isnan(v):
                    values.append(v)
            if values:
                summary[f"{split}_{metric}_mean"] = float(np.mean(values))
                summary[f"{split}_{metric}_std"] = float(np.std(values))

    train_times = [all_results[s]["train_time_s"] for s in seeds]
    summary["train_time_mean_s"] = float(np.mean(train_times))

    return summary


def save_results(method_name, raw_results, summary):
    """Save results to disk."""
    out_dir = RESULTS_DIR / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({str(k): v for k, v in raw_results.items()}, f, indent=2, default=str)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run descriptor-based baselines for SC3")
    parser.add_argument(
        "--model_type", nargs="+", default=["fastprop", "fastsolv", "mlp"],
        choices=["fastprop", "fastsolv", "mlp"],
        help="Model architecture(s) to run",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 seed, 30 epochs")
    parser.add_argument("--gpu", action="store_true", help="Force GPU (fail if unavailable)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 30 if args.quick else args.epochs
    hidden_dims = tuple(args.hidden_dims)

    # Device
    if args.gpu:
        assert torch.cuda.is_available(), "GPU requested but CUDA not available"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("SC3 Descriptor-Based Baseline Runner")
    print(f"  Model types: {args.model_type}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Hidden: {hidden_dims}, Dropout: {args.dropout}")
    print(f"  LR: {args.lr}, Batch size: {args.batch_size}, Patience: {args.patience}")
    print()

    # Load data
    splits = load_all_splits()

    # Build feature cache (shared across all models and seeds)
    featurizer = get_featurizer("rdkit")
    feature_cache, feature_names = build_feature_cache(splits, featurizer)

    for model_type in args.model_type:
        method_name = model_type
        print(f"\n{'='*70}")
        print(f"MODEL TYPE: {model_type.upper()}")
        print(f"{'='*70}")

        all_results = {}
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            results = run_model(
                model_type=model_type,
                splits=splits,
                feature_cache=feature_cache,
                feature_names=feature_names,
                seed=seed,
                device=device,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
                hidden_dims=hidden_dims,
                dropout=args.dropout,
                method_name=method_name,
            )
            all_results[seed] = results

        summary = aggregate_results(all_results, method_name)
        save_results(method_name, all_results, summary)
        _out = RESULTS_DIR / method_name
        print(f"  Saved raw_results.json → {_out / 'raw_results.json'}")
        print(f"  Saved summary.json      → {_out / 'summary.json'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
