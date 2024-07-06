#!/usr/bin/env python3
"""
Run GNN baselines (GCN, GAT, GIN) for SC3 benchmark.

These models operate on molecular graphs directly (no pre-computed features).
Uses the dual-encoder architecture: separate GNNs for solute and solvent,
concatenated with temperature features, fed into an MLP head.

Usage:
    python scripts/run_gnn_baselines.py                     # all GNN types, 5 seeds
    python scripts/run_gnn_baselines.py --gnn_type GCN      # single GNN type
    python scripts/run_gnn_baselines.py --quick              # 1 seed, 30 epochs
    python scripts/run_gnn_baselines.py --gpu                # force GPU
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
from torch.utils.data import DataLoader

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
from src.benchmarks.methods.gnn_models import (
    smiles_to_graph, DualGNNSolubility, SolubilityGraphDataset, collate_graphs
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_graph_cache(splits):
    """Pre-compute molecular graphs for all unique SMILES across all splits."""
    all_smiles = set()
    for name, df in splits.items():
        all_smiles.update(df["Solute"].unique())
        all_smiles.update(df["Solvent"].unique())

    print(f"Building graph cache for {len(all_smiles)} unique molecules...")
    cache = {}
    failed = 0
    for smi in all_smiles:
        g = smiles_to_graph(smi)
        if g is not None:
            cache[smi] = g
        else:
            failed += 1
            # Fallback: single-node graph for unparseable SMILES
            cache[smi] = {
                "node_feats": torch.zeros((1, 7), dtype=torch.float32),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "num_nodes": 1,
            }
    print(f"  Cached {len(cache)} graphs ({failed} fallback)")
    return cache


def move_batch_to_device(batch_data, device):
    """Move a batched graph dict to device."""
    return {
        "node_feats": batch_data["node_feats"].to(device),
        "edge_index": batch_data["edge_index"].to(device),
        "batch_ids": batch_data["batch_ids"].to(device),
        "num_nodes": batch_data["num_nodes"],  # list, stays on CPU
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for solute_data, solvent_data, temp_feats, targets in loader:
        solute_data = move_batch_to_device(solute_data, device)
        solvent_data = move_batch_to_device(solvent_data, device)
        temp_feats = temp_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(solute_data, solvent_data, temp_feats)
        loss = nn.functional.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    for solute_data, solvent_data, temp_feats, targets in loader:
        solute_data = move_batch_to_device(solute_data, device)
        solvent_data = move_batch_to_device(solvent_data, device)
        temp_feats = temp_feats.to(device)

        preds = model(solute_data, solvent_data, temp_feats)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def save_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def run_gnn(gnn_type, splits, graph_cache, seed, device, epochs=100, lr=1e-3,
            batch_size=64, patience=15, hidden_dim=64, num_layers=3, method_name=None):
    """Train and evaluate a single GNN model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if method_name is None:
        method_name = f"gnn_{gnn_type.lower()}"

    # Build datasets
    train_ds = SolubilityGraphDataset(splits["train"], graph_cache)
    eval_ds = SolubilityGraphDataset(splits["eval"], graph_cache)

    use_cuda = device.type == "cuda"
    dl_kwargs = dict(collate_fn=collate_graphs,
                     num_workers=4, pin_memory=use_cuda,
                     persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)

    # Build model
    model = DualGNNSolubility(
        node_dim=7, hidden_dim=hidden_dim, num_layers=num_layers,
        gnn_type=gnn_type, num_temp_feats=4
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    seed_dir = RESULTS_DIR / method_name / f"seed_{seed}"
    checkpoints_dir = seed_dir / "checkpoints"
    predictions_dir = seed_dir / "predictions"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method_name": method_name,
        "gnn_type": gnn_type,
        "seed": seed,
        "device": str(device),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "patience": patience,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
    }
    save_json(seed_dir / "run_config.json", metadata)

    t0 = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validation
        val_preds, val_targets = evaluate(model, eval_loader, device)
        val_loss = float(np.mean((val_preds - val_targets) ** 2))

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
            "val_rmse": float(np.sqrt(val_loss)),
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
            "gnn_type": gnn_type,
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
            y_pred=val_preds,
            y_true=val_targets,
        )

        if (epoch + 1) % 10 == 0 or wait == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  val_rmse={np.sqrt(val_loss):.4f}"
                  + (f"  *best" if wait == 0 else f"  (wait={wait})"))

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    save_pickle(seed_dir / "best_model.pkl", {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "seed": seed,
        "gnn_type": gnn_type,
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
        ds = SolubilityGraphDataset(df, graph_cache)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_graphs,
                            num_workers=4, pin_memory=use_cuda,
                            persistent_workers=True)

        preds, targets = evaluate(model, loader, device)

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
        print(f"  {split_name:15s}: RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}"
              + (f"  PS-RMSE={metrics.get('PS_RMSE', float('nan')):.4f}" if "PS_RMSE" in metrics else ""))

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
    splits = [k for k in split_names if k in all_results[seeds[0]]]

    summary = {"method": method_name, "n_seeds": len(seeds)}

    for split in splits:
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
    parser = argparse.ArgumentParser(description="Run GNN baselines for SC3")
    parser.add_argument("--gnn_type", nargs="+", default=["GCN", "GAT", "GIN"],
                        choices=["GCN", "GAT", "GIN"],
                        help="GNN architecture(s) to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 seed, 30 epochs")
    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU (fail if unavailable)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 30 if args.quick else args.epochs

    # Device
    if args.gpu:
        assert torch.cuda.is_available(), "GPU requested but CUDA not available"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("SC3 GNN Baseline Runner")
    print(f"  GNN types: {args.gnn_type}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Hidden: {args.hidden_dim}, Layers: {args.num_layers}")
    print()

    # Load data
    splits = load_all_splits()

    # Build graph cache (shared across all GNN types and seeds)
    graph_cache = build_graph_cache(splits)

    for gnn_type in args.gnn_type:
        method_name = f"gnn_{gnn_type.lower()}"
        print(f"\n{'='*70}")
        print(f"GNN TYPE: {gnn_type}")
        print(f"{'='*70}")

        all_results = {}
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            results = run_gnn(
                gnn_type=gnn_type,
                splits=splits,
                graph_cache=graph_cache,
                seed=seed,
                device=device,
                epochs=epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
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
