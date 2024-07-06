#!/usr/bin/env python3
"""
Run MolMerger baseline for SC3 benchmark.

MolMerger: Ramani & Karmakar (arXiv 2402.11340, JCTC 2024)
Merges solute + solvent into a single graph via Gasteiger charge-based
virtual bonds, then runs AttentiveFP (PyG) for LogS regression.

Usage:
    python scripts/run_molmerger.py --gpu 2              # single GPU
    python scripts/run_molmerger.py --quick --gpu 3      # smoke test
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path

# ── Must set CUDA_VISIBLE_DEVICES *before* importing torch ──
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", nargs="+", type=int, default=None)
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _pre_args.gpu)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch

torch.set_num_threads(4)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics
from src.benchmarks.methods.molmerger import (
    molmerger_skeleton, stamp_temperature, MolMergerNet, NODE_DIM, EDGE_DIM,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ============================================================================
# Graph cache — keyed by (solute, solvent) pair, not triple
# ============================================================================

def build_skeleton_cache(splits):
    """Pre-compute merged molecular graph skeletons (no temperature)."""
    pairs = set()
    for df in splits.values():
        for sol, solv in zip(df["Solute"].values, df["Solvent"].values):
            pairs.add((sol, solv))

    print(f"Building skeleton cache for {len(pairs)} unique (solute, solvent) pairs...")
    cache = {}
    failed = 0
    for sol, solv in pairs:
        g = molmerger_skeleton(sol, solv)
        if g is not None:
            cache[(sol, solv)] = g
        else:
            failed += 1
    print(f"  Cached {len(cache)} skeletons ({failed} failed)")
    return cache


# ============================================================================
# Dataset — stamps temperature at access time (zero-copy on skeleton tensors)
# ============================================================================

class MolMergerDataset(torch.utils.data.Dataset):
    def __init__(self, df, skeleton_cache):
        solutes = df["Solute"].values
        solvents = df["Solvent"].values
        temps = df["Temperature"].values.astype(np.float64)
        targets = df["LogS"].values.astype(np.float64)

        self.skeletons = []
        self.temps = []
        self.targets = []
        skipped = 0
        for i in range(len(df)):
            key = (solutes[i], solvents[i])
            skel = skeleton_cache.get(key)
            if skel is not None:
                self.skeletons.append(skel)
                self.temps.append(float(temps[i]))
                self.targets.append(float(targets[i]))
            else:
                skipped += 1
        if skipped:
            print(f"    Skipped {skipped}/{len(df)} samples (unparseable molecules)")

    def __len__(self):
        return len(self.skeletons)

    def __getitem__(self, idx):
        g = stamp_temperature(self.skeletons[idx], self.temps[idx])
        return g, self.targets[idx]


def collate_molmerger(batch):
    graphs, targets = zip(*batch)
    return Batch.from_data_list(list(graphs)), torch.tensor(targets, dtype=torch.float32)


# ============================================================================
# Training loop
# ============================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for batch_data, targets in loader:
        batch_data = batch_data.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        preds = model(batch_data)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(targets)
        n += len(targets)
    return total_loss / n


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for batch_data, targets in loader:
        batch_data = batch_data.to(device, non_blocking=True)
        preds = model(batch_data)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def run_molmerger(splits, skeleton_cache, seed, device, epochs=200, lr=1e-3,
                  batch_size=128, patience=25, hidden_dim=200, num_layers=3,
                  num_timesteps=2, dropout=0.2):
    """Train and evaluate a single MolMerger model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    train_ds = MolMergerDataset(splits["train"], skeleton_cache)
    eval_ds = MolMergerDataset(splits["eval"], skeleton_cache)

    # num_workers=0: data is already in RAM as tensors; serializing PyG Data
    # objects through IPC is slower than just stamping temperature in-process.
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_molmerger, num_workers=0, pin_memory=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_molmerger, num_workers=0, pin_memory=False,
    )

    model = MolMergerNet(
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {param_count:,}")
    print(f"  Train samples: {len(train_ds):,}  |  Eval samples: {len(eval_ds):,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    t0 = time.time()

    for epoch in range(epochs):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_preds, val_targets = evaluate_model(model, eval_loader, device)
        val_loss = float(np.mean((val_preds - val_targets) ** 2))
        ep_time = time.time() - ep_start

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 5 == 0 or wait == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d} [{ep_time:.1f}s]: "
                f"train_loss={train_loss:.4f}  val_rmse={np.sqrt(val_loss):.4f}  "
                f"lr={lr_now:.2e}"
                + ("  *best" if wait == 0 else f"  (wait={wait})")
            )

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0
    print(f"  Training took {train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    results = {"train_time_s": train_time}

    for split_name in ["eval", "test_hard", "test_medium", "test_easy"]:
        df = splits[split_name]
        ds = MolMergerDataset(df, skeleton_cache)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_molmerger, num_workers=0,
        )

        preds, targets = evaluate_model(model, loader, device)

        solvent_col = "Solvent_Name" if "Solvent_Name" in df.columns else None
        unc_col = "Uncertainty" if "Uncertainty" in df.columns else None

        metrics = compute_metrics(
            y_true=targets,
            y_pred=preds,
            solvent_names=df[solvent_col].values if solvent_col else None,
            uncertainties=df[unc_col].values if unc_col else None,
        )
        results[split_name] = metrics
        print(
            f"  {split_name:15s}: RMSE={metrics['RMSE']:.4f}  "
            f"MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}"
            + (f"  PS-RMSE={metrics['PS_RMSE']:.4f}" if "PS_RMSE" in metrics else "")
        )

    return results


# ============================================================================
# Aggregation & persistence (same pattern as run_gnn_baselines.py)
# ============================================================================

def aggregate_results(all_results, method_name):
    seeds = list(all_results.keys())
    splits = [k for k in all_results[seeds[0]].keys() if k != "train_time_s"]
    summary = {"method": method_name, "n_seeds": len(seeds)}
    for split in splits:
        for metric in ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE"]:
            vals = [all_results[s].get(split, {}).get(metric) for s in seeds]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if vals:
                summary[f"{split}_{metric}_mean"] = float(np.mean(vals))
                summary[f"{split}_{metric}_std"] = float(np.std(vals))
    summary["train_time_mean_s"] = float(np.mean([all_results[s]["train_time_s"] for s in seeds]))
    return summary


def save_results(method_name, raw_results, summary):
    out_dir = RESULTS_DIR / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({str(k): v for k, v in raw_results.items()}, f, indent=2, default=str)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run MolMerger for SC3 benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="1 seed, 30 epochs")
    parser.add_argument("--gpu", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_timesteps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 30 if args.quick else args.epochs

    if args.gpu is not None:
        device = torch.device("cuda:0")
        actual = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"Using GPU(s): {args.gpu}  (CUDA_VISIBLE_DEVICES={actual})")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("SC3 MolMerger Baseline Runner")
    print("=" * 70)
    print(f"  Seeds:      {seeds}")
    print(f"  Device:     {device}  ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'})")
    print(f"  Epochs:     {epochs}")
    print(f"  Hidden:     {args.hidden_dim}  Layers: {args.num_layers}  "
          f"Timesteps: {args.num_timesteps}  Dropout: {args.dropout}")
    print(f"  Batch size: {args.batch_size}  LR: {args.lr}  Patience: {args.patience}")
    print()

    splits = load_all_splits()
    skeleton_cache = build_skeleton_cache(splits)

    method_name = "molmerger"
    all_results = {}
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed {seed}")
        print(f"{'='*70}")
        results = run_molmerger(
            splits=splits,
            skeleton_cache=skeleton_cache,
            seed=seed,
            device=device,
            epochs=epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_timesteps=args.num_timesteps,
            dropout=args.dropout,
        )
        all_results[seed] = results

    summary = aggregate_results(all_results, method_name)
    save_results(method_name, all_results, summary)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for key in sorted(k for k in summary if k.endswith("_mean")):
        std_key = key.replace("_mean", "_std")
        std_val = summary.get(std_key, 0)
        print(f"  {key:40s}: {summary[key]:.4f} ± {std_val:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
