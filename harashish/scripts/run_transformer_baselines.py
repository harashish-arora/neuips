#!/usr/bin/env python3
"""
Run transformer-based baselines (SolTranNet) for SC3 benchmark.

Usage:
    python scripts/run_transformer_baselines.py              # all seeds
    python scripts/run_transformer_baselines.py --quick       # 1 seed, 30 epochs
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path

# Limit to 1 thread
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
torch.set_num_interop_threads(1)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics
from src.benchmarks.methods.soltrannet import DualSolTranNet, SolTranNetDataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    for solute_tok, solvent_tok, temp_feats, targets in loader:
        solute_tok = solute_tok.to(device)
        solvent_tok = solvent_tok.to(device)
        temp_feats = temp_feats.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(solute_tok, solvent_tok, temp_feats)
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
    for solute_tok, solvent_tok, temp_feats, targets in loader:
        solute_tok = solute_tok.to(device)
        solvent_tok = solvent_tok.to(device)
        temp_feats = temp_feats.to(device)

        preds = model(solute_tok, solvent_tok, temp_feats)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def run_soltrannet(splits, seed, device, epochs=100, lr=1e-3,
                   batch_size=64, patience=15, embed_dim=64):
    torch.manual_seed(seed)
    np.random.seed(seed)

    token_cache = {}

    train_ds = SolTranNetDataset(splits["train"], token_cache)
    eval_ds = SolTranNetDataset(splits["eval"], token_cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DualSolTranNet(embed_dim=embed_dim, num_heads=4, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    t0 = time.time()

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_preds, val_targets = evaluate(model, eval_loader, device)
        val_loss = float(np.mean((val_preds - val_targets) ** 2))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or wait == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}  val_rmse={np.sqrt(val_loss):.4f}"
                  + (f"  *best" if wait == 0 else f"  (wait={wait})"))

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    results = {"train_time_s": train_time}

    for split_name in ["eval", "test_hard", "test_medium", "test_easy"]:
        df = splits[split_name]
        ds = SolTranNetDataset(df, token_cache)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        preds, targets = evaluate(model, loader, device)

        solvent_col = "Solvent_Name" if "Solvent_Name" in df.columns else None
        unc_col = "Uncertainty" if "Uncertainty" in df.columns else None

        metrics = compute_metrics(
            y_true=targets, y_pred=preds,
            solvent_names=df[solvent_col].values if solvent_col else None,
            uncertainties=df[unc_col].values if unc_col else None,
        )
        results[split_name] = metrics
        print(f"  {split_name:15s}: RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}"
              + (f"  PS-RMSE={metrics.get('PS_RMSE', float('nan')):.4f}" if "PS_RMSE" in metrics else ""))

    return results


def aggregate_results(all_results, method_name):
    seeds = list(all_results.keys())
    splits = [k for k in all_results[seeds[0]].keys() if k != "train_time_s"]
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
    out_dir = RESULTS_DIR / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({str(k): v for k, v in raw_results.items()}, f, indent=2, default=str)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run SolTranNet baseline for SC3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="Quick: 1 seed, 30 epochs")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 30 if args.quick else args.epochs

    if args.gpu:
        assert torch.cuda.is_available(), "GPU requested but CUDA not available"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("SC3 SolTranNet Runner")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}, Embed dim: {args.embed_dim}")
    print()

    splits = load_all_splits()

    method_name = "soltrannet"
    all_results = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        results = run_soltrannet(
            splits=splits, seed=seed, device=device,
            epochs=epochs, lr=args.lr, batch_size=args.batch_size,
            patience=args.patience, embed_dim=args.embed_dim,
        )
        all_results[seed] = results

    summary = aggregate_results(all_results, method_name)
    save_results(method_name, all_results, summary)
    print("\nDone!")


if __name__ == "__main__":
    main()
