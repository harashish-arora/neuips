#!/usr/bin/env python3
"""
Run SolTranNet baseline for SC3 benchmark.

Uses dual MAT encoders (solute + solvent) fused with temperature features
to predict LogS.

Usage:
    python scripts/train_soltrannet.py --gpu 0
    python scripts/train_soltrannet.py --quick --gpu 0     # 1 seed, 30 epochs
"""

import sys
import os
import json
import time
import argparse
import warnings
from pathlib import Path

# Set CUDA_VISIBLE_DEVICES before importing torch
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", nargs="+", type=int, default=None)
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _pre_args.gpu)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_num_threads(4)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics, format_metrics
from src.benchmarks.methods.soltrannet import (
    DualSolTranNet,
    SolTranNetDataset,
    soltrannet_collate_fn,
    get_solvent_col,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "src" / "models" / "SolTranNet"

EVAL_SPLITS = ["eval", "test_ood", "test_gold", "test_silver", "test_bronze"]


# ============================================================================
# Metric helpers
# ============================================================================

def _get_solvent_names(df):
    if "Solvent_Name" in df.columns:
        return df["Solvent_Name"].values
    if "Solvent_Canon" in df.columns:
        return df["Solvent_Canon"].values
    return df[get_solvent_col(df)].values


def _get_uncertainties(df):
    if "Uncertainty" in df.columns:
        return df["Uncertainty"].values
    if "sigma" in df.columns:
        return df["sigma"].values
    return None


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_split(model, df, graph_cache, device, batch_size=512, num_workers=4):
    ds = SolTranNetDataset(df, graph_cache=graph_cache)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=soltrannet_collate_fn,
    )

    use_amp = device.type == "cuda"
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for batch in loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            sol_feats, sol_adj, sol_mask, solv_feats, solv_adj, solv_mask, temp_feats, targets = batch

            preds = model(
                sol_feats,
                sol_mask,
                sol_adj,
                solv_feats,
                solv_mask,
                solv_adj,
                temp_feats,
            )

            all_preds.append(preds.float().detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    return compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        solvent_names=_get_solvent_names(df),
        uncertainties=_get_uncertainties(df),
    )


# ============================================================================
# Train + eval one seed
# ============================================================================

def run_one_seed(
    splits,
    graph_cache,
    seed,
    device,
    epochs,
    lr,
    batch_size,
    patience,
    d_model,
    n_layers,
    n_heads,
    dropout,
    lambda_attention,
    n_dense,
    num_workers,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_ds = SolTranNetDataset(splits["train"], graph_cache=graph_cache)
    eval_ds = SolTranNetDataset(splits["eval"], graph_cache=graph_cache)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=soltrannet_collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=soltrannet_collate_fn,
    )

    model = DualSolTranNet(
        d_model=d_model,
        N=n_layers,
        h=n_heads,
        dropout=dropout,
        lambda_attention=lambda_attention,
        N_dense=n_dense,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}  |  Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
        min_lr=1e-6,
    )
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    wait = 0
    epoch_losses = []

    t0 = time.time()

    for epoch in range(epochs):
        ep_start = time.time()
        model.train()

        total_loss = 0.0
        n_seen = 0

        for batch in train_loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            sol_feats, sol_adj, sol_mask, solv_feats, solv_adj, solv_mask, temp_feats, targets = batch

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(
                    sol_feats,
                    sol_mask,
                    sol_adj,
                    solv_feats,
                    solv_mask,
                    solv_adj,
                    temp_feats,
                )
                loss = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        train_loss = total_loss / max(n_seen, 1)
        epoch_losses.append(train_loss)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for batch in eval_loader:
                batch = [x.to(device, non_blocking=True) for x in batch]
                sol_feats, sol_adj, sol_mask, solv_feats, solv_adj, solv_mask, temp_feats, targets = batch

                preds = model(
                    sol_feats,
                    sol_mask,
                    sol_adj,
                    solv_feats,
                    solv_mask,
                    solv_adj,
                    temp_feats,
                )
                val_loss = criterion(preds, targets)

                bs = targets.size(0)
                val_total += val_loss.item() * bs
                val_n += bs

        val_loss = val_total / max(val_n, 1)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            tag = "  *best"
        else:
            wait += 1
            tag = f"  (wait={wait})"

        if (epoch + 1) % 5 == 0 or wait == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d} [{time.time()-ep_start:.1f}s]: "
                f"train_loss={train_loss:.4f}  val_rmse={np.sqrt(val_loss):.4f}  "
                f"lr={lr_now:.2e}{tag}"
            )

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    train_time = time.time() - t0
    print(f"  Training took {train_time:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "seed": seed,
            "config": {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "dropout": dropout,
                "lambda_attention": lambda_attention,
                "n_dense": n_dense,
            },
        },
        MODEL_DIR / f"seed_{seed}.pt",
    )

    results = {"train_time_s": train_time, "epoch_losses": epoch_losses}
    for split_name in EVAL_SPLITS:
        if split_name not in splits:
            continue
        metrics = evaluate_split(
            model,
            splits[split_name],
            graph_cache=graph_cache,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        results[split_name] = metrics
        print(f"  {split_name:15s}: " + format_metrics(metrics).strip().replace("\n", "  "))

    return results


# ============================================================================
# Aggregation & persistence
# ============================================================================

def aggregate_results(all_results, method_name):
    seeds = list(all_results.keys())
    splits_seen = [k for k in all_results[seeds[0]].keys() if k not in ("train_time_s", "epoch_losses")]

    summary = {"method": method_name, "n_seeds": len(seeds)}

    for split in splits_seen:
        for metric in ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE", "f_aleatoric"]:
            vals = [all_results[s].get(split, {}).get(metric) for s in seeds]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
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
    parser = argparse.ArgumentParser(description="Run SolTranNet baseline for SC3 benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="1 seed, 30 epochs")
    parser.add_argument("--gpu", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda_attention", type=float, default=0.5)
    parser.add_argument("--n_dense", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    epochs = 30 if args.quick else args.epochs

    if torch.cuda.is_available():
        device = torch.device("cuda")
        actual = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"Using GPU(s): {args.gpu}  (CUDA_VISIBLE_DEVICES={actual})")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("SC3 SolTranNet Baseline Runner")
    print("=" * 70)
    print(f"  Seeds:      {seeds}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}  Patience: {args.patience}")
    print(f"  Batch:      {args.batch_size}  LR: {args.lr}")
    print(f"  d_model:    {args.d_model}  Layers: {args.n_layers}  Heads: {args.n_heads}")
    print(f"  Dropout:    {args.dropout}  Lambda_attn: {args.lambda_attention}")
    print(f"  Workers:    {args.num_workers}")
    print()

    splits = load_all_splits()
    graph_cache = {}

    method_name = "SolTranNet"
    all_results = {}

    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed {seed}")
        print(f"{'='*70}")
        results = run_one_seed(
            splits=splits,
            graph_cache=graph_cache,
            seed=seed,
            device=device,
            epochs=epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dropout=args.dropout,
            lambda_attention=args.lambda_attention,
            n_dense=args.n_dense,
            num_workers=args.num_workers,
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
        print(f"  {key:45s}: {summary[key]:.4f} ± {std_val:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
