#!/usr/bin/env python3
"""
Run Uni-Mol2 + MLP baseline for SC3 benchmark.

Uses pretrained Uni-Mol2 (84M) as a frozen feature extractor; trains a
regression MLP head on [solute_repr ‖ solvent_repr ‖ temp_features].

Usage:
    python scripts/train_unimol_mlp.py --gpu 0
    python scripts/train_unimol_mlp.py --quick --gpu 0
"""

import sys
import os
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path

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

torch.set_num_threads(4)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics, format_metrics
from src.benchmarks.methods.unimol_method import (
    UniMolMLP,
    extract_unimol_representations,
    build_unimol_features,
    get_solute_col,
    get_solvent_col,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FEATURE_DIR = ROOT / "features" / "UniMol"
MODEL_DIR = ROOT / "src" / "models" / "UniMolMLP"

EVAL_SPLITS = ["eval", "test_ood", "test_gold", "test_silver", "test_bronze"]


def build_or_load_feature_cache(splits, use_cuda):
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "solute_reprs": FEATURE_DIR / "solute_reprs.npy",
        "solvent_reprs": FEATURE_DIR / "solvent_reprs.npy",
        "solute_to_idx": FEATURE_DIR / "solute_to_idx.pkl",
        "solvent_to_idx": FEATURE_DIR / "solvent_to_idx.pkl",
    }

    needed_solutes, needed_solvents = set(), set()
    for df in splits.values():
        needed_solutes.update(df[get_solute_col(df)].unique())
        needed_solvents.update(df[get_solvent_col(df)].unique())

    if all(p.exists() for p in files.values()):
        with open(files["solute_to_idx"], "rb") as f:
            solute_to_idx = pickle.load(f)
        with open(files["solvent_to_idx"], "rb") as f:
            solvent_to_idx = pickle.load(f)

        if needed_solutes.issubset(solute_to_idx) and needed_solvents.issubset(solvent_to_idx):
            solute_reprs = np.load(files["solute_reprs"])
            solvent_reprs = np.load(files["solvent_reprs"])
            print(
                f"  Loaded cached reprs: {len(solute_to_idx)} solutes, "
                f"{len(solvent_to_idx)} solvents from {FEATURE_DIR}"
            )
            return solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx

        n_miss_sol = len(needed_solutes - set(solute_to_idx))
        n_miss_solv = len(needed_solvents - set(solvent_to_idx))
        print(f"  Cache stale ({n_miss_sol} new solutes, {n_miss_solv} new solvents) — rebuilding")

    all_solutes = sorted(needed_solutes)
    all_solvents = sorted(needed_solvents)
    print(f"  Extracting Uni-Mol2 reprs for {len(all_solutes)} solutes and {len(all_solvents)} solvents...")

    solute_reprs = extract_unimol_representations(all_solutes, use_cuda=use_cuda)
    solvent_reprs = extract_unimol_representations(all_solvents, use_cuda=use_cuda)
    solute_to_idx = {s: i for i, s in enumerate(all_solutes)}
    solvent_to_idx = {s: i for i, s in enumerate(all_solvents)}

    np.save(files["solute_reprs"], solute_reprs)
    np.save(files["solvent_reprs"], solvent_reprs)
    with open(files["solute_to_idx"], "wb") as f:
        pickle.dump(solute_to_idx, f)
    with open(files["solvent_to_idx"], "wb") as f:
        pickle.dump(solvent_to_idx, f)

    print(f"  Cached features to {FEATURE_DIR}")
    return solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx


def evaluate_split(model, df, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx, device, batch_size=2048):
    X, y = build_unimol_features(df, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx, device)
    use_amp = device.type == "cuda"

    model.eval()
    preds = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        for i in range(0, X.size(0), batch_size):
            preds.append(model(X[i:i + batch_size]).float().cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = y.cpu().numpy()

    if "Solvent_Name" in df.columns:
        solvent_names = df["Solvent_Name"].values
    elif "Solvent_Canon" in df.columns:
        solvent_names = df["Solvent_Canon"].values
    else:
        solvent_names = df[get_solvent_col(df)].values

    uncertainties = None
    if "Uncertainty" in df.columns:
        uncertainties = df["Uncertainty"].values
    elif "sigma" in df.columns:
        uncertainties = df["sigma"].values

    return compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        solvent_names=solvent_names,
        uncertainties=uncertainties,
    )


def run_one_seed(splits, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
                 seed, device, epochs, lr, batch_size, patience, dropout):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    X_train, y_train = build_unimol_features(
        splits["train"], solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx, device
    )
    X_eval, y_eval = build_unimol_features(
        splits["eval"], solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx, device
    )
    n_train = X_train.size(0)

    model = UniMolMLP(dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}  |  Train: {n_train:,}  Eval: {len(X_eval):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-6
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

        perm = torch.randperm(n_train, device=device)
        total = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            if idx.numel() < 2:
                continue

            bx, by = X_train[idx], y_train[idx]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(bx)
                loss = criterion(preds, by)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total += loss.item() * bx.size(0)

        train_loss = total / n_train
        epoch_losses.append(train_loss)

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            val_loss = float(criterion(model(X_eval), y_eval).item())
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    torch.save(best_state, MODEL_DIR / f"seed_{seed}.pt")

    results = {"train_time_s": train_time, "epoch_losses": epoch_losses}
    for split_name in EVAL_SPLITS:
        if split_name not in splits:
            continue
        metrics = evaluate_split(
            model, splits[split_name],
            solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx, device
        )
        results[split_name] = metrics
        print(f"  {split_name:15s}: " + format_metrics(metrics).strip().replace("\n", "  "))

    return results


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


def main():
    parser = argparse.ArgumentParser(description="Run Uni-Mol2 + MLP for SC3 benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="1 seed, 30 epochs")
    parser.add_argument("--gpu", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=25)
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
    print("SC3 Uni-Mol2 + MLP Baseline Runner")
    print("=" * 70)
    print(f"  Seeds:      {seeds}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}  Patience: {args.patience}")
    print(f"  Batch:      {args.batch_size}  LR: {args.lr}  Dropout: {args.dropout}")
    print()

    splits = load_all_splits()
    print()

    solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx = build_or_load_feature_cache(
        splits, use_cuda=device.type == "cuda"
    )

    method_name = "UniMolMLP"
    all_results = {}
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Seed {seed}")
        print(f"{'='*70}")
        results = run_one_seed(
            splits=splits,
            solute_reprs=solute_reprs,
            solvent_reprs=solvent_reprs,
            solute_to_idx=solute_to_idx,
            solvent_to_idx=solvent_to_idx,
            seed=seed,
            device=device,
            epochs=epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
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
        print(f"  {key:45s}: {summary[key]:.4f} ± {std_val:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
