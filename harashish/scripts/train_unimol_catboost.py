#!/usr/bin/env python3
"""
Run Uni-Mol2 + CatBoost baseline for SC3 benchmark.

Uses pretrained Uni-Mol2 (84M) as a frozen feature extractor; trains a
CatBoost regressor on [solute_repr ‖ solvent_repr ‖ temp_features].

Usage:
    python scripts/train_unimol_catboost.py --gpu 0
    python scripts/train_unimol_catboost.py --quick --gpu 0     # 1 seed, fewer iters
"""

import sys
import os
import json
import time
import pickle
import argparse
import warnings
from pathlib import Path

# Set CUDA_VISIBLE_DEVICES *before* importing torch / catboost
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpu", nargs="+", type=int, default=None)
_pre_args, _ = _parser.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in _pre_args.gpu)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics, format_metrics
from src.benchmarks.methods.unimol_method import (
    extract_unimol_representations,
    build_unimol_features_numpy,
    get_solute_col, get_solvent_col,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FEATURE_DIR = ROOT / "features" / "UniMol"        # SHARED with MLP variant
MODEL_DIR   = ROOT / "src" / "models" / "UniMolCatBoost"

EVAL_SPLITS = ["eval", "test_ood", "test_gold", "test_silver", "test_bronze"]


# ============================================================================
# Feature cache — shared with the MLP runner
# ============================================================================

def build_or_load_feature_cache(splits, use_cuda):
    """Extract Uni-Mol2 reprs for every unique solute/solvent across all splits.

    Cached to FEATURE_DIR; rebuilt automatically if any required SMILES is missing.
    """
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "solute_reprs":   FEATURE_DIR / "solute_reprs.npy",
        "solvent_reprs":  FEATURE_DIR / "solvent_reprs.npy",
        "solute_to_idx":  FEATURE_DIR / "solute_to_idx.pkl",
        "solvent_to_idx": FEATURE_DIR / "solvent_to_idx.pkl",
    }

    needed_solutes, needed_solvents = set(), set()
    for df in splits.values():
        needed_solutes.update(df[get_solute_col(df)].unique())
        needed_solvents.update(df[get_solvent_col(df)].unique())

    if all(p.exists() for p in files.values()):
        with open(files["solute_to_idx"],  "rb") as f: solute_to_idx  = pickle.load(f)
        with open(files["solvent_to_idx"], "rb") as f: solvent_to_idx = pickle.load(f)
        if needed_solutes.issubset(solute_to_idx) and needed_solvents.issubset(solvent_to_idx):
            solute_reprs  = np.load(files["solute_reprs"])
            solvent_reprs = np.load(files["solvent_reprs"])
            print(f"  Loaded cached reprs: {len(solute_to_idx)} solutes, "
                  f"{len(solvent_to_idx)} solvents from {FEATURE_DIR}")
            return solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx
        else:
            n_miss_sol  = len(needed_solutes  - set(solute_to_idx))
            n_miss_solv = len(needed_solvents - set(solvent_to_idx))
            print(f"  Cache stale ({n_miss_sol} new solutes, "
                  f"{n_miss_solv} new solvents) — rebuilding")

    all_solutes  = sorted(needed_solutes)
    all_solvents = sorted(needed_solvents)
    print(f"  Extracting Uni-Mol2 reprs for {len(all_solutes)} solutes "
          f"and {len(all_solvents)} solvents...")

    solute_reprs  = extract_unimol_representations(all_solutes,  use_cuda=use_cuda)
    solvent_reprs = extract_unimol_representations(all_solvents, use_cuda=use_cuda)
    solute_to_idx  = {s: i for i, s in enumerate(all_solutes)}
    solvent_to_idx = {s: i for i, s in enumerate(all_solvents)}

    np.save(files["solute_reprs"],  solute_reprs)
    np.save(files["solvent_reprs"], solvent_reprs)
    with open(files["solute_to_idx"],  "wb") as f: pickle.dump(solute_to_idx,  f)
    with open(files["solvent_to_idx"], "wb") as f: pickle.dump(solvent_to_idx, f)
    print(f"  Cached features to {FEATURE_DIR}")

    return solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx


# ============================================================================
# Per-split metric computation
# ============================================================================

def evaluate_split(model, df, solute_reprs, solvent_reprs,
                   solute_to_idx, solvent_to_idx):
    """Run model on a split and return SC3 metrics dict."""
    X, y_true = build_unimol_features_numpy(
        df, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
    )
    y_pred = model.predict(X)

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


# ============================================================================
# Train + eval one seed
# ============================================================================

def run_one_seed(splits, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
                 seed, iterations, learning_rate, depth, l2_leaf_reg, use_gpu):
    np.random.seed(seed)

    X_train, y_train = build_unimol_features_numpy(
        splits["train"], solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
    )
    X_eval, y_eval = build_unimol_features_numpy(
        splits["eval"], solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
    )

    catboost_params = {
        "iterations":     iterations,
        "learning_rate":  learning_rate,
        "depth":          depth,
        "l2_leaf_reg":    l2_leaf_reg,
        "loss_function":  "RMSE",
        "eval_metric":    "RMSE",
        "random_seed":    seed,
        "verbose":        200,
        "early_stopping_rounds": 100,
    }
    if use_gpu:
        catboost_params["task_type"] = "GPU"
        catboost_params["devices"]   = "0"  # uses CUDA_VISIBLE_DEVICES already-restricted device
    else:
        catboost_params["task_type"] = "CPU"
        catboost_params["thread_count"] = 4

    print(f"  Train: {len(X_train):,}  Eval: {len(X_eval):,}  "
          f"Features: {X_train.shape[1]}")
    print(f"  CatBoost: iters={iterations}  lr={learning_rate}  depth={depth}  "
          f"l2={l2_leaf_reg}  task={'GPU' if use_gpu else 'CPU'}")

    model = CatBoostRegressor(**catboost_params)

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval), use_best_model=True)
    train_time = time.time() - t0

    best_iter = model.get_best_iteration()
    print(f"  Training took {train_time:.1f}s (best iter: {best_iter})")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / f"seed_{seed}.cbm"))

    # Capture per-iteration train/eval loss curves (analog of MLP's epoch_losses)
    evals = model.get_evals_result()
    learning_curve = {
        "train_RMSE": evals.get("learn", {}).get("RMSE", []),
        "eval_RMSE":  evals.get("validation", {}).get("RMSE", []),
    }

    results = {
        "train_time_s": train_time,
        "best_iteration": int(best_iter),
        "learning_curve": learning_curve,
    }
    for split_name in EVAL_SPLITS:
        if split_name not in splits:
            continue
        metrics = evaluate_split(
            model, splits[split_name],
            solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
        )
        results[split_name] = metrics
        print(f"  {split_name:15s}: " + format_metrics(metrics).strip().replace("\n", "  "))

    return results


# ============================================================================
# Aggregation & persistence
# ============================================================================

def aggregate_results(all_results, method_name):
    seeds = list(all_results.keys())
    splits_seen = [k for k in all_results[seeds[0]].keys()
                   if k not in ("train_time_s", "best_iteration", "learning_curve")]
    summary = {"method": method_name, "n_seeds": len(seeds)}
    for split in splits_seen:
        for metric in ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE", "f_aleatoric"]:
            vals = [all_results[s].get(split, {}).get(metric) for s in seeds]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                summary[f"{split}_{metric}_mean"] = float(np.mean(vals))
                summary[f"{split}_{metric}_std"]  = float(np.std(vals))
    summary["train_time_mean_s"] = float(np.mean([all_results[s]["train_time_s"] for s in seeds]))
    summary["best_iteration_mean"] = float(np.mean([all_results[s]["best_iteration"] for s in seeds]))
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
    parser = argparse.ArgumentParser(description="Run Uni-Mol2 + CatBoost for SC3 benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="1 seed, 500 iters")
    parser.add_argument("--gpu", nargs="+", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--l2_leaf_reg", type=float, default=3.0,
                        help="L2 regularization. Grid-search winner: 3.0")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU is available")
    args = parser.parse_args()

    seeds = [42] if args.quick else args.seeds
    iterations = 500 if args.quick else args.iterations

    use_gpu = (not args.cpu) and torch.cuda.is_available()
    if use_gpu:
        actual = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        print(f"Using GPU(s): {args.gpu}  (CUDA_VISIBLE_DEVICES={actual})")
    else:
        print("Using CPU")

    print("=" * 70)
    print("SC3 Uni-Mol2 + CatBoost Baseline Runner")
    print("=" * 70)
    print(f"  Seeds:        {seeds}")
    print(f"  Iterations:   {iterations}  (early stopping after 100 rounds)")
    print(f"  LR:           {args.lr}")
    print(f"  Depth:        {args.depth}")
    print(f"  L2 leaf reg:  {args.l2_leaf_reg}")
    print(f"  Task type:    {'GPU' if use_gpu else 'CPU'}")
    print()

    splits = load_all_splits()
    print()
    solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx = (
        build_or_load_feature_cache(splits, use_cuda=use_gpu)
    )

    method_name = "UniMolCatBoost"
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
            iterations=iterations,
            learning_rate=args.lr,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            use_gpu=use_gpu,
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
