#!/usr/bin/env python3
"""
Grid search over hyperparameters for Uni-Mol2 + MLP / + CatBoost.

Strategy (honest grid search):
  1. For each (head, config) pair, train ONE seed (42) on train, score on eval.
  2. Rank configs by eval RMSE; the winner per head is the "best config".
  3. Re-run the winner on all 5 seeds and save standard {raw,summary}.json
     to results/{UniMolMLP,UniMolCatBoost}_grid/.
  4. Save the full grid trace to results/{...}_grid/grid_trace.json so the
     sensitivity is visible alongside the published numbers.

Tuning on `eval` and reporting test metrics on the same eval-tuned config
is standard practice — but it does mean the eval split has been used for
selection, not just as a clean held-out.

Usage:
    python scripts/grid_search_unimol.py --head mlp      --gpu 1
    python scripts/grid_search_unimol.py --head catboost --gpu 1
    python scripts/grid_search_unimol.py --head both     --gpu 1
"""

import sys
import os
import json
import time
import pickle
import argparse
import warnings
import itertools
from pathlib import Path
from copy import deepcopy

# Pre-parse --gpu before torch import
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
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostRegressor

torch.set_num_threads(4)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics, format_metrics
from src.benchmarks.methods.unimol_method import (
    UniMolMLP,
    extract_unimol_representations,
    build_unimol_features,
    build_unimol_features_numpy,
    get_solute_col, get_solvent_col,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FEATURE_DIR = ROOT / "features" / "UniMol"

EVAL_SPLITS = ["eval", "test_ood", "test_gold", "test_silver", "test_bronze"]
SEEDS = [42, 101, 123, 456, 789]
SEARCH_SEED = 42  # single seed used for the cheap exploration phase


# ============================================================================
# Grids
# ============================================================================

MLP_GRID = {
    "dropout":    [0.1, 0.3, 0.5],
    "lr":         [1e-3, 5e-4, 2e-3],
    "batch_size": [512, 1024, 2048],
}

CATBOOST_GRID = {
    "depth":        [5, 7, 9],
    "learning_rate":[0.03, 0.05, 0.1],
    "l2_leaf_reg":  [1, 3],
}

MLP_FIXED = {
    "epochs":      200,
    "patience":    25,
    "weight_decay": 1e-5,
}

CATBOOST_FIXED = {
    "iterations": 10000,
    "early_stopping_rounds": 100,
    "loss_function": "RMSE",
    "eval_metric":   "RMSE",
}


def grid_iter(grid):
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


# ============================================================================
# Feature cache (shared)
# ============================================================================

def build_or_load_feature_cache(splits, use_cuda):
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
            return (np.load(files["solute_reprs"]),
                    np.load(files["solvent_reprs"]),
                    solute_to_idx, solvent_to_idx)

    print(f"  Extracting Uni-Mol2 reprs for "
          f"{len(needed_solutes)} solutes and {len(needed_solvents)} solvents...")
    all_solutes  = sorted(needed_solutes)
    all_solvents = sorted(needed_solvents)
    solute_reprs  = extract_unimol_representations(all_solutes,  use_cuda=use_cuda)
    solvent_reprs = extract_unimol_representations(all_solvents, use_cuda=use_cuda)
    solute_to_idx  = {s: i for i, s in enumerate(all_solutes)}
    solvent_to_idx = {s: i for i, s in enumerate(all_solvents)}
    np.save(files["solute_reprs"],  solute_reprs)
    np.save(files["solvent_reprs"], solvent_reprs)
    with open(files["solute_to_idx"],  "wb") as f: pickle.dump(solute_to_idx,  f)
    with open(files["solvent_to_idx"], "wb") as f: pickle.dump(solvent_to_idx, f)
    return solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx


# ============================================================================
# Generic split evaluation helpers
# ============================================================================

def _solvent_names(df):
    if "Solvent_Name"  in df.columns: return df["Solvent_Name"].values
    if "Solvent_Canon" in df.columns: return df["Solvent_Canon"].values
    return df[get_solvent_col(df)].values


def _uncertainties(df):
    if "Uncertainty" in df.columns: return df["Uncertainty"].values
    if "sigma"       in df.columns: return df["sigma"].values
    return None


# ============================================================================
# MLP training (single seed, given hyperparams)
# ============================================================================

def train_mlp_one_seed(splits, reprs, seed, hp, device, verbose=False):
    solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx = reprs
    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed(seed)

    X_train, y_train = build_unimol_features(
        splits["train"], solute_reprs, solvent_reprs,
        solute_to_idx, solvent_to_idx, device,
    )
    X_eval, y_eval = build_unimol_features(
        splits["eval"], solute_reprs, solvent_reprs,
        solute_to_idx, solvent_to_idx, device,
    )
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=hp["batch_size"], shuffle=True,
    )

    model = UniMolMLP(dropout=hp["dropout"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=MLP_FIXED["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, min_lr=1e-6,
    )
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(MLP_FIXED["epochs"]):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(bx); loss = criterion(preds, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            val_loss = float(criterion(model(X_eval), y_eval).item())
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= MLP_FIXED["patience"]:
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state); model.to(device)
    return model, train_time


def evaluate_mlp(model, df, reprs, device, batch_size=2048):
    solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx = reprs
    X, y = build_unimol_features(df, solute_reprs, solvent_reprs,
                                 solute_to_idx, solvent_to_idx, device)
    use_amp = device.type == "cuda"
    model.eval()
    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for i in range(0, X.size(0), batch_size):
            preds.append(model(X[i:i+batch_size]).float().cpu().numpy())
    return compute_metrics(
        y_true=y.cpu().numpy(),
        y_pred=np.concatenate(preds),
        solvent_names=_solvent_names(df),
        uncertainties=_uncertainties(df),
    )


# ============================================================================
# CatBoost training (single seed, given hyperparams)
# ============================================================================

def train_catboost_one_seed(splits, reprs, seed, hp, use_gpu, verbose=False):
    solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx = reprs
    np.random.seed(seed)

    X_train, y_train = build_unimol_features_numpy(
        splits["train"], solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
    )
    X_eval, y_eval = build_unimol_features_numpy(
        splits["eval"], solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
    )

    params = {
        "iterations":     CATBOOST_FIXED["iterations"],
        "learning_rate":  hp["learning_rate"],
        "depth":          hp["depth"],
        "l2_leaf_reg":    hp["l2_leaf_reg"],
        "loss_function":  CATBOOST_FIXED["loss_function"],
        "eval_metric":    CATBOOST_FIXED["eval_metric"],
        "early_stopping_rounds": CATBOOST_FIXED["early_stopping_rounds"],
        "random_seed":    seed,
        "verbose":        500 if verbose else False,
    }
    if use_gpu:
        params["task_type"] = "GPU"; params["devices"] = "0"
    else:
        params["task_type"] = "CPU"; params["thread_count"] = 4

    model = CatBoostRegressor(**params)
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval), use_best_model=True)
    return model, time.time() - t0


def evaluate_catboost(model, df, reprs):
    solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx = reprs
    X, y_true = build_unimol_features_numpy(
        df, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx,
    )
    return compute_metrics(
        y_true=y_true,
        y_pred=model.predict(X),
        solvent_names=_solvent_names(df),
        uncertainties=_uncertainties(df),
    )


# ============================================================================
# Grid runner
# ============================================================================

def run_grid(head, splits, reprs, device, use_gpu):
    grid = MLP_GRID if head == "mlp" else CATBOOST_GRID
    configs = list(grid_iter(grid))
    print(f"\n{'='*70}\n{head.upper()}: searching {len(configs)} configs (seed {SEARCH_SEED} only)\n{'='*70}")

    trace = []
    for i, hp in enumerate(configs):
        print(f"\n[{i+1:2d}/{len(configs)}] {hp}")
        try:
            if head == "mlp":
                model, t = train_mlp_one_seed(splits, reprs, SEARCH_SEED, hp, device)
                eval_metrics = evaluate_mlp(model, splits["eval"], reprs, device)
            else:
                model, t = train_catboost_one_seed(splits, reprs, SEARCH_SEED, hp, use_gpu)
                eval_metrics = evaluate_catboost(model, splits["eval"], reprs)
            print(f"   eval RMSE={eval_metrics['RMSE']:.4f}  R²={eval_metrics['R2']:.4f}  ({t:.1f}s)")
            trace.append({
                "config": hp,
                "eval_RMSE": eval_metrics["RMSE"],
                "eval_R2":   eval_metrics["R2"],
                "eval_PS_RMSE": eval_metrics.get("PS_RMSE"),
                "train_time_s": t,
            })
        except Exception as e:
            print(f"   FAILED: {e}")
            trace.append({"config": hp, "error": str(e)})

    valid = [t for t in trace if "error" not in t]
    if not valid:
        raise RuntimeError(f"All {head} configs failed")
    valid.sort(key=lambda x: x["eval_RMSE"])

    print(f"\n{'─'*70}\nTop-3 {head} configs by eval RMSE:")
    for j, t in enumerate(valid[:3]):
        print(f"  {j+1}. eval_RMSE={t['eval_RMSE']:.4f}  config={t['config']}")

    return valid[0]["config"], trace


# ============================================================================
# Re-run winning config across all seeds
# ============================================================================

def final_run(head, best_hp, splits, reprs, device, use_gpu):
    print(f"\n{'='*70}\nFinal {head} run with best config across {len(SEEDS)} seeds")
    print(f"  config: {best_hp}\n{'='*70}")

    all_results = {}
    for seed in SEEDS:
        print(f"\n  [seed {seed}]")
        if head == "mlp":
            model, train_time = train_mlp_one_seed(splits, reprs, seed, best_hp, device)
        else:
            model, train_time = train_catboost_one_seed(splits, reprs, seed, best_hp, use_gpu)
        results = {"train_time_s": train_time}
        for split_name in EVAL_SPLITS:
            if split_name not in splits: continue
            if head == "mlp":
                m = evaluate_mlp(model, splits[split_name], reprs, device)
            else:
                m = evaluate_catboost(model, splits[split_name], reprs)
            results[split_name] = m
        print(f"    eval RMSE={results['eval']['RMSE']:.4f}  ({train_time:.1f}s)")
        all_results[seed] = results
    return all_results


# ============================================================================
# Aggregation
# ============================================================================

def aggregate(all_results, method_name, best_hp):
    seeds = list(all_results.keys())
    splits_seen = [k for k in all_results[seeds[0]] if k != "train_time_s"]
    summary = {"method": method_name, "n_seeds": len(seeds), "best_config": best_hp}
    for split in splits_seen:
        for metric in ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE", "f_aleatoric"]:
            vals = [all_results[s].get(split, {}).get(metric) for s in seeds]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                summary[f"{split}_{metric}_mean"] = float(np.mean(vals))
                summary[f"{split}_{metric}_std"]  = float(np.std(vals))
    summary["train_time_mean_s"] = float(np.mean([all_results[s]["train_time_s"] for s in seeds]))
    return summary


def save_results(method_name, raw_results, summary, trace):
    out_dir = RESULTS_DIR / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({str(k): v for k, v in raw_results.items()}, f, indent=2, default=str)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "grid_trace.json", "w") as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid search Uni-Mol2 + MLP / + CatBoost")
    parser.add_argument("--head", choices=["mlp", "catboost", "both"], required=True)
    parser.add_argument("--gpu", nargs="+", type=int, default=None)
    parser.add_argument("--cpu_catboost", action="store_true",
                        help="Use CPU for CatBoost (sometimes more stable)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda" and not args.cpu_catboost
    print(f"Device: {device}  (CB on {'GPU' if use_gpu else 'CPU'})")

    splits = load_all_splits(); print()
    reprs = build_or_load_feature_cache(splits, use_cuda=device.type == "cuda")

    heads = ["mlp", "catboost"] if args.head == "both" else [args.head]
    for head in heads:
        method_name = "UniMolMLP_grid" if head == "mlp" else "UniMolCatBoost_grid"
        best_hp, trace = run_grid(head, splits, reprs, device, use_gpu)
        all_results = final_run(head, best_hp, splits, reprs, device, use_gpu)
        summary = aggregate(all_results, method_name, best_hp)
        save_results(method_name, all_results, summary, trace)

        print(f"\n{'='*70}\n{method_name} FINAL SUMMARY\n{'='*70}")
        print(f"  best_config: {best_hp}")
        for key in sorted(k for k in summary if k.endswith("_mean")):
            std_key = key.replace("_mean", "_std")
            print(f"  {key:45s}: {summary[key]:.4f} ± {summary.get(std_key, 0):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
