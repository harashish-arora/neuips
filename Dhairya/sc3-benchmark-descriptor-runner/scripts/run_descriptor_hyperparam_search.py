#!/usr/bin/env python3
"""
Manual hyperparameter search for descriptor-based baselines (no sklearn / optuna).

Two-phase sweep per model type (fastprop, fastsolv, mlp):

  Phase A (grid A): lr × batch_size × hidden_dims × dropout with fixed epochs/patience;
                    pick best by lowest eval best_val_loss.

  Phase B (grid B): epochs × patience using the best Phase-A (lr, bs, hd, do).

Outputs go under results/hpo_descriptor/.

Usage:
    python scripts/run_descriptor_hyperparam_search.py
    python scripts/run_descriptor_hyperparam_search.py --gpu --quick
    python scripts/run_descriptor_hyperparam_search.py --only fastprop
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
warnings.filterwarnings("ignore")

# Project root (parent of scripts/)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_descriptor_baselines as rdb  # noqa: E402

from src.benchmarks.data_splits import load_all_splits  # noqa: E402
from src.benchmarks.featurizers import get_featurizer  # noqa: E402
from src.benchmarks.methods.descriptor_models import build_feature_cache  # noqa: E402


# =============================================================================
# Hyperparameter grids  (edit these to adjust the search space)
#
# Grid A: lr × batch_size × hidden_dims × dropout  (fixed epochs / patience)
# Grid B: epochs  at best Grid-A values (fixed patience)
# =============================================================================

_HPO_LRS = (5e-4, 1e-3)
_HPO_BSS = (128, 256)
_HPO_HDS = (
    (256, 128),
    (512, 256, 128),
    (512, 256, 128, 64),
)
_HPO_DROPOUTS = (0.05, 0.1)
_HPO_A_EPOCHS = 100
_HPO_A_PATIENCE = 20

_HPO_EPS = (100, 150, 200)

HPO_OUT_DIR = rdb.RESULTS_DIR / "hpo_descriptor"


def _build_grid_a() -> list[dict]:
    trials: list[dict] = []
    for lr, bs, hd, do in itertools.product(_HPO_LRS, _HPO_BSS, _HPO_HDS, _HPO_DROPOUTS):
        trials.append(
            {
                "lr": lr,
                "batch_size": bs,
                "hidden_dims": list(hd),
                "dropout": do,
                "epochs": _HPO_A_EPOCHS,
                "patience": _HPO_A_PATIENCE,
            }
        )
    return trials


def _build_grid_b(anchor: dict) -> list[dict]:
    out: list[dict] = []
    for ep in _HPO_EPS:
        out.append(
            {
                "lr": float(anchor["lr"]),
                "batch_size": int(anchor["batch_size"]),
                "hidden_dims": list(anchor["hidden_dims"]),
                "dropout": float(anchor["dropout"]),
                "epochs": int(ep),
                "patience": int(anchor["patience"]),
            }
        )
    return out


def _apply_quick_grid_a(trials: list[dict]) -> list[dict]:
    if not trials:
        return []
    t0 = copy.deepcopy(trials[0])
    t0["epochs"] = min(int(t0.get("epochs", 30)), 30)
    t0["patience"] = min(int(t0.get("patience", 10)), 10)
    return [t0]


def _apply_quick_grid_b(trials: list[dict]) -> list[dict]:
    if not trials:
        return []
    t0 = copy.deepcopy(trials[0])
    t0["epochs"] = min(int(t0.get("epochs", 30)), 30)
    t0["patience"] = min(int(t0.get("patience", 10)), 10)
    return [t0]


def _trial_method_name(model_type: str, trial_idx: int) -> str:
    return f"hpo_{model_type}_trial_{trial_idx:03d}"


def _run_trial_row(
    model_type: str,
    trial_idx: int,
    cfg: dict,
    phase: str,
    splits,
    feature_cache,
    feature_names,
    device,
    seed: int,
) -> tuple[dict, float, dict]:
    method_name = _trial_method_name(model_type, trial_idx)
    print("=" * 70)
    print(f"[{model_type}] [{phase}] Trial {trial_idx + 1}  method_name={method_name}")
    print(f"  config: {cfg}")

    summary_file = rdb.RESULTS_DIR / method_name / f"seed_{seed}" / "seed_summary.json"
    if summary_file.exists():
        print(f"  -> Found completed run at {summary_file}, skipping.")
        with open(summary_file, "r") as f:
            results = json.load(f)
        elapsed = results.get("train_time_s", 0.0)
    else:
        t0 = time.time()
        results = rdb.run_model(
            model_type=model_type,
            splits=splits,
            feature_cache=feature_cache,
            feature_names=feature_names,
            seed=seed,
            device=device,
            epochs=int(cfg["epochs"]),
            lr=float(cfg["lr"]),
            batch_size=int(cfg["batch_size"]),
            patience=int(cfg["patience"]),
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout=float(cfg["dropout"]),
            method_name=method_name,
        )
        elapsed = time.time() - t0

    val_loss = float(results.get("best_val_loss", float("nan")))
    row = {
        "model_type": model_type,
        "hpo_phase": phase,
        "trial_idx": trial_idx,
        "method_name": method_name,
        "config": cfg,
        "best_val_loss": val_loss,
        "best_epoch": results.get("best_epoch"),
        "train_time_s": results.get("train_time_s"),
        "elapsed_wall_s": elapsed,
        "eval_rmse": None,
        "test_hard_ps_rmse": None,
    }
    ev = results.get("eval")
    if isinstance(ev, dict):
        row["eval_rmse"] = ev.get("RMSE")
    th = results.get("test_hard")
    if isinstance(th, dict):
        row["test_hard_ps_rmse"] = th.get("PS_RMSE")

    print(f"  -> best_val_loss={val_loss:.6f}  (elapsed {elapsed:.1f}s)")
    full = dict(results)
    full["hpo_trial_idx"] = trial_idx
    full["hpo_phase"] = phase
    full["hpo_config"] = dict(cfg)
    full["hpo_elapsed_wall_s"] = elapsed
    return row, val_loss, full


def main():
    parser = argparse.ArgumentParser(
        description="Manual descriptor-based model hyperparameter search (grid A then grid B)"
    )
    parser.add_argument("--gpu", action="store_true", help="Force CUDA")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Phase A: one config; Phase B: one (ep,pt); both with shorter epochs/patience caps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for each trial")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["fastprop", "fastsolv", "mlp"],
        help="Run HPO for a single model type only",
    )
    args = parser.parse_args()

    if args.gpu:
        assert torch.cuda.is_available(), "GPU requested but CUDA not available"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    types_to_run = [args.only] if args.only else ["fastprop", "fastsolv", "mlp"]

    print("SC3 descriptor-based manual HPO (grid A → best anchor → grid B)")
    print(f"  Device: {device}")
    print(f"  Model types: {types_to_run}")
    print(f"  Seed: {args.seed}")
    print()

    splits = load_all_splits()
    featurizer = get_featurizer("rdkit")
    feature_cache, feature_names = build_feature_cache(splits, featurizer)

    HPO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_type: dict = {}
    raw_store: dict[str, dict[int, dict]] = {}
    t_all = time.time()

    for model_type in types_to_run:
        grid_a = _build_grid_a()
        if args.quick:
            grid_a = _apply_quick_grid_a(grid_a)

        print()
        print("#" * 70)
        print(f"# Model: {model_type}  |  Phase A: {len(grid_a)} trial(s)")
        print("#" * 70)

        rows: list[dict] = []
        raw_by_trial: dict[int, dict] = {}
        trial_idx = 0

        best_a_loss = float("inf")
        best_a_cfg: dict | None = None
        best_a_row: dict | None = None

        for cfg in grid_a:
            row, val_loss, full = _run_trial_row(
                model_type, trial_idx, cfg, "grid_a", splits,
                feature_cache, feature_names, device, args.seed,
            )
            rows.append(row)
            raw_by_trial[trial_idx] = full
            if val_loss < best_a_loss:
                best_a_loss = val_loss
                best_a_cfg = dict(cfg)
                best_a_row = row
            trial_idx += 1

        if best_a_cfg is None:
            print(f"[{model_type}] Phase A produced no valid best — skipping Phase B.")
            by_type[model_type] = {
                "skipped": True,
                "reason": "no grid_a results",
                "trials": rows,
                "best": None,
            }
            raw_store[model_type] = raw_by_trial
            continue

        grid_b = _build_grid_b(best_a_cfg)
        if args.quick:
            grid_b = _apply_quick_grid_b(grid_b)

        print()
        print("#" * 70)
        print(
            f"# Model: {model_type}  |  Phase B: {len(grid_b)} trial(s) "
            f"(anchor from grid A: lr={best_a_cfg['lr']}, bs={best_a_cfg['batch_size']}, "
            f"hd={best_a_cfg['hidden_dims']}, do={best_a_cfg['dropout']})"
        )
        print("#" * 70)

        best_b_loss = float("inf")
        best_b_row: dict | None = None

        for cfg in grid_b:
            row, val_loss, full = _run_trial_row(
                model_type, trial_idx, cfg, "grid_b", splits,
                feature_cache, feature_names, device, args.seed,
            )
            rows.append(row)
            raw_by_trial[trial_idx] = full
            if val_loss < best_b_loss:
                best_b_loss = val_loss
                best_b_row = row
            trial_idx += 1

        best_overall_loss = float("inf")
        best_overall: dict | None = None
        for r in rows:
            v = float(r.get("best_val_loss", float("nan")))
            if v < best_overall_loss:
                best_overall_loss = v
                best_overall = r

        best_block = None
        if best_overall is not None:
            best_block = {
                "best_val_loss": best_overall_loss,
                "best_phase": best_overall.get("hpo_phase"),
                "best_config": best_overall.get("config"),
                "method_name": best_overall.get("method_name"),
                "results_dir": str(rdb.RESULTS_DIR / best_overall["method_name"]),
                "grid_a_best": {
                    "best_val_loss": best_a_loss,
                    "config": best_a_cfg,
                    "method_name": best_a_row.get("method_name") if best_a_row else None,
                },
                "grid_b_best": (
                    {
                        "best_val_loss": best_b_loss,
                        "config": best_b_row.get("config") if best_b_row else None,
                        "method_name": best_b_row.get("method_name") if best_b_row else None,
                    }
                    if best_b_row is not None
                    else None
                ),
            }

        raw_store[model_type] = raw_by_trial
        by_type[model_type] = {
            "n_trials_phase_a": len(grid_a),
            "n_trials_phase_b": len(grid_b),
            "n_trials_total": len(rows),
            "trials": rows,
            "best": best_block,
        }

        out_type = HPO_OUT_DIR / f"hpo_best_{model_type}.json"
        with open(out_type, "w") as f:
            json.dump({"model_type": model_type, **by_type[model_type]}, f, indent=2, default=str)
        print(f"[{model_type}] Wrote {out_type}")

        # Same layout as run_descriptor_baselines: raw_results.json + summary.json per model type
        method_hpo = f"hpo_descriptor_{model_type}"
        summary_model = rdb.aggregate_results(raw_by_trial, method_hpo)
        rdb.save_results(method_hpo, raw_by_trial, summary_model)
        print(f"[{model_type}] Also wrote {rdb.RESULTS_DIR / method_hpo / 'raw_results.json'}")
        print(f"[{model_type}] Also wrote {rdb.RESULTS_DIR / method_hpo / 'summary.json'}")

    summary = {
        "metric": "best_val_loss (eval MSE); Phase B uses best Phase-A (lr,bs,hd,do)",
        "seed": args.seed,
        "device": str(device),
        "model_types": types_to_run,
        "by_type": by_type,
        "total_wall_s": time.time() - t_all,
    }

    out_json = HPO_OUT_DIR / "hpo_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    best_all = HPO_OUT_DIR / "best_per_type.json"
    with open(best_all, "w") as f:
        json.dump(
            {k: v.get("best") for k, v in by_type.items() if isinstance(v, dict)},
            f,
            indent=2,
            default=str,
        )

    # Roll-up: raw_results.json + summary.json under results/hpo_descriptor/
    combined_raw: dict = {}
    per_model_summaries: dict = {}
    for model_type in types_to_run:
        rb = raw_store.get(model_type)
        if not rb:
            continue
        combined_raw[model_type] = {str(k): v for k, v in rb.items()}
        per_model_summaries[model_type] = rdb.aggregate_results(
            rb,
            f"hpo_descriptor::{model_type}",
        )

    umbrella_summary = {
        "method": "hpo_descriptor",
        "metric": "two-phase: grid A then grid B on best A anchor",
        "seed": args.seed,
        "device": str(device),
        "model_types": types_to_run,
        "total_wall_s": summary["total_wall_s"],
        "best_per_model": {k: v.get("best") for k, v in by_type.items() if isinstance(v, dict)},
        "aggregate_metrics_by_model": per_model_summaries,
        "by_type": {k: v for k, v in by_type.items() if isinstance(v, dict)},
    }

    rdb.save_results(
        "hpo_descriptor",
        combined_raw if combined_raw else {"_note": "no trials completed"},
        umbrella_summary,
    )
    print(f"Wrote {rdb.RESULTS_DIR / 'hpo_descriptor' / 'raw_results.json'}")
    print(f"Wrote {rdb.RESULTS_DIR / 'hpo_descriptor' / 'summary.json'}")

    print()
    print("=" * 70)
    print("Summary (best overall per model type)")
    for model_type in types_to_run:
        b = by_type.get(model_type, {}).get("best")
        if b:
            print(
                f"  {model_type}: best_val_loss={b.get('best_val_loss')}  "
                f"phase={b.get('best_phase')}  config={b.get('best_config')}"
            )
        elif by_type.get(model_type, {}).get("skipped"):
            print(f"  {model_type}: (skipped)")
    print(f"Wrote {out_json}")
    print(f"Wrote {best_all}")


if __name__ == "__main__":
    main()
