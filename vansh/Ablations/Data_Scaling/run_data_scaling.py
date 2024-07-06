#!/usr/bin/env python
"""
Data-scaling ablation driver.

For each (method, fraction, seed) triple, train the method on the requested
fraction of training rows and evaluate on eval / ood / sc3_gold.  Per-run
metrics are saved as JSON; an aggregated summary (mean / std across seeds)
is written when all runs for a method finish.

Hypothesis behind the experiment
--------------------------------
Simple models (LightGBM on RDKit) saturate after seeing 20-40% of the data,
while data-hungry models (FastProp deep MLP, MolMerger AttentiveFP) keep
improving as we feed them more.  The accuracy-vs-fraction curve makes the
plateau (or its absence) visible, supporting the data-scaling argument.

Usage
-----
    # Smoke test (one fraction, one seed, no GNN):
    python run_data_scaling.py --methods lgb_rdkit --fractions 0.1 --seeds 42

    # Full sweep (default fractions and seeds, all 3 methods):
    python run_data_scaling.py --gpu 0

    # Resume / extend (existing JSONs are skipped unless --force):
    python run_data_scaling.py --gpu 0 --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap CPU usage to ~60% of available cores so we don't starve other jobs
# (e.g. an existing GNN training in tmux).  Must be set BEFORE importing
# numpy / torch / lightgbm so they pick up the limits.
_N_CPUS_TOTAL = os.cpu_count() or 16
_N_JOBS = max(1, int(round(_N_CPUS_TOTAL * 0.60)))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_N_JOBS))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent
VANSH_ROOT = HERE.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.data import load_all_splits, load_cached_features  # noqa: E402
from sc3_bench.registry import get_hp  # noqa: E402

from scaling_trainers import (  # noqa: E402
    EVAL_SPLITS,
    train_lgb_rdkit,
    train_fastprop,
    train_molmerger,
    build_molmerger_cache,
)

RESULTS_DIR = HERE / "results"
DEFAULT_METHODS = ["lgb_rdkit", "fastprop", "fastprop_big", "fastprop_xl", "molmerger"]
DEFAULT_FRACTIONS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
DEFAULT_SEEDS = [42]


def _log(msg: str) -> None:
    """Single-line stamped log to stdout that flushes immediately."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _result_path(method: str, fraction: float, seed: int) -> Path:
    """Per-run JSON path: results/<method>/frac_<f>/seed_<s>.json"""
    frac_tag = f"{fraction:.3f}".rstrip("0").rstrip(".") or "0"
    out_dir = RESULTS_DIR / method / f"frac_{frac_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"seed_{seed}.json"


def _save_run(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _aggregate_method(method: str, fractions: list, seeds: list) -> dict:
    """Aggregate per-seed JSONs into a single summary for this method.

    Always scans the on-disk results for *all* fractions found, not just
    the ones in this run, so parallel sweeps that touch disjoint fractions
    don't overwrite each other's summaries.
    """
    method_dir = RESULTS_DIR / method
    found_fracs = set(fractions)
    if method_dir.exists():
        for d in method_dir.iterdir():
            if d.is_dir() and d.name.startswith("frac_"):
                try:
                    found_fracs.add(float(d.name[len("frac_"):]))
                except ValueError:
                    pass
    found_seeds = set(seeds)
    for d in method_dir.glob("frac_*/seed_*.json") if method_dir.exists() else []:
        try:
            found_seeds.add(int(d.stem[len("seed_"):]))
        except ValueError:
            pass
    fractions = sorted(found_fracs)
    seeds = sorted(found_seeds)
    summary = {"method": method, "fractions": fractions, "seeds": seeds, "by_fraction": {}}
    # Top-level (training-diagnostic) metrics that live next to the splits
    # in the per-run JSON, e.g. _train_RMSE_at_best, _best_epoch.
    DIAG_KEYS = ("_train_RMSE_at_best", "_val_RMSE_best", "_n_params",
                 "_best_epoch", "_n_train")
    for f in fractions:
        per_split: dict = {sn: {} for sn in EVAL_SPLITS}
        diag: dict = {k: [] for k in DIAG_KEYS}
        for s in seeds:
            jp = _result_path(method, f, s)
            if not jp.exists():
                continue
            with open(jp) as jf:
                payload = json.load(jf)
            metrics = payload["metrics"]
            for sn in EVAL_SPLITS:
                if sn not in metrics:
                    continue
                for mk, mv in metrics[sn].items():
                    if isinstance(mv, (int, float)) and not np.isnan(mv):
                        per_split[sn].setdefault(mk, []).append(mv)
            for k in DIAG_KEYS:
                v = metrics.get(k)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                    diag[k].append(v)
        agg = {}
        for sn in EVAL_SPLITS:
            agg[sn] = {}
            for mk, vals in per_split[sn].items():
                if vals:
                    agg[sn][f"{mk}_mean"] = float(np.mean(vals))
                    agg[sn][f"{mk}_std"] = float(np.std(vals))
                    agg[sn][f"{mk}_n"] = len(vals)
        diag_agg = {k: float(np.mean(vs)) for k, vs in diag.items() if vs}
        summary["by_fraction"][f"{f:.3f}"] = {
            "aggregated": agg,
            "n_train_mean": diag_agg.get("_n_train"),
            "diagnostics": diag_agg,
        }
    sp = RESULTS_DIR / method / "summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Data-scaling ablation: train models on increasing data fractions.",
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=DEFAULT_METHODS,
                        help="Methods to run (default: lgb + 3 fastprop sizes + molmerger).")
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS,
                        help="Training-data fractions in (0, 1].")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds (controls subsample + model init).")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index for fastprop/molmerger.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run runs whose JSON already exists.")
    parser.add_argument("--epochs-fastprop", type=int, default=None,
                        help="Override fastprop epochs (debug).")
    parser.add_argument("--epochs-molmerger", type=int, default=None,
                        help="Override molmerger epochs (debug).")
    args = parser.parse_args()

    fastprop_methods = {"fastprop", "fastprop_big", "fastprop_xl"}
    needs_gpu = any(m in fastprop_methods or m == "molmerger" for m in args.methods)
    if needs_gpu and args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = None
    if needs_gpu:
        import torch
        torch.set_num_threads(_N_JOBS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log(f"Using device: {device}  torch_threads={torch.get_num_threads()}")
    _log(f"CPU cap: using {_N_JOBS}/{_N_CPUS_TOTAL} cores (60%)")

    _log("Loading splits...")
    splits = load_all_splits(verbose=True)
    n_train = len(splits["train"])
    _log(f"Total training rows: {n_train}")
    for f in args.fractions:
        _log(f"  fraction={f:.3f}  ->  ~{int(round(n_train * f))} rows")

    cached = None
    if any(m == "lgb_rdkit" or m in fastprop_methods for m in args.methods):
        _log("Loading RDKit feature cache...")
        cached = load_cached_features("rdkit")
        if cached is None:
            _log("ERROR: feature_cache/rdkit.npz not found. Run `python sc3 cache` first.")
            sys.exit(1)
        _log(f"  RDKit cache: X_train={cached['X_train'].shape}  X_eval={cached['X_eval'].shape}")

    mm_cache = None
    if "molmerger" in args.methods:
        mm_cache = build_molmerger_cache(splits, verbose=True)

    # Build the run grid up-front so we can show overall progress.
    grid = []
    for method in args.methods:
        for fraction in args.fractions:
            for seed in args.seeds:
                grid.append((method, fraction, seed))
    n_total = len(grid)
    n_skipped = sum(1 for (m, f, s) in grid
                    if _result_path(m, f, s).exists() and not args.force)
    n_to_run = n_total - n_skipped
    _log(f"Run grid: {n_total} total ({n_skipped} already done, {n_to_run} to run)")

    grand_t0 = time.time()
    overall_pbar = tqdm(total=n_total, desc="overall", ncols=100, position=0,
                        initial=n_skipped)

    last_method = None
    for (method, fraction, seed) in grid:
        out_path = _result_path(method, fraction, seed)
        if out_path.exists() and not args.force:
            continue

        if method != last_method:
            params = get_hp(method)
            if method in fastprop_methods and args.epochs_fastprop is not None:
                params = dict(params); params["epochs"] = args.epochs_fastprop
            if method == "molmerger" and args.epochs_molmerger is not None:
                params = dict(params); params["epochs"] = args.epochs_molmerger
            _log(f"\n{'='*70}\n  METHOD: {method}\n{'='*70}")
            _log(f"  HP: {params}")
            last_method = method

        _log(f"\n--- {method}  fraction={fraction:.3f}  seed={seed}  "
             f"({overall_pbar.n+1}/{n_total}) ---")
        t0 = time.time()
        try:
            if method == "lgb_rdkit":
                metrics = train_lgb_rdkit(fraction, seed, splits, params, cached)
            elif method in fastprop_methods:
                metrics = train_fastprop(fraction, seed, splits, params, cached, device)
            elif method == "molmerger":
                metrics = train_molmerger(fraction, seed, splits, params, mm_cache, device)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            _log(f"[ERROR] {method} f={fraction} seed={seed}: {e}")
            import traceback; traceback.print_exc()
            overall_pbar.update(1)
            continue

        dt = time.time() - t0
        for sn in EVAL_SPLITS:
            if sn in metrics:
                m = metrics[sn]
                _log(f"  {sn:10s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}  N={m['N']}")
        _log(f"  done in {dt:.1f}s   train_rows={metrics.get('_n_train', '?')}")

        payload = {
            "method": method,
            "fraction": fraction,
            "seed": seed,
            "params": params,
            "wall_time_s": dt,
            "metrics": metrics,
        }
        _save_run(out_path, payload)

        # Re-aggregate this method as we go so partial results are usable.
        _aggregate_method(method, args.fractions, args.seeds)
        overall_pbar.update(1)
        elapsed = time.time() - grand_t0
        done_now = overall_pbar.n - n_skipped
        if done_now > 0 and overall_pbar.n < n_total:
            eta = elapsed / done_now * (n_total - overall_pbar.n)
            _log(f"  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    overall_pbar.close()
    grand_dt = time.time() - grand_t0
    _log(f"\nDone in {grand_dt/60:.1f} min.  Results under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
