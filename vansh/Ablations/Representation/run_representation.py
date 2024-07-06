#!/usr/bin/env python
"""
Representation ablation driver  (Q1.2: representation matters how much?)

Trains a single fixed architecture (LightGBM with the tuned ``lgb_rdkit``
hyperparameters) on top of several different molecular featurizers and
evaluates each on every benchmark split.  Holding the model and the HPs
fixed isolates the contribution of the representation itself.

Featurizers (all live in ``sc3_bench/featurizers.py``):

  rdkit         158 RDKit 2D descriptors      (~320 features w/ T)
  morgan        1024-bit Morgan ECFP4         (~2052 features w/ T)
  dissolvr      RDKit + MOSE + Joback + Abr.  (~356 features w/ T)
  mordred       ~1600 Mordred 2D descriptors  (~3226 features w/ T)
  maccs         166-bit MACCS substructures   (~338 features w/ T)
  atompair      1024-bit Atom-Pair fingerprt  (~2052 features w/ T)
  abraham_only  6 Abraham/Joback proxies      (~16 features w/ T)

For each (featurizer, seed) pair we:
  1. Load the precomputed feature cache (``feature_cache/<feat>.npz``).
  2. Train LightGBM with the tuned HPs and early-stopping on ``eval``.
  3. Evaluate on eval / ood / sc3_gold / sc3_silver / sc3_bronze.
  4. Save per-run JSON; aggregated mean/std summary is rebuilt after each run.

Usage
-----
    # Smoke test (one featurizer, one seed)
    python run_representation.py --featurizers rdkit --seeds 42

    # Full sweep with default seed (single seed, all featurizers)
    python run_representation.py

    # Full sweep with 5 seeds (publication-quality error bars)
    python run_representation.py --seeds 42 101 123 456 789

    # Resume / extend (existing JSONs are skipped unless --force)
    python run_representation.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Cap CPU usage to ~60% of cores so we don't starve other jobs.  Must be
# set BEFORE numpy / lightgbm import.
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
from sc3_bench.evaluate import compute_metrics  # noqa: E402
from sc3_bench.registry import get_hp  # noqa: E402

RESULTS_DIR = HERE / "results"

DEFAULT_FEATURIZERS = [
    "rdkit", "morgan", "dissolvr", "mordred",
    "maccs", "atompair", "abraham_only",
]
DEFAULT_SEEDS = [42]
EVAL_SPLITS = ["eval", "ood", "sc3_gold", "sc3_silver", "sc3_bronze"]

# All featurizers use the same fixed model + HPs to isolate representation.
FIXED_MODEL_HP_KEY = "lgb_rdkit"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _result_path(featurizer: str, seed: int) -> Path:
    out_dir = RESULTS_DIR / featurizer
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"seed_{seed}.json"


def _split_meta(splits: dict) -> dict:
    """Collect (y_true, solvent_names, uncertainties) per evaluation split."""
    meta = {}
    for sname in EVAL_SPLITS:
        df = splits[sname]
        meta[sname] = {
            "y_true": df["LogS"].values,
            "solvent_names": df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
            "uncertainties": df["Uncertainty"].values if "Uncertainty" in df.columns else None,
        }
    return meta


def _train_lgb_on_featurizer(featurizer: str, seed: int, splits: dict, params: dict) -> dict:
    """Train one LightGBM model on the given featurizer's cached features."""
    from lightgbm import LGBMRegressor
    import lightgbm as lgb

    cached = load_cached_features(featurizer)
    if cached is None:
        raise FileNotFoundError(
            f"feature_cache/{featurizer}.npz not found. "
            f"Run `python sc3 cache --featurizers {featurizer}` first."
        )

    X_tr, y_tr = cached["X_train"], cached["y_train"]
    X_ev, y_ev = cached["X_eval"], cached["y_eval"]
    n_features = X_tr.shape[1]
    _log(f"  features: n_features={n_features}  X_train={X_tr.shape}  X_eval={X_ev.shape}")

    n_estimators = int(params.get("n_estimators", 3000))
    model = LGBMRegressor(random_state=seed, n_jobs=_N_JOBS, verbose=-1, **params)
    pbar = tqdm(total=n_estimators, desc=f"lgb {featurizer} s={seed}", ncols=100, leave=False)

    def _tqdm_cb(env):
        pbar.update(1)
        if env.evaluation_result_list:
            name, _, val, _ = env.evaluation_result_list[0]
            pbar.set_postfix(val=f"{val:.4f}")

    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_ev, y_ev)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            _tqdm_cb,
        ],
    )
    pbar.close()
    best_iter = getattr(model, "best_iteration_", None)
    train_dt = time.time() - t0
    _log(f"  trained in {train_dt:.1f}s  best_iter={best_iter}")

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        Xs = cached.get(f"X_{sname}")
        if Xs is None:
            _log(f"  WARN: cache for {featurizer} missing X_{sname}; skipping split.")
            continue
        preds = model.predict(Xs)
        metrics[sname] = compute_metrics(
            meta[sname]["y_true"], preds,
            meta[sname]["solvent_names"], meta[sname]["uncertainties"],
        )
        m = metrics[sname]
        ps = m.get("PS_RMSE", float("nan"))
        ps_str = f"PS={ps:.4f}" if not np.isnan(ps) else "PS=  n/a"
        _log(f"  eval[{sname:11s}]  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
             f"R2={m['R2']:.4f}  {ps_str}  N={m['N']}")

    metrics["_n_features"] = int(n_features)
    metrics["_n_train"] = int(len(y_tr))
    metrics["_best_iter"] = int(best_iter) if best_iter else None
    metrics["_train_time_s"] = float(train_dt)
    return metrics


def _save_run(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _aggregate_featurizer(featurizer: str, seeds: list) -> dict:
    """Aggregate per-seed JSONs into a single summary for this featurizer."""
    feat_dir = RESULTS_DIR / featurizer
    found_seeds = set(seeds)
    if feat_dir.exists():
        for jp in feat_dir.glob("seed_*.json"):
            try:
                found_seeds.add(int(jp.stem[len("seed_"):]))
            except ValueError:
                pass
    seeds = sorted(found_seeds)

    DIAG_KEYS = ("_n_features", "_n_train", "_best_iter", "_train_time_s")
    summary = {"featurizer": featurizer, "seeds": seeds, "by_split": {sn: {} for sn in EVAL_SPLITS}}
    diag: dict = {k: [] for k in DIAG_KEYS}
    for s in seeds:
        jp = _result_path(featurizer, s)
        if not jp.exists():
            continue
        with open(jp) as f:
            payload = json.load(f)
        metrics = payload["metrics"]
        for sn in EVAL_SPLITS:
            if sn not in metrics:
                continue
            for mk, mv in metrics[sn].items():
                if isinstance(mv, (int, float)) and not (isinstance(mv, float) and np.isnan(mv)):
                    summary["by_split"][sn].setdefault(mk, []).append(mv)
        for k in DIAG_KEYS:
            v = metrics.get(k)
            if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                diag[k].append(v)

    agg = {sn: {} for sn in EVAL_SPLITS}
    for sn in EVAL_SPLITS:
        for mk, vals in summary["by_split"][sn].items():
            if vals:
                agg[sn][f"{mk}_mean"] = float(np.mean(vals))
                agg[sn][f"{mk}_std"] = float(np.std(vals))
                agg[sn][f"{mk}_n"] = len(vals)
    summary["aggregated"] = agg
    summary["diagnostics"] = {k: float(np.mean(vs)) for k, vs in diag.items() if vs}
    summary.pop("by_split", None)

    sp = feat_dir / "summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Representation ablation: fix LightGBM, vary featurizer.",
    )
    parser.add_argument("--featurizers", nargs="+", default=DEFAULT_FEATURIZERS,
                        choices=DEFAULT_FEATURIZERS,
                        help="Featurizers to evaluate (default: all 7).")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Random seeds (default: [42]; use 5 for publication).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run runs whose JSON already exists.")
    parser.add_argument("--hp-key", default=FIXED_MODEL_HP_KEY,
                        help=f"Method key whose HPs to use (default: {FIXED_MODEL_HP_KEY}).")
    args = parser.parse_args()

    _log(f"CPU cap: using {_N_JOBS}/{_N_CPUS_TOTAL} cores (60%)")
    _log(f"Featurizers: {args.featurizers}")
    _log(f"Seeds:       {args.seeds}")
    _log(f"Fixed model: LightGBM with HPs from `{args.hp_key}`")

    splits = load_all_splits(verbose=True)
    params = get_hp(args.hp_key)
    _log(f"HPs: {params}")

    # Preflight: check every featurizer's cache exists
    missing = []
    for f in args.featurizers:
        if load_cached_features(f) is None:
            missing.append(f)
    if missing:
        _log(f"ERROR: missing feature caches: {missing}")
        _log(f"Build them with: python sc3 cache --featurizers {' '.join(missing)}")
        sys.exit(1)

    grid = [(f, s) for f in args.featurizers for s in args.seeds]
    n_total = len(grid)
    n_skipped = sum(1 for (f, s) in grid
                    if _result_path(f, s).exists() and not args.force)
    n_to_run = n_total - n_skipped
    _log(f"Run grid: {n_total} total ({n_skipped} already done, {n_to_run} to run)")

    grand_t0 = time.time()
    overall_pbar = tqdm(total=n_total, desc="overall", ncols=100, position=0,
                        initial=n_skipped)

    last_feat = None
    for (featurizer, seed) in grid:
        out_path = _result_path(featurizer, seed)
        if out_path.exists() and not args.force:
            continue

        if featurizer != last_feat:
            _log(f"\n{'='*72}\n  FEATURIZER: {featurizer}\n{'='*72}")
            last_feat = featurizer

        _log(f"\n--- {featurizer}  seed={seed}  ({overall_pbar.n+1}/{n_total}) ---")
        t0 = time.time()
        try:
            metrics = _train_lgb_on_featurizer(featurizer, seed, splits, params)
        except Exception as e:
            _log(f"[ERROR] {featurizer} seed={seed}: {e}")
            import traceback; traceback.print_exc()
            overall_pbar.update(1)
            continue
        dt = time.time() - t0

        payload = {
            "featurizer": featurizer,
            "model": "lightgbm",
            "hp_key": args.hp_key,
            "seed": seed,
            "params": params,
            "wall_time_s": dt,
            "metrics": metrics,
        }
        _save_run(out_path, payload)
        _aggregate_featurizer(featurizer, args.seeds)

        overall_pbar.update(1)
        elapsed = time.time() - grand_t0
        done_now = overall_pbar.n - n_skipped
        if done_now > 0 and overall_pbar.n < n_total:
            eta = elapsed / done_now * (n_total - overall_pbar.n)
            _log(f"  elapsed={elapsed/60:.1f}min  ETA={eta/60:.1f}min")

    overall_pbar.close()
    _log(f"\nDone in {(time.time()-grand_t0)/60:.1f} min.  Results under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
