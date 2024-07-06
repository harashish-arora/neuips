#!/usr/bin/env python
"""
Interpretability ablation - LightGBM SHAP driver  (Q5: what does the model learn?)

For each of the 7 featurizers studied in the Representation ablation we:

  1. Train a LightGBM with the tuned ``lgb_rdkit`` HPs (held constant, matching
     the Representation ablation; resumable from cached models).
  2. Compute exact Tree-SHAP values on the *eval* and *ood* splits (full).
  3. Recover human-readable feature names by running each featurizer's
     ``transform_single`` on one SMILES (the cached .npz drops column names).
  4. Save:
       - results/<feat>/model_seed_<seed>.pkl              the trained LGBM
       - results/<feat>/shap_eval.npz, shap_ood.npz        SHAP arrays + names
       - results/<feat>/feature_names.json                  full ordered names
       - results/<feat>/metrics.json                        sanity-check RMSE

A separate ``analyze_shap.py`` consumes these per-featurizer dumps.

Featurizers (same set as the Representation ablation):
    rdkit, morgan, dissolvr, mordred, maccs, atompair, abraham_only

Usage
-----
    # Smoke test (rdkit only)
    python run_shap.py --featurizers rdkit

    # Full sweep, single seed=42 (default), resumable
    python run_shap.py

    # Force retraining + re-SHAP
    python run_shap.py --force

We also write a brief sanity-check .json with eval/ood RMSE so we know the
trained model matches the Representation ablation numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Cap CPU usage to ~60% of cores.  Must be set BEFORE numpy / lightgbm import.
_N_CPUS_TOTAL = os.cpu_count() or 16
_N_JOBS = max(1, int(round(_N_CPUS_TOTAL * 0.60)))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_N_JOBS))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
VANSH_ROOT = HERE.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.data import load_all_splits, load_cached_features  # noqa: E402
from sc3_bench.evaluate import compute_metrics  # noqa: E402
from sc3_bench.registry import get_hp  # noqa: E402
from sc3_bench.featurizers import get_featurizer  # noqa: E402

RESULTS_DIR = HERE / "results"

DEFAULT_FEATURIZERS = [
    "rdkit", "morgan", "dissolvr", "mordred",
    "maccs", "atompair", "abraham_only",
]
DEFAULT_SEED = 42
SHAP_SPLITS = ["eval", "ood"]

# All featurizers use the same fixed model + HPs (matches Representation).
FIXED_MODEL_HP_KEY = "lgb_rdkit"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Feature-name recovery
# ---------------------------------------------------------------------------
#
# The cached .npz files only store the numerical arrays, not the column names.
# To make SHAP interpretable we instantiate each featurizer and call
# ``transform_single`` once on a representative SMILES, then build the full
# concatenated name list:  ``[solute_<f>..., solv_<f>..., T_norm, T_inv,
#                              T_sq, T_log]``.

_PROBE_SMILES = "CCO"  # ethanol; small enough that even Mordred is fast

def _featurizer_feature_names(feat_name: str) -> list[str]:
    """Reconstruct the full feature name list used by build_features()."""
    feat = get_featurizer(feat_name)
    # Trigger discovery for featurizers that build feature_names lazily
    # (Dissolvr, Mordred).
    out = feat.transform_single(_PROBE_SMILES)
    if isinstance(out, dict):
        names = list(out.keys())
        if hasattr(feat, "feature_names") and feat.feature_names:
            names = list(feat.feature_names)
        else:
            feat.feature_names = names
    else:
        names = list(feat.feature_names)
    solute_names  = [f"solute_{n}" for n in names]
    solvent_names = [f"solv_{n}"   for n in names]
    temp_names    = ["T_norm", "T_inv", "T_sq", "T_log"]
    return solute_names + solvent_names + temp_names


# ---------------------------------------------------------------------------
# Model train / load
# ---------------------------------------------------------------------------

def _train_or_load_lgb(featurizer: str, seed: int, splits: dict,
                       params: dict, force: bool = False):
    """Return (model, n_features, train_dt_s, best_iter, model_path)."""
    from lightgbm import LGBMRegressor
    import lightgbm as lgb

    out_dir = RESULTS_DIR / featurizer
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_seed_{seed}.pkl"

    cached = load_cached_features(featurizer)
    if cached is None:
        raise FileNotFoundError(
            f"feature_cache/{featurizer}.npz not found. Build with `python sc3 cache --featurizers {featurizer}`."
        )

    X_tr, y_tr = cached["X_train"], cached["y_train"]
    X_ev, y_ev = cached["X_eval"],  cached["y_eval"]
    n_features = int(X_tr.shape[1])

    if model_path.exists() and not force:
        _log(f"  loading cached model: {model_path.name}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, n_features, 0.0, getattr(model, "best_iteration_", None), model_path

    _log(f"  features: n_features={n_features}  X_train={X_tr.shape}  X_eval={X_ev.shape}")
    n_estimators = int(params.get("n_estimators", 3000))
    model = LGBMRegressor(random_state=seed, n_jobs=_N_JOBS, verbose=-1, **params)
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_ev, y_ev)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    train_dt = time.time() - t0
    best_iter = getattr(model, "best_iteration_", None)
    _log(f"  trained in {train_dt:.1f}s  best_iter={best_iter}")

    with open(model_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    return model, n_features, float(train_dt), best_iter, model_path


# ---------------------------------------------------------------------------
# Sanity-check metrics
# ---------------------------------------------------------------------------

def _eval_split(model, splits, featurizer, sname, cached) -> dict:
    Xs = cached.get(f"X_{sname}")
    if Xs is None:
        return {}
    df = splits[sname]
    preds = model.predict(Xs)
    return compute_metrics(
        df["LogS"].values, preds,
        df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
        df["Uncertainty"].values  if "Uncertainty"  in df.columns else None,
    )


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def _compute_shap(model, X: np.ndarray, label: str) -> dict:
    """Compute exact Tree-SHAP values for the LightGBM model on X.

    Returns:
        shap_values : (N, F) float32
        base_value  : float (E[f(X)] of the explainer)
    """
    import shap
    _log(f"    SHAP[{label}]: building TreeExplainer for {len(X)} rows...")
    t0 = time.time()
    expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = expl.shap_values(X, check_additivity=False)
    base = float(np.atleast_1d(expl.expected_value)[0])
    dt = time.time() - t0
    _log(f"    SHAP[{label}]: {sv.shape}  base={base:+.4f}  ({dt:.1f}s)")
    return {"shap": sv.astype(np.float32), "base_value": base, "wall_s": float(dt)}


def _save_shap_dump(out_dir: Path, sname: str, shap_arr: np.ndarray,
                    base_value: float, splits: dict) -> Path:
    """Save SHAP + a few alignment columns so analyze_shap.py is self-contained."""
    df = splits[sname]
    sp = out_dir / f"shap_{sname}.npz"
    np.savez_compressed(
        sp,
        shap=shap_arr,
        base_value=np.array([base_value], dtype=np.float32),
        y_true=df["LogS"].values.astype(np.float32),
        solvent_names=df["Solvent_Name"].values.astype(object) if "Solvent_Name" in df.columns else np.array([], dtype=object),
        solute_smiles=df["Solute"].values.astype(object),
        solvent_smiles=df["Solvent"].values.astype(object),
        temperature=df["Temperature"].values.astype(np.float32),
    )
    return sp


# ---------------------------------------------------------------------------
# Main per-featurizer driver
# ---------------------------------------------------------------------------

def _run_featurizer(featurizer: str, seed: int, splits: dict,
                    params: dict, force: bool):
    out_dir = RESULTS_DIR / featurizer
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"\n{'='*72}\n  FEATURIZER: {featurizer}\n{'='*72}")

    cached = load_cached_features(featurizer)
    n_features_cached = int(cached["X_train"].shape[1])

    # 1. Recover human-readable feature names.
    fnames_path = out_dir / "feature_names.json"
    if fnames_path.exists() and not force:
        with open(fnames_path) as f:
            feature_names = json.load(f)
        _log(f"  loaded {len(feature_names)} feature names from cache")
    else:
        _log(f"  recovering feature names for `{featurizer}`...")
        feature_names = _featurizer_feature_names(featurizer)
        with open(fnames_path, "w") as f:
            json.dump(feature_names, f)
        _log(f"  saved {len(feature_names)} feature names")

    if len(feature_names) != n_features_cached:
        _log(
            f"  WARN: name count {len(feature_names)} != cached n_features {n_features_cached}; "
            "padding/truncating with generic names so SHAP indices stay aligned."
        )
        if len(feature_names) < n_features_cached:
            feature_names = feature_names + [f"unk_{i}" for i in range(n_features_cached - len(feature_names))]
        else:
            feature_names = feature_names[:n_features_cached]
        with open(fnames_path, "w") as f:
            json.dump(feature_names, f)

    # 2. Train (or load) LightGBM.
    model, n_features, train_dt, best_iter, model_path = _train_or_load_lgb(
        featurizer, seed, splits, params, force=force,
    )

    # 3. Sanity-check metrics on eval + ood (must match Representation summary
    # within seed noise).
    metrics = {}
    for sname in SHAP_SPLITS:
        m = _eval_split(model, splits, featurizer, sname, cached)
        if m:
            metrics[sname] = m
            ps = m.get("PS_RMSE", float("nan"))
            ps_str = f"PS={ps:.4f}" if not np.isnan(ps) else "PS=  n/a"
            _log(f"  eval[{sname:6s}]  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
                 f"R2={m['R2']:.4f}  {ps_str}  N={m['N']}")

    metrics_payload = {
        "featurizer": featurizer,
        "model": "lightgbm",
        "hp_key": FIXED_MODEL_HP_KEY,
        "seed": seed,
        "params": params,
        "n_features": n_features,
        "train_time_s": train_dt,
        "best_iter": int(best_iter) if best_iter else None,
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # 4. SHAP on eval + ood.
    for sname in SHAP_SPLITS:
        sp = out_dir / f"shap_{sname}.npz"
        if sp.exists() and not force:
            _log(f"  SHAP[{sname}] already cached at {sp.name}, skipping.")
            continue
        Xs = cached.get(f"X_{sname}")
        if Xs is None:
            _log(f"  WARN: no X_{sname} in cache, skipping SHAP for {sname}")
            continue
        info = _compute_shap(model, Xs, label=sname)
        _save_shap_dump(out_dir, sname, info["shap"], info["base_value"], splits)
        _log(f"  saved SHAP -> {sp.name}")


def main():
    parser = argparse.ArgumentParser(
        description="LightGBM TreeSHAP driver for the Interpretability ablation.",
    )
    parser.add_argument("--featurizers", nargs="+", default=DEFAULT_FEATURIZERS,
                        choices=DEFAULT_FEATURIZERS,
                        help="Featurizers to run (default: all 7).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Single random seed (default: {DEFAULT_SEED}).")
    parser.add_argument("--force", action="store_true",
                        help="Re-train models and recompute SHAP even when cached.")
    parser.add_argument("--hp-key", default=FIXED_MODEL_HP_KEY,
                        help=f"Method key whose HPs to use (default: {FIXED_MODEL_HP_KEY}).")
    args = parser.parse_args()

    _log(f"CPU cap: using {_N_JOBS}/{_N_CPUS_TOTAL} cores (60%)")
    _log(f"Featurizers: {args.featurizers}")
    _log(f"Seed:        {args.seed}")
    _log(f"Fixed model: LightGBM with HPs from `{args.hp_key}`")

    splits = load_all_splits(verbose=True)
    params = get_hp(args.hp_key)
    _log(f"HPs: {params}")

    # Preflight: every featurizer's cache must exist.
    missing = [f for f in args.featurizers if load_cached_features(f) is None]
    if missing:
        _log(f"ERROR: missing feature caches: {missing}")
        _log(f"Build them with: python sc3 cache --featurizers {' '.join(missing)}")
        sys.exit(1)

    grand_t0 = time.time()
    for f in args.featurizers:
        try:
            _run_featurizer(f, args.seed, splits, params, force=args.force)
        except Exception as e:
            _log(f"[ERROR] {f}: {e}")
            import traceback; traceback.print_exc()
            continue
    _log(f"\nDone in {(time.time()-grand_t0)/60:.1f} min.  Results under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
