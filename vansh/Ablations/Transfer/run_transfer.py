"""
Driver for the Transfer-Learning ablation (Q3).

Runs:
  Phase 1 — Pretrain a FastProp trunk on CombiSolv-QM ΔG_solv (~1 M rows).
  Phase 2 — Fine-tune the trunk on a fraction of SC3 train, with three
            variants (full / head_only / last_two), against a from-scratch
            baseline that uses the *same* training pipeline.

Resumable: existing per-run JSONs are skipped unless `--force`.

Output layout:
  results/<protocol>__<variant>/frac_<f>/seed_<s>.json
  results/transfer_table.csv      (long-format, regenerated each run)
  results/summary.json            (mean ± std across seeds)
  models/pretrained_qm.pt         (the QM-pretrained trunk; reused across runs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
VANSH_ROOT = ABLATIONS_TRANSFER_DIR.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.data import load_all_splits, load_cached_features  # noqa: E402

from transfer_trainers import (    # noqa: E402
    pretrain_combisolv, finetune_on_sc3,
    PRETRAIN_EPOCHS, PRETRAIN_BATCH, PRETRAIN_LR, PRETRAIN_PATIENCE,
)

# --- defaults ---------------------------------------------------------------

DEFAULT_PROTOCOLS = ["scratch", "qm"]
DEFAULT_VARIANTS  = ["full", "head_only"]
DEFAULT_FRACTIONS = [0.05, 0.25, 1.0]
DEFAULT_SEEDS     = [42, 101, 123]
DEFAULT_EVAL_SPLITS = ("eval", "ood", "sc3_gold")

CACHE_DIR  = ABLATIONS_TRANSFER_DIR / "cache"
MODEL_DIR  = ABLATIONS_TRANSFER_DIR / "models"
RESULTS_DIR = ABLATIONS_TRANSFER_DIR / "results"
LOG_DIR    = ABLATIONS_TRANSFER_DIR / "logs"
for d in (CACHE_DIR, MODEL_DIR, RESULTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- helpers ----------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _result_path(protocol: str, variant: str, fraction: float, seed: int) -> Path:
    return RESULTS_DIR / f"{protocol}__{variant}" / f"frac_{fraction:g}" / f"seed_{seed}.json"


def _ensure_caches(force_qm_cache: bool = False, force_exp_cache: bool = False) -> None:
    """Build CombiSolv RDKit caches if missing."""
    qm_cache  = CACHE_DIR / "combisolv_qm.npz"
    exp_cache = CACHE_DIR / "combisolv_exp.npz"
    need = (force_qm_cache or not qm_cache.exists()) or \
           (force_exp_cache or not exp_cache.exists())
    if need:
        from cache_combisolv_features import main as build_caches
        only = None
        if (qm_cache.exists() and not force_qm_cache):
            only = "exp"
        elif (exp_cache.exists() and not force_exp_cache):
            only = "qm"
        build_caches(force=(force_qm_cache or force_exp_cache), only=only)


def _ensure_pretrained_qm(seed: int, device, force: bool = False) -> Path:
    """Pretrain (or reload) FastProp on CombiSolv-QM."""
    ckpt = MODEL_DIR / f"pretrained_qm_seed{seed}.pt"
    if ckpt.exists() and not force:
        _log(f"pretrain[qm/seed={seed}]: cached at {ckpt.name}, skipping")
        return ckpt

    cache_path = CACHE_DIR / "combisolv_qm.npz"
    if not cache_path.exists():
        _ensure_caches()

    out = pretrain_combisolv(cache_path, seed=seed, device=device)
    torch.save(out, ckpt)
    _log(f"pretrain[qm/seed={seed}]: saved {ckpt.name}  "
         f"val_RMSE(dGsolv)={out['val_RMSE']:.4f} kcal/mol  "
         f"({ckpt.stat().st_size/1e6:.1f} MB)")
    return ckpt


def _load_pretrained(ckpt: Path, device) -> dict:
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    obj["state_dict"] = {k: v.to(device) for k, v in obj["state_dict"].items()}
    return obj


# --- table writer ----------------------------------------------------------

def _flatten_metrics(metrics: dict) -> dict:
    """Turn the nested per-split metrics dict into a flat row."""
    row: dict[str, object] = {}
    for k, v in metrics.items():
        if k.startswith("_"):
            row[k.lstrip("_")] = v
        elif isinstance(v, dict):
            for mk, mv in v.items():
                row[f"{k}__{mk}"] = mv
    return row


def _rebuild_table(write_csv: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for json_path in sorted(RESULTS_DIR.rglob("seed_*.json")):
        with open(json_path) as f:
            d = json.load(f)
        rows.append(_flatten_metrics(d))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if write_csv:
        df.to_csv(RESULTS_DIR / "transfer_table.csv", index=False)
    return df


def _aggregate_summary(df: pd.DataFrame, write: bool = True) -> dict:
    if df.empty:
        return {}
    summary: dict = {}
    metric_cols = [c for c in df.columns if "__" in c
                   and not c.startswith("_")]  # eval__RMSE, ood__PS_RMSE, ...
    grp_keys = ["protocol", "variant", "fraction"]
    for keys, g in df.groupby(grp_keys):
        proto, var, frac = keys
        d = summary.setdefault(proto, {}).setdefault(var, {})
        d[f"frac={frac:g}"] = {
            "n_seeds": int(len(g)),
            "n_train_mean": float(g["n_train"].mean()) if "n_train" in g.columns else None,
            "metrics": {
                m: {"mean": float(g[m].mean()), "std": float(g[m].std(ddof=0))}
                for m in metric_cols
                if g[m].notna().any()
            }
        }
    if write:
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(
    *,
    protocols:  list[str] = None,
    variants:   list[str] = None,
    fractions:  list[float] = None,
    seeds:      list[int]  = None,
    eval_splits: tuple[str, ...] = DEFAULT_EVAL_SPLITS,
    force: bool = False,
    pretrain_only: bool = False,
    gpu: Optional[int] = None,
) -> None:
    protocols = protocols or DEFAULT_PROTOCOLS
    variants  = variants  or DEFAULT_VARIANTS
    fractions = fractions or DEFAULT_FRACTIONS
    seeds     = seeds     or DEFAULT_SEEDS

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"device={device}  CUDA available={torch.cuda.is_available()}  "
         f"({torch.cuda.device_count()} devices visible)")

    # Make sure feature caches exist for both pretraining and SC3.
    _log("Loading SC3 splits and rdkit feature cache ...")
    splits = load_all_splits(verbose=False)
    cached_sc3 = load_cached_features("rdkit")
    if cached_sc3 is None:
        raise FileNotFoundError(
            "feature_cache/rdkit.npz missing.  Run `python sc3 cache "
            "--featurizers rdkit` from vansh/.")
    _ensure_caches()

    # Phase 1: pretrain (one model per seed; reused across all variants/fractions)
    _log("=" * 70)
    _log("PHASE 1: PRETRAINING ON CombiSolv-QM")
    _log("=" * 70)
    pretrained_paths: dict[int, Path] = {}
    needs_pretrain = "qm" in protocols
    if needs_pretrain or pretrain_only:
        for seed in seeds:
            pretrained_paths[seed] = _ensure_pretrained_qm(seed, device, force=force and pretrain_only)
    if pretrain_only:
        _log("Pretrain-only mode: stopping after Phase 1.")
        return

    # Phase 2: full grid
    _log("=" * 70)
    _log("PHASE 2: FINE-TUNING ON SC3")
    _log("=" * 70)
    total = len(protocols) * len(variants) * len(fractions) * len(seeds)
    run_idx = 0
    new_runs = 0

    pretrained_cache: dict[int, dict] = {}  # seed -> loaded pretrained dict

    for protocol in protocols:
        for variant in variants:
            # `head_only` makes no sense from a random init: it freezes the
            # trunk so only the final 1-d linear is trainable.  We still run
            # it for symmetry — it's a useful sanity check (RMSE should be
            # very large for scratch/head_only, near-baseline for qm/head_only).
            for fraction in fractions:
                for seed in seeds:
                    run_idx += 1
                    out_json = _result_path(protocol, variant, fraction, seed)
                    if out_json.exists() and not force:
                        _log(f"[{run_idx}/{total}] SKIP {out_json.relative_to(ABLATIONS_TRANSFER_DIR)}")
                        continue

                    _log(f"[{run_idx}/{total}] proto={protocol} variant={variant} "
                         f"fraction={fraction} seed={seed}")

                    pretrained = None
                    if protocol == "qm":
                        if seed not in pretrained_cache:
                            pretrained_cache[seed] = _load_pretrained(pretrained_paths[seed], device)
                        pretrained = pretrained_cache[seed]

                    metrics = finetune_on_sc3(
                        protocol=protocol, variant=variant,
                        fraction=fraction, seed=seed,
                        cached_sc3=cached_sc3, splits=splits,
                        device=device, pretrained=pretrained,
                        eval_splits=eval_splits,
                    )

                    out_json.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_json, "w") as f:
                        json.dump(metrics, f, indent=2, default=str)
                    new_runs += 1

                    # Periodic table refresh
                    if new_runs % 4 == 0:
                        df = _rebuild_table(write_csv=True)
                        _aggregate_summary(df, write=True)

    # Final
    df = _rebuild_table(write_csv=True)
    _aggregate_summary(df, write=True)
    _log("=" * 70)
    _log(f"Done.  {new_runs} new run(s); table at {RESULTS_DIR/'transfer_table.csv'}")
    _log("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Q3 Transfer-Learning ablation driver")
    p.add_argument("--protocols", nargs="+", default=None,
                   choices=["scratch", "qm"])
    p.add_argument("--variants",  nargs="+", default=None,
                   choices=["full", "head_only", "last_two"])
    p.add_argument("--fractions", nargs="+", type=float, default=None)
    p.add_argument("--seeds",     nargs="+", type=int,   default=None)
    p.add_argument("--eval-splits", nargs="+", default=list(DEFAULT_EVAL_SPLITS))
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if per-run JSON exists.")
    p.add_argument("--pretrain-only", action="store_true",
                   help="Pretrain on CombiSolv-QM and exit (no fine-tuning).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        protocols=args.protocols,
        variants=args.variants,
        fractions=args.fractions,
        seeds=args.seeds,
        eval_splits=tuple(args.eval_splits),
        gpu=args.gpu,
        force=args.force,
        pretrain_only=args.pretrain_only,
    )
