"""
Build RDKit feature cache for CombiSolv pretraining datasets.

Uses the same `RDKitFeaturizer` and `build_features()` pipeline as the
main `sc3_bench` benchmark, so the pretrained trunk and the fine-tuned
trunk see *exactly* the same feature ordering / scaling.

CombiSolv data lives at
`Solubility/sc3-benchmark/Additional_Experiments/transfer_v2/data/`
and was already pair-level cleaned against the (older) SC3 holdouts.
We additionally drop any rows whose canonical (solute, solvent) pair
appears in the *current* SC3 holdouts (eval / ood / sc3-tiers), so the
resulting cache is fully leakage-free against the v2 splits.

Output (cached):
  cache/combisolv_qm.npz     (X, y)
  cache/combisolv_exp.npz    (X, y)

Both X are float32 (n_rows, 320) — 158 RDKit descriptors x 2 + 4
temperature features, exactly as in the SC3 `rdkit.npz` cache.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
VANSH_ROOT = ABLATIONS_TRANSFER_DIR.parent.parent
SOLUBILITY_ROOT = VANSH_ROOT.parent

sys.path.insert(0, str(VANSH_ROOT))
from sc3_bench.featurizers import get_featurizer, build_features  # noqa: E402

CACHE_DIR = ABLATIONS_TRANSFER_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEMP_K = 298.15

COMBISOLV_QM_SRC = (
    SOLUBILITY_ROOT
    / "sc3-benchmark"
    / "Additional_Experiments"
    / "transfer_v2"
    / "data"
    / "CombiSolv-QM-clean.csv"
)
COMBISOLV_EXP_SRC = (
    SOLUBILITY_ROOT
    / "sc3-benchmark"
    / "Additional_Experiments"
    / "transfer_v2"
    / "data"
    / "CombiSolv-Exp-clean.csv"
)


def _canon_smi(s: str, cache: dict) -> str | None:
    if s in cache:
        return cache[s]
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        cache[s] = None
        return None
    cache[s] = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return cache[s]


def _load_holdout_pairs() -> set[tuple[str, str]]:
    """Canonical (solute, solvent) pairs to drop from pretraining."""
    sc3_splits_dir = SOLUBILITY_ROOT / "sc3_benchmark_data_curation_v2" / "data" / "splits"
    sc3_tiers_dir  = SOLUBILITY_ROOT / "sc3_benchmark_data_curation_v2" / "data" / "sc3"
    paths = [
        sc3_splits_dir / "bench_eval.csv",
        sc3_splits_dir / "bench_ood.csv",
        sc3_tiers_dir / "gold.csv",
        sc3_tiers_dir / "silver.csv",
        sc3_tiers_dir / "bronze.csv",
    ]
    cache: dict = {}
    pairs: set[tuple[str, str]] = set()
    for p in paths:
        df = pd.read_csv(p)
        for sol, solv in zip(df["Solute_Canon"].values, df["Solvent_Canon"].values):
            sol_c  = _canon_smi(str(sol),  cache)
            solv_c = _canon_smi(str(solv), cache)
            if sol_c is None or solv_c is None:
                continue
            pairs.add((sol_c, solv_c))
    return pairs


def _filter_combisolv(df: pd.DataFrame, holdout: set[tuple[str, str]]) -> pd.DataFrame:
    """Drop rows whose canonical (Solute, Solvent) pair is in `holdout`."""
    cache: dict = {}
    keep = np.ones(len(df), dtype=bool)
    for i, (sol, solv) in enumerate(zip(df["Solute"].values, df["Solvent"].values)):
        sol_c  = _canon_smi(str(sol),  cache)
        solv_c = _canon_smi(str(solv), cache)
        if sol_c is None or solv_c is None:
            continue
        if (sol_c, solv_c) in holdout:
            keep[i] = False
    return df[keep].reset_index(drop=True)


def _load_combisolv(path: Path) -> pd.DataFrame:
    """Load a CombiSolv csv and rename to the SC3 schema.

    Columns produced:
      Solute, Solvent, Temperature (constant 298.15 K), LogS = ΔG_solv
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"dgsolv": "y"})
    out = pd.DataFrame({
        "Solute":      df["Solute"].astype(str),
        "Solvent":     df["Solvent"].astype(str),
        "Temperature": np.full(len(df), DEFAULT_TEMP_K, dtype=np.float32),
        "LogS":        df["y"].astype(np.float32),
    })
    return out


def _featurise_and_cache(name: str, df: pd.DataFrame, force: bool = False) -> Path:
    out_file = CACHE_DIR / f"combisolv_{name}.npz"
    if out_file.exists() and not force:
        print(f"[cache] {name}: cache exists, skipping ({out_file})")
        return out_file

    print(f"[cache] {name}: featurizing {len(df):,} rows ...")
    feat = get_featurizer("rdkit")
    cache: dict = {}
    t0 = time.time()
    X = build_features(df, feat, cache)
    y = df["LogS"].values.astype(np.float32)
    print(f"[cache] {name}: built X={X.shape}  in {time.time()-t0:.1f}s")
    np.savez_compressed(out_file, X=X, y=y)
    print(f"[cache] {name}: saved {out_file}  ({out_file.stat().st_size/1e6:.1f} MB)")
    return out_file


def main(force: bool = False, only: str | None = None):
    print("[cache] Building holdout pair set ...")
    holdout = _load_holdout_pairs()
    print(f"[cache]   {len(holdout):,} canonical holdout pairs")

    if only in (None, "exp"):
        df = _load_combisolv(COMBISOLV_EXP_SRC)
        n0 = len(df)
        df = _filter_combisolv(df, holdout)
        print(f"[cache] CombiSolv-Exp: kept {len(df):,} / {n0:,} rows after pair filter")
        _featurise_and_cache("exp", df, force=force)

    if only in (None, "qm"):
        df = _load_combisolv(COMBISOLV_QM_SRC)
        n0 = len(df)
        df = _filter_combisolv(df, holdout)
        print(f"[cache] CombiSolv-QM: kept {len(df):,} / {n0:,} rows after pair filter")
        _featurise_and_cache("qm", df, force=force)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Re-featurize even if the cache file exists.")
    p.add_argument("--only", choices=["qm", "exp"], default=None,
                   help="Build only one of the two caches.")
    args = p.parse_args()
    main(force=args.force, only=args.only)
