"""
Data loading for SC3 benchmark.

Loads train/eval/OOD/SC3-tier splits from the curation v2 data directory
and standardizes column names. Also handles precomputed feature caching.

Set SC3_DATA_DIR env var to override the default data path.
"""

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "sc3_benchmark_data_curation_v2" / "data"
DATA_DIR = Path(os.environ.get("SC3_DATA_DIR", str(_DEFAULT_DATA_DIR)))
CACHE_DIR = Path(__file__).resolve().parent.parent / "feature_cache"


def _standardize_bench(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Solute_Canon": "Solute", "Temperature_K": "Temperature"})
    if "Solvent" in df.columns and "Solvent_Canon" in df.columns:
        df = df.rename(columns={"Solvent": "Solvent_Name", "Solvent_Canon": "Solvent"})
    elif "Solvent_Canon" in df.columns:
        df = df.rename(columns={"Solvent_Canon": "Solvent"})
    return df


def _standardize_sc3(df: pd.DataFrame, solvent_map: dict) -> pd.DataFrame:
    df = df.rename(columns={
        "Solute_Canon": "Solute", "Solvent_Canon": "Solvent",
        "Temperature_K": "Temperature", "LogS_consensus": "LogS", "sigma": "Uncertainty",
    })
    df["Solvent_Name"] = df["Solvent"].map(solvent_map).fillna("Unknown")
    return df


def load_all_splits(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Load all benchmark splits with standardized column names."""
    train = _standardize_bench(pd.read_csv(DATA_DIR / "splits" / "bench_train.csv"))
    eval_ = _standardize_bench(pd.read_csv(DATA_DIR / "splits" / "bench_eval.csv"))
    ood   = _standardize_bench(pd.read_csv(DATA_DIR / "splits" / "bench_ood.csv"))

    solvent_map = {}
    for df in [train, eval_, ood]:
        if "Solvent_Name" in df.columns and "Solvent" in df.columns:
            for _, row in df[["Solvent", "Solvent_Name"]].drop_duplicates().iterrows():
                solvent_map[row["Solvent"]] = row["Solvent_Name"]

    gold   = _standardize_sc3(pd.read_csv(DATA_DIR / "sc3" / "gold.csv"),   solvent_map)
    silver = _standardize_sc3(pd.read_csv(DATA_DIR / "sc3" / "silver.csv"), solvent_map)
    bronze = _standardize_sc3(pd.read_csv(DATA_DIR / "sc3" / "bronze.csv"), solvent_map)

    splits = {
        "train": train, "eval": eval_, "ood": ood,
        "sc3_gold": gold, "sc3_silver": silver, "sc3_bronze": bronze,
    }
    if verbose:
        print("Loaded splits:")
        for name, df in splits.items():
            n_sol = df["Solute"].nunique()
            n_solv = df["Solvent_Name"].nunique() if "Solvent_Name" in df.columns else df["Solvent"].nunique()
            print(f"  {name:15s}: {len(df):6d} rows, {n_sol:5d} solutes, {n_solv:3d} solvents")
    return splits


def load_cached_features(feat_name: str) -> dict | None:
    """Load precomputed feature arrays from disk cache."""
    cache_file = CACHE_DIR / f"{feat_name}.npz"
    if not cache_file.exists():
        return None
    data = np.load(cache_file)
    return {k: data[k] for k in data.files}


def precompute_features(feat_names: list[str] | None = None):
    """Precompute and cache all featurized matrices to disk."""
    from .featurizers import get_featurizer, build_features

    if feat_names is None:
        feat_names = ["rdkit", "morgan", "dissolvr", "mordred",
                      "maccs", "atompair", "abraham_only"]

    CACHE_DIR.mkdir(exist_ok=True)
    splits = load_all_splits()
    split_names = list(splits.keys())

    for fname in feat_names:
        out_file = CACHE_DIR / f"{fname}.npz"
        if out_file.exists():
            print(f"[{fname}] Already cached, skipping.")
            continue
        print(f"Featurizing: {fname} ...")
        feat = get_featurizer(fname)
        cache = {}
        arrays = {}
        t0 = time.time()
        for sname in split_names:
            df = splits[sname]
            X = build_features(df, feat, cache)
            arrays[f"X_{sname}"] = X
            arrays[f"y_{sname}"] = df["LogS"].values.astype(np.float32)
        np.savez_compressed(out_file, **arrays)
        print(f"  Saved {out_file} ({time.time()-t0:.1f}s)")
    print("All features cached.")
