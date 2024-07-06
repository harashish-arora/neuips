"""
Data split infrastructure for SC3 benchmarking.

Loads precomputed CSV splits from the curation database (not built at runtime).

Set SC3_DATA_DIR to the curation ``data`` folder if the default resolution fails
(e.g. after moving this project under a subfolder like ``Dhairya/``).
"""

import os

import pandas as pd
from pathlib import Path


def _resolve_data_dir() -> Path:
    """Find ``.../sc3_benchmark_data_curation_v2/data`` (sibling of bundle or grandparent)."""
    env = os.environ.get("SC3_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root.parent / "sc3_benchmark_data_curation_v2" / "data",
        project_root.parent.parent / "sc3_benchmark_data_curation_v2" / "data",
    ]
    for c in candidates:
        if (c / "splits" / "bench_train.csv").is_file():
            return c.resolve()
    return candidates[0].resolve()


DATA_DIR = _resolve_data_dir()
SPLIT_DIR = DATA_DIR / "splits"
SC3_DIR = DATA_DIR / "sc3"


def _normalize_split_columns(df):
    """Map split CSV columns to benchmark-standard names used by models."""
    out = df.copy()
    # Preserve human-readable solvent names before canonical solvent mapping.
    if "Solvent" in out.columns and "Solvent_Canon" in out.columns:
        out = out.rename(columns={"Solvent": "Solvent_Name"})

    rename_map = {
        "Solute_Canon": "Solute",
        "Solvent_Canon": "Solvent",
        "Temperature_K": "Temperature",
    }
    out = out.rename(columns=rename_map).copy()
    if "Solvent_Name" not in out.columns:
        out["Solvent_Name"] = out["Solvent"]
    return out


def _normalize_sc3_columns(df):
    """Map SC3 tier CSV columns to benchmark-standard names used by models."""
    rename_map = {
        "Solute_Canon": "Solute",
        "Solvent_Canon": "Solvent",
        "Temperature_K": "Temperature",
        "LogS_consensus": "LogS",
        "sigma": "Uncertainty",
    }
    out = df.rename(columns=rename_map).copy()
    if "Solvent_Name" not in out.columns:
        out["Solvent_Name"] = out["Solvent"]
    return out


def build_splits():
    """Load precomputed benchmark splits from external curation database."""
    train_out = SPLIT_DIR / "bench_train.csv"
    eval_out = SPLIT_DIR / "bench_eval.csv"
    ood_out = SPLIT_DIR / "bench_ood.csv"
    for path in [train_out, eval_out, ood_out]:
        if not path.exists():
            raise FileNotFoundError(f"Expected split file not found: {path}")
    df_train = _normalize_split_columns(pd.read_csv(train_out))
    df_eval = _normalize_split_columns(pd.read_csv(eval_out))
    df_ood = _normalize_split_columns(pd.read_csv(ood_out))
    return df_train, df_eval, df_ood


def load_sc3_test(tier: str = "hard"):
    """Load an SC3 tier from curation DB (hard->gold, medium->silver, easy->bronze)."""
    tier_map = {"hard": "gold", "medium": "silver", "easy": "bronze"}
    if tier not in tier_map:
        raise ValueError(f"Unknown tier {tier}. Choose from {list(tier_map.keys())}")
    path = SC3_DIR / f"{tier_map[tier]}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected SC3 tier file not found: {path}")
    return _normalize_sc3_columns(pd.read_csv(path))


def load_all_splits():
    """Load all splits for benchmarking."""
    df_train, df_eval, df_ood = build_splits()
    splits = {
        "train": df_train,
        "eval": df_eval,
        "test_ood": df_ood,
        "test_hard": load_sc3_test("hard"),
        "test_medium": load_sc3_test("medium"),
        "test_easy": load_sc3_test("easy"),
    }
    print(f"\nLoaded splits:")
    for name, df in splits.items():
        n_solutes = df["Solute"].nunique()
        n_solvents = df["Solvent_Name"].nunique() if "Solvent_Name" in df.columns else df["Solvent"].nunique()
        print(f"  {name:15s}: {len(df):6d} rows, {n_solutes:5d} solutes, {n_solvents:3d} solvents")
    return splits


if __name__ == "__main__":
    splits = load_all_splits()
