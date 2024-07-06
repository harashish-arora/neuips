"""
Data split infrastructure for SC3 benchmarking.

Splits (from full clean data, after SC3 solute removal):
  - train:       top-K solvents, ~90% of solutes (all temps for each solute)
  - eval:        top-K solvents, held-out ~10% solutes (all temps), pair-disjoint from train
  - test_ood:    all rows with solvents NOT in top-K (solvent-OOD)
  - test_hard / test_medium / test_easy: SC3 consensus tiers (solute-OOD)
"""

import numpy as np
import pandas as pd
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CLEAN_DIR = DATA_DIR / "clean"
SC3_DIR = DATA_DIR / "sc3"
SPLIT_DIR = DATA_DIR / "splits"

TOP_K_SOLVENTS = 25
EVAL_FRAC = 0.10
SEED = 42


def _build_pool():
    """Reconstruct the full clean pool (train.csv + val.csv, i.e. everything minus SC3 solutes)."""
    df_train = pd.read_csv(CLEAN_DIR / "train.csv")
    df_val = pd.read_csv(CLEAN_DIR / "val.csv")
    pool = pd.concat([df_train, df_val], ignore_index=True)
    return pool


def build_splits(
    top_k: int = TOP_K_SOLVENTS,
    eval_frac: float = EVAL_FRAC,
    seed: int = SEED,
    force_rebuild: bool = False,
):
    """
    Build train / eval / OOD splits from the full clean pool.

    Strategy:
      1. Pool = train.csv + val.csv (full clean data, SC3 solutes already removed).
      2. Rank solvents by row count; top-K are the "in-distribution" solvents.
      3. OOD = all rows whose solvent is NOT in top-K.
      4. In-distribution pool = rows with top-K solvents.
      5. For each top-K solvent, hold out eval_frac of its *solutes*
         (all temperature rows for a held-out solute go to eval together).
      6. train = in-distribution rows not in eval.
      7. Guarantee: no (solute, solvent) pair appears in both train and eval.

    Returns:
        (df_train, df_eval, df_ood) DataFrames
    """
    split_dir = SPLIT_DIR
    split_dir.mkdir(parents=True, exist_ok=True)

    train_out = split_dir / "bench_train.csv"
    eval_out = split_dir / "bench_eval.csv"
    ood_out = split_dir / "bench_ood.csv"

    if not force_rebuild and train_out.exists() and eval_out.exists() and ood_out.exists():
        return pd.read_csv(train_out), pd.read_csv(eval_out), pd.read_csv(ood_out)

    pool = _build_pool()
    print(f"Building splits from pool: {len(pool)} rows, "
          f"{pool['Solute'].nunique()} solutes, {pool['Solvent_Name'].nunique()} solvents")

    rng = np.random.RandomState(seed)

    # Top-K solvents by row count
    solvent_counts = pool.groupby("Solvent_Name").size().sort_values(ascending=False)
    top_solvents = set(solvent_counts.head(top_k).index)
    print(f"  Top-{top_k} solvents: {', '.join(sorted(top_solvents))}")

    # OOD = everything outside top-K solvents
    ood_mask = ~pool["Solvent_Name"].isin(top_solvents)
    df_ood = pool[ood_mask].reset_index(drop=True)

    # In-distribution = top-K solvents only
    in_dist = pool[~ood_mask].copy()

    # Hold out eval_frac of solutes per solvent (all temps go together)
    eval_indices = []
    for solvent in top_solvents:
        solvent_df = in_dist[in_dist["Solvent_Name"] == solvent]
        solutes = solvent_df.get("Solute_Canon", df.get("Solute")).unique()
        n_eval = max(1, int(len(solutes) * eval_frac))
        eval_solutes = rng.choice(solutes, size=n_eval, replace=False)
        eval_indices.extend(solvent_df[solvent_df.get("Solute_Canon", df.get("Solute")).isin(eval_solutes)].index.tolist())

    eval_indices = set(eval_indices)
    df_eval = in_dist.loc[list(eval_indices)].reset_index(drop=True)
    df_train = in_dist.loc[~in_dist.index.isin(eval_indices)].reset_index(drop=True)

    # Verify no (solute, solvent) leakage between train and eval
    train_pairs = set(zip(df_train["Solute"], df_train["Solvent_Name"]))
    eval_pairs = set(zip(df_eval["Solute"], df_eval["Solvent_Name"]))
    overlap = train_pairs & eval_pairs
    assert len(overlap) == 0, f"Leakage: {len(overlap)} overlapping (solute, solvent) pairs"

    # Save
    df_train.to_csv(train_out, index=False)
    df_eval.to_csv(eval_out, index=False)
    df_ood.to_csv(ood_out, index=False)

    print(f"\n  Train:  {len(df_train):6d} rows, {df_train['Solute'].nunique():4d} solutes, "
          f"{df_train['Solvent_Name'].nunique():3d} solvents  (top-{top_k}, ~{100-eval_frac*100:.0f}% solutes)")
    print(f"  Eval:   {len(df_eval):6d} rows, {df_eval['Solute'].nunique():4d} solutes, "
          f"{df_eval['Solvent_Name'].nunique():3d} solvents  (top-{top_k}, ~{eval_frac*100:.0f}% solutes)")
    print(f"  OOD:    {len(df_ood):6d} rows, {df_ood['Solute'].nunique():4d} solutes, "
          f"{df_ood['Solvent_Name'].nunique():3d} solvents  (non-top-{top_k})")
    print(f"  Leakage check: PASSED (0 overlapping train/eval pairs)")
    print(f"  Saved to {split_dir}")

    return df_train, df_eval, df_ood


def load_sc3_test(tier: str = "gold"):
    """Load an SC3 test tier."""
    path = SC3_DIR / f"{tier}.csv"
    return pd.read_csv(path)


def load_all_splits():
    """Load all splits for benchmarking."""
    df_train, df_eval, df_ood = build_splits()
    splits = {
        "train": df_train,
        "eval": df_eval,
        "test_ood": df_ood,
        "test_gold": load_sc3_test("gold"),
        "test_silver": load_sc3_test("silver"),
        "test_bronze": load_sc3_test("bronze"),
    }
    print(f"\nLoaded splits:")
    for name, df in splits.items():
        n_solutes = df.get("Solute_Canon", df.get("Solute")).nunique()
        n_solvents = df["Solvent_Name"].nunique() if "Solvent_Name" in df.columns else df.get("Solvent_Canon", df.get("Solvent")).nunique()
        print(f"  {name:15s}: {len(df):6d} rows, {n_solutes:5d} solutes, {n_solvents:3d} solvents")
    return splits


if __name__ == "__main__":
    splits = load_all_splits()
