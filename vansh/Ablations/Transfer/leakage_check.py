"""
Leakage check between CombiSolv pretraining data and SC3 holdouts.

We define a "test pair" as the canonical (solute_smiles, solvent_smiles)
pair appearing in any of:
  - bench_eval.csv
  - bench_ood.csv
  - sc3/gold.csv
  - sc3/silver.csv
  - sc3/bronze.csv

Pretraining rows whose pair matches a test pair are dropped.  Crucially,
overlap on a single side (same solute in a different solvent, or same
solvent for a different solute) is NOT considered leakage: knowing
ΔG_solv(aspirin, ethanol) does not tell you the solubility of aspirin
in hexane, and the model has to learn the (solute, solvent) interaction.

This re-verifies the cleaning that was done in the v2 transfer
experiment (which only had access to test_hard / test_medium / test_easy
in the old SC3 schema).  The output is written to
`data/leakage_report.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
VANSH_ROOT = ABLATIONS_TRANSFER_DIR.parent.parent
SOLUBILITY_ROOT = VANSH_ROOT.parent

# Source CombiSolv files (already pair-level cleaned in transfer_v2).
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

# SC3 splits in the v2 curation
SC3_SPLITS_DIR = (
    SOLUBILITY_ROOT / "sc3_benchmark_data_curation_v2" / "data" / "splits"
)
SC3_TIERS_DIR = (
    SOLUBILITY_ROOT / "sc3_benchmark_data_curation_v2" / "data" / "sc3"
)

OUT_DATA_DIR = ABLATIONS_TRANSFER_DIR / "data"
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _canon_smi(s: str) -> str | None:
    if not isinstance(s, str) or not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _canonicalise_pairs(df: pd.DataFrame, solute_col: str, solvent_col: str) -> set[tuple[str, str]]:
    """Return the set of canonical (solute, solvent) pairs in df."""
    pairs: set[tuple[str, str]] = set()
    cache: dict[str, str | None] = {}
    for sol, solv in zip(df[solute_col].values, df[solvent_col].values):
        for s in (sol, solv):
            if s not in cache:
                cache[s] = _canon_smi(s)
        sol_c, solv_c = cache[sol], cache[solv]
        if sol_c is None or solv_c is None:
            continue
        pairs.add((sol_c, solv_c))
    return pairs


def _holdout_pairs() -> set[tuple[str, str]]:
    """Union of canonical pairs in eval / ood / sc3_gold / sc3_silver / sc3_bronze."""
    paths = [
        ("eval",  SC3_SPLITS_DIR / "bench_eval.csv",  "Solute_Canon", "Solvent_Canon"),
        ("ood",   SC3_SPLITS_DIR / "bench_ood.csv",   "Solute_Canon", "Solvent_Canon"),
        ("gold",   SC3_TIERS_DIR / "gold.csv",        "Solute_Canon", "Solvent_Canon"),
        ("silver", SC3_TIERS_DIR / "silver.csv",      "Solute_Canon", "Solvent_Canon"),
        ("bronze", SC3_TIERS_DIR / "bronze.csv",      "Solute_Canon", "Solvent_Canon"),
    ]
    union: set[tuple[str, str]] = set()
    per_split = []
    for name, p, sol_col, solv_col in paths:
        df = pd.read_csv(p)
        pairs = _canonicalise_pairs(df, sol_col, solv_col)
        per_split.append((name, len(df), len(pairs)))
        union |= pairs
    return union, per_split


def _scan_pretraining(path: Path, holdout: set[tuple[str, str]]) -> dict:
    """Count overlap between a CombiSolv file and the holdout pair set."""
    df = pd.read_csv(path)
    cache: dict[str, str | None] = {}
    n = len(df)
    overlapping_rows = 0
    overlapping_pairs: set[tuple[str, str]] = set()
    for sol, solv in zip(df["Solute"].values, df["Solvent"].values):
        for s in (sol, solv):
            if s not in cache:
                cache[s] = _canon_smi(s)
        sol_c, solv_c = cache[sol], cache[solv]
        if sol_c is None or solv_c is None:
            continue
        if (sol_c, solv_c) in holdout:
            overlapping_rows += 1
            overlapping_pairs.add((sol_c, solv_c))
    return {
        "path":      str(path),
        "n_rows":    n,
        "overlapping_rows":  overlapping_rows,
        "overlapping_pairs": len(overlapping_pairs),
        "n_unique_solutes":  df["Solute"].nunique(),
        "n_unique_solvents": df["Solvent"].nunique(),
    }


def main():
    print("[leakage] Building union of holdout (solute, solvent) pairs ...")
    holdout, per_split = _holdout_pairs()
    print(f"[leakage]   Total holdout pairs (canonical): {len(holdout):,}")
    for name, n_rows, n_pairs in per_split:
        print(f"[leakage]     {name:8s}  rows={n_rows:6d}  unique_pairs={n_pairs:6d}")

    print("\n[leakage] Scanning CombiSolv-QM ...")
    qm = _scan_pretraining(COMBISOLV_QM_SRC, holdout)
    print(f"[leakage]   QM rows={qm['n_rows']:,}  overlap_rows={qm['overlapping_rows']}  "
          f"overlap_pairs={qm['overlapping_pairs']}")

    print("\n[leakage] Scanning CombiSolv-Exp ...")
    exp = _scan_pretraining(COMBISOLV_EXP_SRC, holdout)
    print(f"[leakage]   Exp rows={exp['n_rows']:,}  overlap_rows={exp['overlapping_rows']}  "
          f"overlap_pairs={exp['overlapping_pairs']}")

    out_path = OUT_DATA_DIR / "leakage_report.md"
    with open(out_path, "w") as f:
        f.write("# Pair-level leakage check (Transfer ablation)\n\n")
        f.write(
            "Pretraining sources (CombiSolv-QM, CombiSolv-Exp) are checked against\n"
            "the union of canonical `(solute, solvent)` pairs appearing in the SC3\n"
            "v2 holdouts: `bench_eval`, `bench_ood`, `sc3/gold`, `sc3/silver`,\n"
            "`sc3/bronze`.  Single-side overlap (same solute, different solvent\n"
            "or vice versa) is *not* counted as leakage.\n\n"
        )
        f.write("## Holdout pair counts\n\n")
        f.write("| Split | Rows | Unique canonical pairs |\n")
        f.write("|-------|-----:|----------------------:|\n")
        for name, n_rows, n_pairs in per_split:
            f.write(f"| {name} | {n_rows:,} | {n_pairs:,} |\n")
        f.write(f"| **Union** |  | **{len(holdout):,}** |\n\n")
        f.write("## Pretraining source overlap\n\n")
        f.write("| Source | Rows | Unique solutes | Unique solvents | Overlapping rows | Overlapping pairs |\n")
        f.write("|--------|-----:|---------------:|----------------:|-----------------:|------------------:|\n")
        for tag, d in [("CombiSolv-QM", qm), ("CombiSolv-Exp", exp)]:
            f.write(
                f"| {tag} | {d['n_rows']:,} | {d['n_unique_solutes']:,} | "
                f"{d['n_unique_solvents']:,} | {d['overlapping_rows']} | "
                f"{d['overlapping_pairs']} |\n"
            )
        f.write(
            "\nNote: the source files were already pair-level cleaned in\n"
            "`Solubility/sc3-benchmark/Additional_Experiments/transfer_v2/`.\n"
            "Any non-zero overlap here would mean that the new SC3 v2 splits\n"
            "(gold/silver/bronze) introduced pairs that were not present in the\n"
            "old test-tier definition used for the original cleaning.  Those are\n"
            "removed in the trainer at load time (`load_combisolv(filter_pairs=...)`).\n"
        )
    print(f"\n[leakage] Wrote {out_path}")
    return holdout


if __name__ == "__main__":
    main()
