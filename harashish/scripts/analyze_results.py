#!/usr/bin/env python3
"""
Detailed analysis of benchmark results.

Produces:
  - Tier degradation analysis (Easy → Medium → Hard)
  - Per-solvent breakdown for top methods
  - Aleatoric analysis (Z-RMSE, f_aleatoric)
  - Representation comparison (RDKit vs Morgan)

Usage:
    python scripts/analyze_results.py
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

METHOD_DISPLAY = {
    "gse": "GSE", "esol": "ESOL",
    "rf_rdkit": "RF (RDKit)", "xgb_rdkit": "XGBoost (RDKit)",
    "lgb_rdkit": "LightGBM (RDKit)", "catboost_rdkit": "CatBoost (RDKit)",
    "knn_rdkit": "kNN (RDKit)", "mlp_rdkit": "MLP (RDKit)", "dt_rdkit": "DT (RDKit)",
    "rf_morgan": "RF (Morgan)", "xgb_morgan": "XGBoost (Morgan)",
    "lgb_morgan": "LightGBM (Morgan)", "catboost_morgan": "CatBoost (Morgan)",
    "gnn_gcn": "GCN", "gnn_gat": "GAT", "gnn_gin": "GIN",
    "fastprop": "FastProp", "dissolvr": "Dissolvr",
    "soltrannet": "SolTranNet",
}


def load_all_summaries():
    summaries = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summaries[d.name] = json.load(f)
    return summaries


def tier_degradation(summaries):
    """How much does PS-RMSE increase from Easy → Hard?"""
    print("\n" + "=" * 80)
    print("TIER DEGRADATION ANALYSIS (PS-RMSE: Easy -> Medium -> Hard)")
    print("=" * 80)

    header = f"{'Method':<22} {'Easy PS-RMSE':>12} {'Med PS-RMSE':>12} {'Hard PS-RMSE':>12} {'D Hard-Easy':>12} {'% Increase':>12}"
    print(header)
    print("-" * 85)

    rows = []
    for name, s in summaries.items():
        easy = s.get("test_easy_PS_RMSE_mean")
        med = s.get("test_medium_PS_RMSE_mean")
        hard = s.get("test_hard_PS_RMSE_mean")
        if easy and med and hard:
            delta = hard - easy
            pct = 100 * delta / easy if easy > 0 else float("nan")
            rows.append((name, easy, med, hard, delta, pct))

    for name, easy, med, hard, delta, pct in sorted(rows, key=lambda x: x[3]):
        display = METHOD_DISPLAY.get(name, name)
        print(f"{display:<22} {easy:>12.4f} {med:>12.4f} {hard:>12.4f} {delta:>+12.4f} {pct:>+11.1f}%")


def aleatoric_analysis(summaries):
    """Z-RMSE analysis: how close are models to the aleatoric limit?"""
    print("\n" + "=" * 80)
    print("ALEATORIC ANALYSIS (Z-RMSE -- lower is better, 1.0 = aleatoric limit)")
    print("=" * 80)

    header = f"{'Method':<22} {'Hard Z-RMSE':>12} {'Med Z-RMSE':>12} {'Easy Z-RMSE':>12}"
    print(header)
    print("-" * 60)

    rows = []
    for name, s in summaries.items():
        hard_z = s.get("test_hard_Z_RMSE_mean")
        med_z = s.get("test_medium_Z_RMSE_mean")
        easy_z = s.get("test_easy_Z_RMSE_mean")
        if hard_z:
            rows.append((name, hard_z, med_z, easy_z))

    for name, hard_z, med_z, easy_z in sorted(rows, key=lambda x: x[1]):
        display = METHOD_DISPLAY.get(name, name)
        m_str = f"{med_z:.1f}" if med_z else "---"
        e_str = f"{easy_z:.1f}" if easy_z else "---"
        print(f"{display:<22} {hard_z:>12.1f} {m_str:>12} {e_str:>12}")


def representation_comparison(summaries):
    """Compare RDKit vs Morgan fingerprints for the same ML methods."""
    print("\n" + "=" * 80)
    print("REPRESENTATION COMPARISON (RDKit vs Morgan, Hard PS-RMSE)")
    print("=" * 80)

    pairs = [
        ("rf_rdkit", "rf_morgan", "Random Forest"),
        ("xgb_rdkit", "xgb_morgan", "XGBoost"),
        ("lgb_rdkit", "lgb_morgan", "LightGBM"),
        ("catboost_rdkit", "catboost_morgan", "CatBoost"),
    ]

    header = f"{'Method':<22} {'RDKit PS-RMSE':>14} {'Morgan PS-RMSE':>14} {'Winner':>10}"
    print(header)
    print("-" * 65)

    for rdkit_key, morgan_key, display in pairs:
        r = summaries.get(rdkit_key, {}).get("test_hard_PS_RMSE_mean")
        m = summaries.get(morgan_key, {}).get("test_hard_PS_RMSE_mean")

        r_str = f"{r:.4f}" if r else "---"
        m_str = f"{m:.4f}" if m else "---"

        if r and m:
            winner = "RDKit" if r < m else "Morgan"
        else:
            winner = "---"

        print(f"{display:<22} {r_str:>14} {m_str:>14} {winner:>10}")


def timing_summary(summaries):
    """Training time comparison."""
    print("\n" + "=" * 80)
    print("TRAINING TIME (single seed, seconds)")
    print("=" * 80)

    rows = []
    for name, s in summaries.items():
        t = s.get("train_time_mean_s", 0)
        rows.append((name, t))

    for name, t in sorted(rows, key=lambda x: x[1]):
        display = METHOD_DISPLAY.get(name, name)
        print(f"  {display:<22} {t:>8.1f}s")


def main():
    summaries = load_all_summaries()
    if not summaries:
        print("No results found")
        return

    print(f"Analyzing {len(summaries)} methods: {sorted(summaries.keys())}")

    tier_degradation(summaries)
    aleatoric_analysis(summaries)
    representation_comparison(summaries)
    timing_summary(summaries)


if __name__ == "__main__":
    main()
