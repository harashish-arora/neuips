#!/usr/bin/env python3
"""
Run all CPU-bound baselines for SC3 benchmark.

Usage:
    python scripts/run_cpu_baselines.py [--quick]
    python scripts/run_cpu_baselines.py --methods rf_rdkit xgb_rdkit
"""

import sys
import os
import argparse

# Limit to 1 thread to avoid hogging the machine
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.pipeline import run_benchmark


CPU_METHODS = [
    "gse",
    "esol",
    "dt_rdkit",
    "knn_rdkit",
    "mlp_rdkit",
    "rf_rdkit",
    "xgb_rdkit",
    "lgb_rdkit",
    "catboost_rdkit",
    "fastprop",
    "dissolvr",
]


def main():
    parser = argparse.ArgumentParser(description="Run CPU baselines")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to run. Default: all CPU methods. Available: {CPU_METHODS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 seed only")
    parser.add_argument("--featurizer", default=None,
                        choices=["rdkit", "morgan", "dissolvr"],
                        help="Override featurizer for all methods")
    args = parser.parse_args()

    methods = args.methods or CPU_METHODS
    seeds = [42] if args.quick else args.seeds

    print("SC3 CPU Baseline Runner")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")
    print(f"  Featurizer: {args.featurizer or 'per-method default'}")
    print()

    run_benchmark(methods, seeds, args.featurizer)


if __name__ == "__main__":
    main()
