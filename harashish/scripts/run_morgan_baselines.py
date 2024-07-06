#!/usr/bin/env python3
"""
Run Morgan fingerprint baselines for SC3 benchmark.

Same ML models as CPU baselines but with Morgan ECFP4 features instead of RDKit descriptors.

Usage:
    python scripts/run_morgan_baselines.py [--quick]
    python scripts/run_morgan_baselines.py --methods rf_rdkit xgb_rdkit
"""

import sys
import os
import argparse

# Limit to 1 thread to avoid hogging the machine
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarks.pipeline import run_benchmark


MORGAN_METHODS = [
    "rf_rdkit",
    "xgb_rdkit",
    "lgb_rdkit",
    "catboost_rdkit",
]


def main():
    parser = argparse.ArgumentParser(description="Run Morgan fingerprint baselines")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to run. Default: {MORGAN_METHODS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 123, 456, 789])
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 seed only")
    args = parser.parse_args()

    methods = args.methods or MORGAN_METHODS
    seeds = [42] if args.quick else args.seeds

    print("SC3 Morgan Fingerprint Baseline Runner")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")
    print(f"  Featurizer: morgan (ECFP4, 1024-bit)")
    print()

    run_benchmark(methods, seeds, featurizer_name="morgan")


if __name__ == "__main__":
    main()
