#!/bin/bash
# Run GNN benchmarks only (bundle-friendly script).
#
# Usage:
#   bash scripts/run_all.sh          # full run (5 seeds)
#   bash scripts/run_all.sh --quick  # smoke test (1 seed)

set -e

# Limit threads to avoid hogging the machine
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

QUICK_FLAG=""
if [[ "$1" == "--quick" ]]; then
    QUICK_FLAG="--quick"
    echo "=== QUICK MODE (1 seed) ==="
fi

echo "============================================"
echo "SC3 Benchmark: GNN Pipeline"
echo "============================================"

# Phase 1: GNN baselines
echo ""
echo "[1/1] GNN baselines..."
python scripts/run_gnn_baselines.py $QUICK_FLAG

echo ""
echo "============================================"
echo "GNN benchmarks complete!"
echo "============================================"