#!/bin/bash
# Run all SC3 benchmarks.
# CPU methods run sequentially. GPU methods should be run on a cluster.
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
echo "SC3 Benchmark: Full Pipeline"
echo "============================================"

# Phase 1: CPU baselines with RDKit descriptors
echo ""
echo "[1/6] CPU baselines (RDKit descriptors)..."
python scripts/run_cpu_baselines.py $QUICK_FLAG

# Phase 2: CPU baselines with Morgan fingerprints
echo ""
echo "[2/6] CPU baselines (Morgan fingerprints)..."
python scripts/run_morgan_baselines.py $QUICK_FLAG

# Phase 3: Dissolvr method
echo ""
echo "[3/6] Dissolvr baseline..."
python scripts/run_cpu_baselines.py --methods dissolvr $QUICK_FLAG

# Phase 4: GNN baselines (CPU — slow, prefer GPU)
echo ""
echo "[4/6] GNN baselines (CPU)..."
python scripts/run_gnn_baselines.py $QUICK_FLAG

# Phase 5: SolTranNet (CPU — slow, prefer GPU)
echo ""
echo "[5/6] SolTranNet..."
python scripts/run_transformer_baselines.py $QUICK_FLAG

# Phase 6: Collect results
echo ""
echo "[6/6] Collecting results..."
python scripts/collect_results.py

echo ""
echo "============================================"
echo "All benchmarks complete!"
echo "============================================"
