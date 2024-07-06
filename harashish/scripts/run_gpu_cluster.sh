#!/bin/bash
#SBATCH --job-name=sc3-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sc3_gpu_%j.log

# SC3 GPU benchmark runner (for SLURM cluster)
#
# Usage:
#   sbatch scripts/run_gpu_cluster.sh          # submit to cluster
#   bash scripts/run_gpu_cluster.sh            # run locally with GPU

set -e

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "============================================"
echo "SC3 GPU Benchmarks"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "============================================"

# GNN baselines with more epochs on GPU
echo ""
echo "[1/2] GNN baselines (GPU)..."
python scripts/run_gnn_baselines.py --gpu --epochs 200 --patience 25 --hidden_dim 128

# SolTranNet with more capacity on GPU
echo ""
echo "[2/2] SolTranNet (GPU)..."
python scripts/run_transformer_baselines.py --gpu --epochs 200 --patience 25 --embed_dim 128

# Collect
echo ""
python scripts/collect_results.py

echo "GPU benchmarks complete!"
