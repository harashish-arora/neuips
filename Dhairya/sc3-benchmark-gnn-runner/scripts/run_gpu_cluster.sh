#!/bin/bash
#SBATCH --job-name=sc3-gnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sc3_gnn_%j.log

# SC3 GPU GNN benchmark runner (for SLURM cluster)
#
# Usage:
#   sbatch scripts/run_gpu_cluster.sh          # submit to cluster
#   bash scripts/run_gpu_cluster.sh            # run locally with GPU

set -e
mkdir -p logs

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "============================================"
echo "SC3 GPU GNN Benchmarks"
echo "Device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")')"
echo "============================================"

# GNN baselines with more epochs on GPU
echo ""
echo "[1/1] GNN baselines (GPU)..."
python scripts/run_gnn_baselines.py --gpu --epochs 200 --patience 25 --hidden_dim 128

echo "GPU GNN benchmarks complete!"
