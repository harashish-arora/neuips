#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1
cd /DATATWO/users/solubility/sc3-benchmark
eval "$(/opt/anaconda/anaconda3/bin/conda shell.bash hook)"
conda activate /DATATWO/users/solubility/myenv
echo "=== MolMerger run starting at $(date) ==="
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -u scripts/run_molmerger.py --gpu 2 --epochs 200 --patience 25 --batch_size 256 --hidden_dim 200 --num_layers 3 --num_timesteps 2 --dropout 0.2 --lr 1e-3
echo "=== MolMerger run finished at $(date) ==="
