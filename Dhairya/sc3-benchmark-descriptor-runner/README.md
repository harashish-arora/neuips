# SC3 Descriptor Runner (Dhairya)

Lightweight runner bundle for descriptor-based neural network baselines and manual hyperparameter search.

This folder is meant for running:
- `scripts/run_descriptor_baselines.py`
- `scripts/run_descriptor_hyperparam_search.py`

It is aligned to the Hulk workflow in `HULK_DATA/HOW_TO_RUN.md`.

## Models

| Model    | Architecture                              | Notes |
|----------|-------------------------------------------|-------|
| FastProp | Linear→BN→ReLU→Dropout × N → Linear(1)   | RDKit descriptors, default (512,256,128) |
| FastSolv | Same + Sobolev gradient regularization    | Enforces thermodynamic consistency (dlogS/dT) |
| MLP      | Linear→ReLU→Dropout × N → Linear(1)      | No BatchNorm, PyTorch-based |

All models use dual-descriptor featurization: solute RDKit descriptors ⊕ solvent RDKit descriptors ⊕ 4 temperature features [T/300, 1000/T, (T/300)², ln(T/300)].

## Paths and Environment (Hulk)

```bash
PYTHON=/DATATWO/users/solubility/myenv/bin/python3
RUNNER_DIR=/DATATWO/users/solubility/HULK_DATA/Dhairya/sc3-benchmark-descriptor-runner
```

Use this Python directly (system Python is not configured with all packages).

Quick check:
```bash
$PYTHON -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data Source

By default, the runner loads splits from:
- `../sc3_benchmark_data_curation_v2/data`
- or `../../sc3_benchmark_data_curation_v2/data`

If needed, set explicitly:
```bash
export SC3_DATA_DIR=/DATATWO/users/solubility/HULK_DATA/sc3_benchmark_data_curation_v2/data
```

## GPU Selection

Check free GPUs first:
```bash
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
```

Pin to one GPU:
```bash
CUDA_VISIBLE_DEVICES=2
```

## Baseline Runs

```bash
cd "$RUNNER_DIR"

# All models (FastProp, FastSolv, MLP)
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_descriptor_baselines.py --gpu

# Single model, quick smoke test
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_descriptor_baselines.py --gpu --model_type fastprop --quick

# Custom hyperparameters
$PYTHON scripts/run_descriptor_baselines.py --model_type mlp --hidden_dims 256 128 --dropout 0.2
```

Optional Makefile shortcuts:
```bash
cd "$RUNNER_DIR"
make quick
make fastprop
make fastsolv
make mlp
make all
```

## Hyperparameter Search (Manual Grid)

Edit search space in:
- `scripts/run_descriptor_hyperparam_search.py` (top-level `_HPO_*` tuples)

Run all model types:
```bash
cd "$RUNNER_DIR"
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_descriptor_hyperparam_search.py --gpu
```

Run only one model type:
```bash
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_descriptor_hyperparam_search.py --gpu --only fastprop
```

Quick mode:
```bash
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_descriptor_hyperparam_search.py --gpu --quick
```

## Long Runs (tmux recommended)

```bash
PYTHON=/DATATWO/users/solubility/myenv/bin/python3
RUNNER_DIR=/DATATWO/users/solubility/HULK_DATA/Dhairya/sc3-benchmark-descriptor-runner

tmux new-session -d -s desc_hpo \
  "cd $RUNNER_DIR && CUDA_VISIBLE_DEVICES=2 stdbuf -oL $PYTHON -u scripts/run_descriptor_hyperparam_search.py --gpu > hpo.log 2>&1"

tail -f "$RUNNER_DIR/hpo.log"
```

## Outputs

All outputs are under `results/`.

### Baselines
- `results/fastprop/`, `results/fastsolv/`, `results/mlp/`
- each contains:
  - `raw_results.json`
  - `summary.json`
  - per-seed folders with checkpoints, epoch history, predictions

### HPO
- per-trial runs:
  - `results/hpo_fastprop_trial_XXX/`, `results/hpo_fastsolv_trial_XXX/`, `results/hpo_mlp_trial_XXX/`
- per-model rollups:
  - `results/hpo_descriptor_fastprop/raw_results.json`, `summary.json`
  - `results/hpo_descriptor_fastsolv/raw_results.json`, `summary.json`
  - `results/hpo_descriptor_mlp/raw_results.json`, `summary.json`
- global rollup:
  - `results/hpo_descriptor/raw_results.json`
  - `results/hpo_descriptor/summary.json`

## Notes

- Thread counts are intentionally capped in scripts to avoid hogging shared CPUs.
- `--seed` in HPO is a single seed per sweep run (default `42`).
- FastSolv uses `sobolev_scale=10.0` (fixed, not tuned in HPO).
