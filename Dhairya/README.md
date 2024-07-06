# Dhairya — method experiments

This folder holds self-contained runners and hyperparameter tuning for individual model families.

## Contents

- **`sc3-benchmark-gnn-runner/`** — GNN baselines (`run_gnn_baselines.py`), manual HPO (`run_gnn_hyperparam_search.py`), and minimal `src/` + data wiring.

Add other method folders here the same way (each with its own `scripts/`, `requirements`, and `results/`).

## Data

The GNN runner loads splits from `sc3_benchmark_data_curation_v2/data/` (resolved automatically if that repo sits next to this folder or one level up). Override with:

`export SC3_DATA_DIR=/path/to/sc3_benchmark_data_curation_v2/data`
