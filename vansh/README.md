# SC3 Benchmark

Benchmarking toolkit for the SC3 multi-solvent solubility prediction challenge. Trains and evaluates 18 methods across 5 random seeds on standardized train/eval/OOD/SC3-tier splits.

## Methods

| Family | Methods |
|--------|---------|
| Desc + Tree | LightGBM, CatBoost, XGBoost, Random Forest, Decision Tree (all with RDKit 2D descriptors) |
| Domain | Dissolvr (LightGBM + MOSE/Joback/Abraham features) |
| FP + Tree | LightGBM, CatBoost, XGBoost, RF (Morgan ECFP4 1024-bit) |
| FP + GP | GP with Tanimoto kernel (Morgan fingerprints) |
| Deep Desc | FastProp, FastSolv, MLP (PyTorch NNs on RDKit descriptors) |
| GNN | GCN, GAT, GIN (dual-encoder graph neural networks) |
| Mordred + RF | Tayyebi (Mordred descriptors + variance/correlation filtering + RF) |
| Merged GNN | MolMerger (Gasteiger charge-based graph merging + AttentiveFP) |

## Quick start

```bash
pip install -r requirements.txt

# 1. Precompute feature caches (one-time, ~4 min)
python sc3 cache

# 2. List all methods
python sc3 list

# 3. Train a single method
python sc3 run --method lgb_rdkit
python sc3 run --method gcn --gpu 0

# 4. Train all methods
python sc3 run --all --gpu 0

# 5. Check results
python sc3 status

# 6. Export to CSV
python sc3 collect
```

## Evaluation splits

| Split | Description | Rows |
|-------|-------------|------|
| `eval` | In-distribution held-out pairs (top-25 solvents) | ~7K |
| `ood` | Solvent out-of-distribution (long-tail solvents) | ~12K |
| `sc3_gold` | SC3 Gold tier (consensus labels, MAE <= 0.1) | ~4.5K |
| `sc3_silver` | SC3 Silver tier (MAE <= 0.2) | ~5.5K |
| `sc3_bronze` | SC3 Bronze tier (MAE <= 0.5) | ~6.3K |

## Metrics

- **RMSE** -- root mean squared error
- **MAE** -- mean absolute error
- **R2** -- coefficient of determination
- **PS-RMSE** -- per-solvent RMSE (mean of per-solvent RMSEs)
- **Z-RMSE** -- aleatoric-normalized RMSE (error / sigma, SC3 tiers only)
- **f_aleatoric** -- fraction of predictions within 2 sigma

## Directory structure

```
sc3                      # CLI entry point
sc3_bench/               # Python package
    data.py              # data loading + feature caching
    evaluate.py          # metrics
    featurizers.py       # RDKit, Morgan, Dissolvr, Mordred
    registry.py          # method/HP registries
    train.py             # unified training dispatcher
    collect.py           # results aggregation
    models/
        tree_models.py       # RF, XGB, LGB, CatBoost, DT, Tayyebi, GP
        descriptor_models.py # FastProp, FastSolv, MLP
        gnn_models.py        # GCN, GAT, GIN
configs/
    best_hps.json        # tuned hyperparameters for all 18 methods
results/                 # per-method results (generated)
feature_cache/           # precomputed .npz (generated)
```

## Data

Expects SC3 curation v2 data at `../sc3_benchmark_data_curation_v2/data/`. Override with `SC3_DATA_DIR` env var.
