# Ablations / Data Scaling

This experiment answers part of *Q1: What constitutes a good solubility
method — data, representation, or model?* by comparing how three very
different methods scale with training data size.

## Hypothesis

Simple, high-bias models (LightGBM on RDKit descriptors) should saturate
after seeing 20–40% of the training set. Larger, data-hungry models
(FastProp deep MLP, MolMerger AttentiveFP) should keep improving as we
feed them more data. If the larger models do not plateau on our full
training set, we can extrapolate the trend to argue that more data
(rather than a different architecture) is the limiting factor.

## Methods compared

| Key         | Description                                            | Family            |
|-------------|--------------------------------------------------------|-------------------|
| `lgb_rdkit` | LightGBM on 158 RDKit 2D descriptors + temperature     | Tabular tree      |
| `fastprop`  | Deep MLP (512-256-128, BN, dropout) on RDKit + T       | Deep descriptor   |
| `molmerger` | AttentiveFP on Gasteiger-merged solute–solvent graphs  | Merged-graph GNN  |

All three use the existing `sc3_bench` code paths and best HPs from
`vansh/configs/best_hps.json`.

## What it does

For every `(method, fraction, seed)`:

1. Reproducibly subsample `int(round(fraction * N_train))` rows from the
   training set (`np.random.RandomState(seed).choice(...)`, no
   replacement).
2. Train the method (with early stopping on the *full* `eval` split).
3. Evaluate on three held-out splits: `eval`, `ood`, `sc3_gold`.
4. Save per-run metrics to JSON.

Aggregation across seeds (mean / std) is written to
`results/<method>/summary.json`.

## Default sweep

- **Fractions:** 5%, 10%, 20%, 40%, 60%, 80%, 100%
- **Seeds:** 42, 101, 123 (3 seeds keeps GPU runs tractable; bump up if
  you want tighter error bars)
- **Eval splits:** `eval`, `ood`, `sc3_gold`

That is `3 methods x 7 fractions x 3 seeds = 63` training runs.

## Prerequisites

```bash
# 1. RDKit feature cache (used by lgb_rdkit and fastprop)
cd /DATATWO/users/solubility/Solubility/vansh
python sc3 cache --featurizers rdkit

# 2. (Optional, but recommended) Pre-build the graph caches.
#    Both files end up in feature_cache/ so they're reused across runs:
#       feature_cache/molmerger_skeletons.pt  (~9.5 K skeletons)
#       feature_cache/gcn_graphs.pt           (~1.7 K SMILES graphs)
cd Ablations/Data_Scaling
python cache_graphs.py
```

If you skip step 2, the data-scaling driver will build the caches on the
first run and persist them to the same location automatically.

## How to run

```bash
cd /DATATWO/users/solubility/Solubility/vansh/Ablations/Data_Scaling

# Smoke test (one fraction, one seed, no GPU)
python run_data_scaling.py --methods lgb_rdkit --fractions 0.1 --seeds 42

# Full sweep (default fractions and seeds, all 3 methods, GPU 0)
python run_data_scaling.py --gpu 0

# Resume / extend (existing JSONs are skipped unless --force):
python run_data_scaling.py --gpu 0 --force
```

`run_data_scaling.py` is **resumable**: existing per-run JSONs are skipped
unless you pass `--force`. So you can run lgb on CPU first, then GPU
methods on different GPUs in parallel terminals (use `--gpu 0`,
`--gpu 1`, etc., with `--methods` filters).

### Suggested parallel layout (3-GPU, 4-GPU machine)

```bash
# Tree model on CPU
python run_data_scaling.py --methods lgb_rdkit

# FastProp on GPU 0
CUDA_VISIBLE_DEVICES=0 python run_data_scaling.py --methods fastprop --gpu 0

# MolMerger on GPU 1 (slowest of the three)
CUDA_VISIBLE_DEVICES=1 python run_data_scaling.py --methods molmerger --gpu 1
```

## How to plot

```bash
python make_plots.py                      # default RMSE, linear x
python make_plots.py --metric MAE         # plot MAE instead
python make_plots.py --use-n-train --log-x   # x-axis = number of training rows, log scale
```

This writes:

- `figures/data_scaling_<split>_<metric>.png`  (per split)
- `figures/data_scaling_panel_<metric>.png`     (3-panel: eval / ood / sc3_gold)
- `results/data_scaling_table_<metric>.csv`     (long-format table you can paste into the paper)

## Output layout

```
Ablations/Data_Scaling/
    run_data_scaling.py      # driver
    scaling_trainers.py      # per-method training routines
    make_plots.py            # plotting / CSV export
    README.md                # this file
    results/
        lgb_rdkit/
            frac_0.05/  seed_42.json  seed_101.json  ...
            frac_0.1/   ...
            ...
            summary.json
        fastprop/   (same layout)
        molmerger/  (same layout)
        data_scaling_table_RMSE.csv
    figures/
        data_scaling_panel_RMSE.png
        ...
```

## Notes

- All three methods early-stop on the **full** `eval` split (not on a
  fraction of it). That keeps the validation signal stable across
  fractions, so the only thing changing per run is the *training* size.
- Subsampling is at the **row level**, not at the molecule level. If you
  want a molecule-level scaling study (rarer solutes harder to learn from),
  that's a separate ablation.
- Min training set size is clamped to 32 rows so the smallest fraction is
  still trainable.
