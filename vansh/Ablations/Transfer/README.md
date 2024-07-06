# Ablations / Transfer Learning

This experiment answers **Q3: Can other chemical properties guide solubility?**
by pretraining a FastProp MLP on a large auxiliary dataset of solvation
free energies (CombiSolv-QM, ~1 M COSMO-RS computed ΔG_solv values for
solute/solvent pairs) and then fine-tuning on a small fraction of SC3
solubility (logS) data.

## Hypothesis

If solvation free energy and solubility share enough underlying chemistry
(both are governed by solute–solvent interactions), then a model
pretrained on ΔG_solv should give better solubility predictions than
a model trained from scratch — especially in the **low-data regime**.

## Why CombiSolv-QM?

| | match to SC3 |
|---|---|
| Same input shape | (solute SMILES, solvent SMILES, T) → scalar |
| Same featurization | RDKit 2D descriptors per molecule + temperature |
| Same architecture works | FastProp MLP with one regression head |
| Massive scale | 999 743 rows = 1 000× our SC3 train set, 11 029 unique solutes, 284 unique solvents |
| QM-grade labels | computed via COSMO-RS, low experimental noise |

This is the same pretraining choice used by Vermeire & Green (2022)
for D-MPNN; we replicate it on the FastProp architecture so the
transfer benefit is comparable to the rest of the SC3 benchmark
(`fastprop` baseline at 100% data is `RMSE_eval ≈ 0.460`).

## Methods compared

| Protocol | Variant | What changes |
|---|---|---|
| `scratch`  | `full`      | FastProp trained from scratch on SC3 fraction |
| `scratch`  | `head_only` | (Sanity check) only final linear head trains; trunk is random-init frozen |
| `qm`       | `full`      | Pretrain on CombiSolv-QM ΔG_solv → swap head → fine-tune all params on SC3 logS |
| `qm`       | `head_only` | Pretrain on CombiSolv-QM → freeze trunk → train only the new logS head |

## Data

| | source | rows |
|---|---|---|
| **Pretraining** | `CombiSolv-QM-clean.csv` (pair-level leakage cleaned against SC3) | 999 743 |
| **Fine-tuning** | `bench_train.csv` (SC3 train) | 61 403 |
| **Validation**  | `bench_eval.csv`             | 6 969 |
| **Test (ID)**   | `bench_eval.csv`             | 6 969 |
| **Test (OOD solvents)** | `bench_ood.csv`      | 11 940 |
| **Test (SC3 gold)** | `sc3/gold.csv`           | 4 507 |

CombiSolv data is taken from
`Solubility/sc3-benchmark/Additional_Experiments/transfer_v2/data/`,
which was already pair-level cleaned against the SC3 holdouts in
`transfer_v2`.  We re-verify the leakage against the *current*
`sc3_gold/silver/bronze` splits in `leakage_check.py`.

## Default sweep (quick mode)

- **Fractions:** 5 %, 25 %, 100 % of `bench_train`
- **Seeds:** 42, 101, 123
- **Protocols:** `scratch`, `qm`
- **Variants:** `full`, `head_only`

That is `2 × 2 × 3 × 3 = 36` fine-tune runs plus 1 pretraining run
(reused across all seeds/fractions/variants).

## How to run

```bash
cd /DATATWO/users/solubility/Solubility/vansh/Ablations/Transfer

# 0. (one-time) Verify leakage and build the CombiSolv RDKit feature cache.
#    Both are also called automatically on the first run.
python leakage_check.py
python cache_combisolv_features.py

# 1. Pretrain the FastProp trunk on CombiSolv-QM (~1 M rows).
#    Cached to models/pretrained_qm.pt; reused across the whole sweep.
python run_transfer.py --pretrain-only --gpu 0

# 2. Full sweep (smoke test: one fraction, one seed first to verify):
python run_transfer.py --protocols scratch qm --variants full --fractions 0.05 --seeds 42 --gpu 0

# 3. Full sweep (default fractions/seeds/protocols/variants):
python run_transfer.py --gpu 0

# 4. Resume / extend (existing per-run JSONs are skipped unless --force):
python run_transfer.py --gpu 0 --force
```

Per-run JSONs land in `results/<protocol>__<variant>/frac_<f>/seed_<s>.json`.
The driver writes a flat `results/transfer_table.csv` after each run for
quick inspection.

## How to plot

```bash
python make_plots.py                  # default RMSE on eval+ood+sc3_gold
python make_plots.py --metric MAE     # MAE
```

## Output layout

```
Ablations/Transfer/
    README.md                          # (this file)
    leakage_check.py                   # pair-level overlap re-verification
    cache_combisolv_features.py        # build RDKit cache for CombiSolv mols
    transfer_trainers.py               # pretrain + finetune routines (FastProp)
    run_transfer.py                    # driver
    make_plots.py                      # plotting / CSV export
    cache/
        combisolv_qm.npz               # cached RDKit features for CombiSolv-QM
        combisolv_exp.npz              # (optional) for CombiSolv-Exp
    models/
        pretrained_qm.pt               # the QM-pretrained trunk
    data/
        leakage_report.md              # # of overlapping pairs vs current SC3 splits
    results/
        scratch__full/
            frac_0.05/seed_42.json     # one JSON per (proto, variant, frac, seed)
            ...
        qm__full/                       ...
        qm__head_only/                  ...
        scratch__head_only/             ...
        transfer_table.csv             # long-format table
        summary.json                   # aggregated mean ± std
    figures/
        transfer_panel_<metric>.png    # 3-panel (eval/ood/sc3_gold) per metric
        transfer_grouped_<metric>.png  # all splits, grouped by protocol
    logs/
```

## Notes

- Both pretraining and fine-tuning use **input normalisation fit on the
  pretraining set** (so the pretrained trunk sees inputs in the same
  scale at fine-tune time).  For the `scratch` runs we fit normalisation
  on the SC3 training fraction itself.
- BatchNorm running stats are **re-calibrated** with one no-grad forward
  pass over the fine-tune training data before the first eval, otherwise
  stale pretrained BN means corrupt the early-stopping signal.  The
  same calibration is applied to the scratch model for an apples-to-apples
  comparison.
- All training (pretrain and fine-tune) early-stops on the *full*
  `bench_eval` split with patience = 20 (same as Q4).
- The 7th/8th feature column is the *temperature* feature block.
  CombiSolv-QM is at a constant 298.15 K, so during pretraining those
  4 columns are constant; that's not a problem (BN absorbs the offset),
  and at fine-tune time SC3 introduces real temperature variation.
- Subsampling of SC3 training is **stratified by solvent** (`Solvent_Name`)
  so even at 5 % we still see every common solvent.
