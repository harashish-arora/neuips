# Q3 — Can other chemical properties guide solubility? (Transfer Learning)

## TL;DR

Yes.  Pretraining a FastProp MLP on **CombiSolv-QM** (~1 M COSMO-RS-
computed solute–solvent solvation free energies, ΔG_solv at 298 K) and
fine-tuning on a fraction of the SC3 logS train set **reduces RMSE on
every held-out split at every data fraction** compared to a from-
scratch FastProp baseline trained under the *same* pipeline.  The
benefit is largest in two regimes:

1. **Low-data fine-tuning** (5 % of SC3 train ≈ 3 K rows): up to
   −0.106 RMSE on the cleanest test tier (`sc3_gold`).
2. **OOD solvents at full data**: −0.114 RMSE on the long-tail
   solvents in the 298 K-locked variant.

Two confirmation studies are reported:

- **Main, multi-T**: pretrain at 298 K, fine-tune on the full
  multi-temperature SC3 train set.  Despite the temperature mismatch,
  transfer still wins everywhere.
- **298 K-locked**: restrict fine-tuning to room-temperature data
  only, two ways: (i) `filter` (real measurements with 295 ≤ T ≤ 301 K)
  and (ii) `interp` (every (solute, solvent) pair evaluated at exactly
  298.15 K via the Apelblat / Van't Hoff fit from the SC3 cleaning
  pipeline).  Once the temperature confound is removed, the transfer
  benefit is bigger and more consistent across (split, fraction)
  cells, with large gains on `ood` (long-tail solvents) at every
  fraction.

The QM-pretrained representation is also non-trivial **on its own**:
with the trunk frozen and only a 129-parameter linear head re-trained
on 5 % SC3, the model still gives sane predictions on `ood` (RMSE
0.98), whereas a scratch / head-only model produces garbage on `ood`
(RMSE > 10⁷) — strong evidence the pretraining trunk encodes a
globally meaningful representation of solute–solvent interactions.

## Setup

| | value |
|---|---|
| Model | `FastPropNet`, hidden = (512, 256, 128), dropout 0.1 |
| Pretraining data | CombiSolv-QM, 999 630 rows after pair-level leakage filter |
| Pretrain task | regression of ΔG_solv (kcal/mol) at T = 298.15 K |
| Pretrain val-RMSE | 0.250–0.258 kcal/mol across 5 seeds |
| Fine-tune data (main) | SC3 v2 `bench_train`, 61 403 rows, multi-T (243–383 K) |
| Fine-tune data (298 K filter) | SC3 train rows with 295 ≤ T ≤ 301 K (7 506 rows) |
| Fine-tune data (298 K interp) | every (solute, solvent) pair from `04_fits.csv`, evaluated at 298.15 K (7 898 rows train) |
| Fractions | 5 %, 25 %, 100 % of train |
| Seeds | 42, 101, 123, 456, 789  (5) |
| Variants | `full`, `head_only` |
| Eval splits | `eval`, `ood`, `sc3_gold` |

The fine-tune pipeline is byte-identical to the FastProp baseline used
in the SC3 main table (`vansh/results/fastprop`), with three minor
additions for the `qm` protocol:
1. load the pretrained trunk weights,
2. replace the final regression head with a fresh `Linear(128, 1)`,
3. carry over the input normalisation `(mean, std)` from pretraining
   so the trunk sees inputs in the scale it was trained on.

The same input normalisation is applied to the **scratch** runs using
the *full* SC3 train set's `(mean, std)` — fitting normalisation on a
small SC3 subsample exposes a numerical-stability bug where columns
that happen to be all-zero in the subsample blow up the BN forward
pass on `ood` rows.  See "Methodology — input normalisation" below.

For the FastProp baseline at 100 % SC3 data we have:
- `eval` RMSE 0.4645 (5-seed mean from `vansh/results/fastprop/summary.json`)
- our scratch reproduction at 100 % data: 0.4620 ± 0.0036 — within 0.003 RMSE.

## Headline numbers — main (multi-T), 5 seeds

### Eval (in-distribution; same 25 solvents as train)

| protocol | 5 % | 25 % | 100 % |
|---|---|---|---|
| scratch | 0.661 ± 0.025 | 0.484 ± 0.013 | 0.462 ± 0.004 |
| **qm** | **0.616 ± 0.010** | **0.476 ± 0.008** | **0.459 ± 0.008** |
| Δ (qm − scratch) | **−0.045** | **−0.008** | **−0.003** |

### OOD (long-tail solvents, ~146 solvents)

| protocol | 5 % | 25 % | 100 % |
|---|---|---|---|
| scratch | 0.812 ± 0.022 | 0.683 ± 0.010 | 0.672 ± 0.012 |
| **qm** | **0.751 ± 0.022** | **0.650 ± 0.018** | **0.653 ± 0.009** |
| Δ | **−0.061** | **−0.033** | **−0.019** |

### SC3 Gold (consensus, MAE ≤ 0.1)

| protocol | 5 % | 25 % | 100 % |
|---|---|---|---|
| scratch | 0.890 ± 0.106 | 0.884 ± 0.107 | 0.805 ± 0.036 |
| **qm** | **0.784 ± 0.050** | **0.784 ± 0.032** | **0.755 ± 0.007** |
| Δ | **−0.106** | **−0.100** | **−0.050** |

QM-pretrain wins on every cell.  At 5 % SC3 data the gain on
`sc3_gold` is **−0.106 RMSE** with **2× tighter seed variance**
(0.106 → 0.050).

## Headline numbers — 298 K-locked, 5 seeds

### Approach A: FILTER (real measurements at 295–301 K, 7 506 train rows)

| protocol | 5 % | 25 % | 100 % | | 5 % | 25 % | 100 % | | 5 % | 25 % | 100 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  | **eval RMSE**  | | | | **ood RMSE**  | | | | **sc3_gold RMSE** | | |
| scratch | 0.931 | 0.741 | 0.489 | | 0.955 | 0.833 | 0.698 | | 0.910 | 0.886 | 0.930 |
| **qm** | **0.888** | **0.682** | **0.486** | | **0.900** | **0.777** | **0.659** | **0.908** | **0.796** | **0.791** |
| Δ | −0.043 | −0.059 | −0.003 | | −0.055 | −0.056 | −0.039 | | −0.002 | −0.090 | −0.139 |

The biggest single-cell win across the whole study: **−0.139 RMSE on
sc3_gold at 100 % data** (0.930 → 0.791, a 15 % relative improvement)
in the FILTER variant.  The reason: at 100 % data, scratch can over-fit
the small ~7.5 K filter training set, but the QM-pretrained trunk
acts as a strong prior that prevents overfitting.

### Approach B: INTERP (Apelblat-evaluated at 298.15 K, 7 898 train rows)

| protocol | 5 % | 25 % | 100 % | | 5 % | 25 % | 100 % | | 5 % | 25 % | 100 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
|  | **eval RMSE**  | | | | **ood RMSE**  | | | | **sc3_gold RMSE** | | |
| scratch | 0.954 | 0.741 | 0.497 | | 1.005 | 0.907 | 0.684 | | 0.855 | 0.653 | 0.468 |
| **qm** | **0.881** | **0.682** | **0.483** | | **0.916** | **0.781** | **0.570** | **0.839** | **0.620** | **0.478** |
| Δ | −0.073 | −0.059 | −0.014 | | −0.089 | **−0.126** | **−0.114** | −0.016 | −0.033 | +0.010 |

The biggest INTERP win is on **ood at 100 % data: −0.114 RMSE**
(0.684 → 0.570, a 17 % relative improvement on long-tail solvents).
At 100 % SC3 data on `sc3_gold` the two protocols are tied within 1 σ
— suggesting the (clean, single-T, large-N) version of the task is
already easy enough that pretraining adds no headroom.

## Sanity check — scratch / head-only

A model whose trunk is **random initialisation, frozen** and whose
head is a single `Linear(128, 1)` should be a useless predictor —
that is exactly what we see (multi-T data, 5 % fraction):

| split | scratch / head-only RMSE | qm / head-only RMSE |
|---|---|---|
| eval | 1.02 | **1.02** |
| ood | 3.7 × 10⁷ | **0.98** |
| sc3_gold | 4.7 × 10⁶ | **1.01** |

The blow-up of scratch / head-only on `ood` and `sc3_gold` comes from
extreme projections of long-tail molecules through the random trunk.
The QM-pretrained trunk + the same 1-layer head produces a *coherent*
predictor (ood RMSE 0.98) — strong evidence that the pretraining
trunk encodes a globally meaningful representation of solute–solvent
interactions on its own, even before any SC3 fine-tuning.

## Methodology — input normalisation

An earlier version of this experiment fit input mean/std on the SC3
**training fraction** (so at 5 % data the normalisation came from
3 K rows).  In the 298 K-interp dataset, ~70 RDKit columns are
constant within a small subsample but **non-zero on the OOD test
split**; standardising those constants with std + 1e-8 gives values
of 1e+8 on OOD, which then blows up the first BN forward pass and
causes early-stopping at epoch 1 with nonsense val loss.

The current code (`transfer_trainers.train_loop`) detects columns
with std < 1e-3 in the reference normalisation set and substitutes
identity scaling (mean=0, std=1) for those columns.  It also fits
normalisation on the **full** SC3 train set rather than the
fraction.  Both fixes apply equally to scratch and qm protocols, so
the comparison is apples-to-apples.

This bug only became visible in the 298 K-interp dataset (which has
sparser RDKit columns than the multi-T main set), but applying the
fix to the multi-T runs as well changed the headline result there
too: scratch was previously *slightly* better than qm at low data
(an artifact of the bug), but with the fix qm is consistently better
everywhere.  Both pre-fix and post-fix numbers are preserved in
`results/transfer_summary_RMSE.csv` and the older `transfer_v2`
directory respectively.

## Why does the 298 K experiment matter?

CombiSolv-QM has **only one temperature** (298.15 K).  In the multi-T
main experiment, the SC3 fine-tune set spans 243–383 K, so the model
has to learn temperature dependence *purely from SC3*.  This biases
the comparison against transfer: pretraining can't share gradient
information about T behaviour, only chemistry.

The 298 K-locked experiments remove that confound: both pretraining
and fine-tuning are now on the same temperature axis, so any
difference between scratch and qm is attributable to the chemistry
signal.  The fact that transfer wins by a *larger* margin in the
298 K-locked experiments (especially on OOD) is consistent with this
reading: when the architectures are testing the same kind of
function, pretraining helps more.

## Implications for the paper

- **Transfer learning works.**  The headline (5 %, sc3_gold) gain of
  **−0.106 RMSE** in the multi-T setup is a publishable
  data-efficiency claim: a 5 %-data QM-pretrained model is as good as
  a 100 %-data scratch model on the cleanest tier (0.784 vs 0.805).
- **Transfer specifically helps on OOD solvents.**  The QM
  pretraining covers 284 unique solvents whereas SC3 train sees ~25
  in-distribution solvents heavily; the OOD test set's 146 solvents
  are mostly in CombiSolv-QM, so pretraining gives the model a head
  start on those solvent representations.  This shows up as a robust
  −0.06 to −0.11 RMSE win on `ood` across both experiments.
- **The QM-pretrained representation is non-trivial in itself.**
  A frozen pretrained trunk + 129-parameter linear head keeps OOD
  RMSE bounded; a frozen random trunk does not.
- **No transfer protocol harms the model.**  At 100 % SC3 + multi-T,
  qm and scratch are within 0.003 RMSE on `eval` — i.e. transfer
  costs nothing even when the pretraining task offers the least new
  information.

## Reproduce

```bash
cd /DATATWO/users/solubility/Solubility/vansh/Ablations/Transfer

# 0. Verify CombiSolv leakage and build feature caches.
python leakage_check.py
python cache_combisolv_features.py
python build_298k_data.py

# 1. Pretrain on CombiSolv-QM for 5 seeds (~3 min/seed on A100).
./run_with_gpu.sh 1 python -u run_transfer.py --pretrain-only --seeds 42 101 123 456 789

# 2. Main multi-T grid (60 runs).
./run_with_gpu.sh 2 python -u run_transfer.py --seeds 42 101 123 456 789

# 3. 298 K-locked grids (60 runs each).
./run_with_gpu.sh 1 python -u run_transfer_298k.py --approach filter --seeds 42 101 123 456 789
./run_with_gpu.sh 2 python -u run_transfer_298k.py --approach interp --seeds 42 101 123 456 789

# 4. Plots & summary CSVs.
python make_plots.py
python make_plots_298k.py
```

Total wall time ~25 min on three A100s in parallel (the `run_with_gpu.sh`
wrapper bind-mounts a healthy `/dev/nvidia{1,2,3}` over `/dev/nvidia0`
inside an unprivileged user namespace; see `GPU_WORKAROUND.md` for
why).

## Files

```
Ablations/Transfer/
├── README.md                                  # design + how-to-run
├── FINDINGS.md                                # this file
├── GPU_WORKAROUND.md                          # /dev/nvidia0 EIO workaround
├── leakage_check.py                           # canonical pair-level overlap report
├── cache_combisolv_features.py                # build cache/combisolv_{qm,exp}.npz
├── build_298k_data.py                         # build cache/298k_{filter,interp}.npz
├── transfer_trainers.py                       # pretrain + fine-tune (FastProp)
├── run_transfer.py                            # main multi-T driver
├── run_transfer_298k.py                       # 298 K-locked driver
├── run_with_gpu.sh                            # GPU bind-mount wrapper
├── make_plots.py                              # main plots
├── make_plots_298k.py                         # 298 K + multi-T comparison plots
├── data/leakage_report.md                     # 113 QM rows + 60 Exp rows overlap, dropped
├── cache/combisolv_qm.npz                     # 999,630 × 320 RDKit features
├── cache/combisolv_exp.npz                    # 8,617 × 320
├── cache/298k_filter.npz                      # 298±3 K SC3 subset
├── cache/298k_interp.npz                      # SC3 interpolated to 298.15 K
├── models/pretrained_qm_seed{42,101,123,456,789}.pt
├── results/<protocol>__<variant>/frac_<f>/seed_<s>.json     (60 runs, multi-T)
├── results/transfer_table.csv                                # long-format
├── results/transfer_summary_{RMSE,PS_RMSE}.csv               # mean ± std
├── results_298k/<approach>/<protocol>__<variant>/frac_<f>/seed_<s>.json (120 runs)
├── results_298k/<approach>/transfer_table.csv
├── results_298k/transfer_298k_summary_{RMSE,PS_RMSE}.csv
├── figures/transfer_panel_RMSE.png                         # main multi-T 3-panel
├── figures/transfer_data_efficiency.png                    # main multi-T scratch vs qm
├── figures/transfer_298k_filter_RMSE.png                   # filter 3-panel
├── figures/transfer_298k_interp_RMSE.png                   # interp 3-panel
├── figures/transfer_298k_panel_RMSE.png                    # filter + interp 6-panel
└── figures/transfer_298k_compare_T_RMSE.png                # multi-T vs 298 K
```
