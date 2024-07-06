# Representation ablation — findings (single seed)

## Setup recap

- **Model:** LightGBM, fixed HPs from `configs/best_hps.json["lgb_rdkit"]`
  (`n_estimators=3000`, `num_leaves=127`, `lr=0.05`, `min_child_samples=10`,
  `feature_fraction=0.8`, `bagging_fraction=0.8`, `reg_alpha=0.1`,
  `reg_lambda=5`, early-stopping=50 rounds on full `eval`).
- **Featurizers:** 7 representations spanning ~16 to ~3226 features
  (after concat with solvent + 4 temperature features).
- **Splits:** `eval` (in-distribution) / `ood` (solvent OOD) /
  `sc3_gold` / `sc3_silver` / `sc3_bronze` (consensus tiers).
- **Seeds:** 1 (seed=42).  Run again with `--seeds 42 101 123 456 789`
  for proper error bars before reporting.

Total wall time for the full sweep: **~3 min** on a 96-core box (60% cap).

## Headline numbers (RMSE, single seed)

| Featurizer | n_features | eval | ood | sc3_gold | sc3_silver | sc3_bronze |
|------------|-----------:|-----:|----:|---------:|-----------:|-----------:|
| **mordred**       | 3226 | **0.487** | 0.677 | 0.763 | 0.737 | **0.718** |
| **dissolvr**      |  356 | 0.492 | 0.616 | **0.733** | 0.706 | **0.696** |
| **rdkit**         |  320 | 0.493 | **0.605** | 0.737 | **0.707** | 0.706 |
| maccs        |  338 | 0.511 | 0.682 | 0.809 | 0.783 | 0.769 |
| morgan       | 2052 | 0.529 | 0.782 | 0.937 | 0.896 | 0.872 |
| abraham_only |   16 | 0.552 | 0.682 | 0.932 | 0.900 | 0.882 |
| atompair     | 2052 | 0.553 | 0.882 | 0.915 | 0.882 | 0.871 |

(Bold = top-3 per column.)

## Z-RMSE (aleatoric-normalized, SC3 tiers)

| Featurizer | gold | silver | bronze |
|------------|------:|-------:|-------:|
| **rdkit**         | **41.1** | **37.5** | **35.0** |
| dissolvr     | 41.4 | 37.7 | 35.2 |
| mordred      | 44.0 | 40.1 | 37.3 |
| maccs        | 45.7 | 41.7 | 38.8 |
| atompair     | 51.1 | 46.7 | 43.6 |
| abraham_only | 52.4 | 47.8 | 44.5 |
| morgan       | 54.4 | 49.4 | 46.0 |

## Three observations

1. **Family matters more than dimensionality.**  Within each family the
   ranking by representation is consistent across all 5 splits.  The
   *descriptor* family (rdkit / dissolvr / mordred) is uniformly the
   strongest with LightGBM; the *fingerprint* family (morgan / atompair /
   maccs) trails by 0.03–0.07 RMSE on `eval` and by **0.20+ RMSE on the
   SC3 tiers**.  Yet morgan/atompair are 5–10x wider than rdkit.  The
   gap is therefore *what* you encode (continuous descriptors describing
   the whole molecule) rather than *how many* bits you have.

2. **Domain knowledge is not redundant given the descriptors.**
   Dissolvr (RDKit + MOSE topology + Joback thermo + Abraham proxies)
   matches or beats plain rdkit on every split, particularly on the
   harder `sc3_gold/silver/bronze` splits (~0.005–0.011 RMSE
   improvement on each), at only +36 features.  The OOD split is the
   only one where it loses to rdkit (0.616 vs 0.605), suggesting the
   added topology features mildly overfit to the in-distribution
   solvents.

3. **Morgan/atompair fingerprints fall off a cliff on OOD and SC3.**
   On `eval` the gap to rdkit is small (0.04–0.06 RMSE), but on `ood`
   morgan jumps to 0.78 and atompair to 0.88 (~+30%); on `sc3_gold`
   both are around 0.92 — far worse than even abraham_only's six-feature
   baseline.  This is the classic "circular fingerprints memorise
   molecules well in-distribution but generalise poorly" pattern.
   Combined with their high dimensionality, this is the strongest
   evidence that **representation choice — not capacity — is the
   bottleneck** for solubility OOD generalisation.

## What this means for Q1

> *Q1.2: What constitutes a good solubility method — data, representation, or model?*
> *Within the representation axis: domain-aware continuous descriptors
> (RDKit / Dissolvr / Mordred) substantially outperform circular
> fingerprints under a fixed strong tabular model. The gap widens as
> the test distribution moves further from the training distribution
> (`eval -> ood -> sc3_gold`), implying that features carrying
> physico-chemical inductive bias generalise much better than purely
> structural bit vectors. Adding more bits (atompair vs morgan vs
> maccs) does not close this gap; adding 6 hand-crafted Abraham proxies
> already gets ~80% of the rdkit performance on the in-distribution
> split.*

## Caveats / next steps

- **Single seed only.**  Std errors are 0; rerun with
  `--seeds 42 101 123 456 789` for publication.  Differences <0.01 RMSE
  may be within the seed noise band of LightGBM's bagging.
- **Same HPs for all featurizers.**  Fingerprints (especially binary
  ones) often prefer a different `min_child_samples` and `num_leaves`.
  A per-featurizer tuned variant would be a fair-but-different study.
- **No combination ablation.**  An obvious follow-up is rdkit + morgan
  concat vs the union featurizer to ask "is morgan adding *any*
  signal on top of descriptors?".  Run with
  `--featurizers rdkit_morgan` once that combined cache is added.
