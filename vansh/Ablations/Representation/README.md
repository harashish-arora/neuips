# Ablations / Representation

This experiment answers part of *Q1: What constitutes a good solubility
method — data, representation, or model?* by **fixing the model** to
a single tuned LightGBM and **varying the molecular representation**.

## Hypothesis

If accuracy is dominated by representation, swapping featurizers under a
fixed architecture should produce a clear, ordered separation in test
RMSE/MAE.  If accuracy is dominated by the model, then with a strong
estimator (LightGBM) every reasonable representation should land in a
similar band, and the gap to the deep models studied in the data-scaling
ablation must come from the model class, not the features.

## Design

| Knob | Value |
|------|-------|
| Model | **LightGBM** |
| HPs | `configs/best_hps.json["lgb_rdkit"]` (held constant) |
| Early stopping | 50 rounds on the full `eval` split |
| Eval splits | `eval`, `ood`, `sc3_gold`, `sc3_silver`, `sc3_bronze` |
| Metrics | RMSE, MAE, R^2, PS-RMSE, Z-RMSE (SC3 tiers), f_aleatoric |

## Featurizers compared

All featurizers go through the same `build_features()` pipeline (concat
solute + solvent + 4 temperature features), so every model sees a
solute-solvent-T input vector — only the chemistry encoding changes.

| Key | Description | Family | Total dims (with solv+T) |
|------|-------------|--------|--------------------------|
| `rdkit`        | 158 RDKit 2D descriptors                       | Descriptor       | ~320 |
| `morgan`       | 1024-bit Morgan ECFP4 (radius 2)               | Fingerprint      | ~2052 |
| `dissolvr`     | RDKit + MOSE topology + Joback thermo + Abr.    | Domain hybrid    | ~356 |
| `mordred`      | ~1600 Mordred 2D descriptors                   | Descriptor (rich)| ~3226 |
| `maccs`        | 167-bit MACCS substructure keys                | Fingerprint (sub)| ~338 |
| `atompair`     | 1024-bit Atom-Pair fingerprint (RDKit)         | Fingerprint (top)| ~2052 |
| `abraham_only` | 6 Abraham/Joback proxies (A,B,S,E,V,Tm)        | Pure domain      | ~16 |

The 7 featurizers span four distinct families (descriptor / fingerprint /
domain / minimal-domain) and three orders of magnitude in dimensionality
(16 → 3226), giving a useful spread.

## How to run

```bash
cd /DATATWO/users/solubility/Solubility/vansh/Ablations/Representation

# 0. (one-time) Build feature caches if you haven't already.
cd /DATATWO/users/solubility/Solubility/vansh
python sc3 cache --featurizers maccs atompair abraham_only
cd Ablations/Representation

# 1. Smoke test (rdkit only, one seed)
python run_representation.py --featurizers rdkit --seeds 42

# 2. Full sweep with single seed (default; ~10-15 min total)
python run_representation.py

# 3. Publication run (5 seeds for tight error bars)
python run_representation.py --seeds 42 101 123 456 789

# 4. Resume / extend (existing JSONs are skipped unless --force)
python run_representation.py --force
```

`run_representation.py` is **resumable**: existing per-seed JSONs are
skipped unless you pass `--force`.  You can therefore add seeds (or
featurizers) incrementally without losing earlier work.

## How to plot

```bash
python make_plots.py                          # default RMSE
python make_plots.py --metric MAE             # MAE bars instead
python make_plots.py --metrics RMSE MAE R2 PS_RMSE   # write all four metrics
```

Output files:

- `figures/representation_grouped_<metric>.png`  All splits, one bar per
  featurizer per split.
- `figures/representation_panel_<metric>.png`     Per-split bar charts
  side-by-side, featurizers ranked left → right by performance.
- `results/representation_table_<metric>.csv`     Long-format table
  (one row per `(featurizer, split)` pair).

## Output layout

```
Ablations/Representation/
    run_representation.py
    make_plots.py
    README.md
    results/
        rdkit/
            seed_42.json
            ...
            summary.json
        morgan/
            ...
        dissolvr/  ...
        mordred/   ...
        maccs/     ...
        atompair/  ...
        abraham_only/  ...
        representation_table_<metric>.csv
    figures/
        representation_panel_<metric>.png
        representation_grouped_<metric>.png
    logs/
```

## Notes

- All representations early-stop LightGBM on the **full** `eval` split
  with the same patience (50 rounds), so the only thing changing between
  runs is the input feature matrix.
- We deliberately reuse the `lgb_rdkit` HPs without per-featurizer
  tuning.  This keeps the comparison clean ("same model"), at the cost
  of slightly handicapping representations whose optimal HPs differ
  (e.g. fingerprints might prefer different `num_leaves` /
  `min_child_samples`).  If you want a per-featurizer tuned variant,
  pass `--hp-key <other_method>` per run, or extend the runner to read
  from a per-featurizer HP table.
- For the SC3 tiers (gold/silver/bronze), `Z_RMSE` and `f_aleatoric` use
  the per-row consensus uncertainty `sigma`, so they're directly
  comparable to the values in the main `sc3_bench` benchmark table.
