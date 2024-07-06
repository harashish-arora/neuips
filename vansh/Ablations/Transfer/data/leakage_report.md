# Pair-level leakage check (Transfer ablation)

Pretraining sources (CombiSolv-QM, CombiSolv-Exp) are checked against
the union of canonical `(solute, solvent)` pairs appearing in the SC3
v2 holdouts: `bench_eval`, `bench_ood`, `sc3/gold`, `sc3/silver`,
`sc3/bronze`.  Single-side overlap (same solute, different solvent
or vice versa) is *not* counted as leakage.

## Holdout pair counts

| Split | Rows | Unique canonical pairs |
|-------|-----:|----------------------:|
| eval | 6,969 | 771 |
| ood | 11,940 | 1,445 |
| gold | 4,507 | 335 |
| silver | 5,475 | 400 |
| bronze | 6,331 | 469 |
| **Union** |  | **2,685** |

## Pretraining source overlap

| Source | Rows | Unique solutes | Unique solvents | Overlapping rows | Overlapping pairs |
|--------|-----:|---------------:|----------------:|-----------------:|------------------:|
| CombiSolv-QM | 999,743 | 11,029 | 284 | 113 | 113 |
| CombiSolv-Exp | 8,677 | 1,297 | 275 | 60 | 60 |

Note: the source files were already pair-level cleaned in
`Solubility/sc3-benchmark/Additional_Experiments/transfer_v2/`.
Any non-zero overlap here would mean that the new SC3 v2 splits
(gold/silver/bronze) introduced pairs that were not present in the
old test-tier definition used for the original cleaning.  Those are
removed in the trainer at load time (`load_combisolv(filter_pairs=...)`).
