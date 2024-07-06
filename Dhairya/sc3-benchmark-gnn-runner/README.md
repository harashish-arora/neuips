# SC³ — Third Solubility Challenge

A multi-solvent solubility benchmark with calibrated aleatoric ground truth, built from 101,535 experimental measurements spanning 1,311 solutes, 204 solvents, and 1,494 literature sources.

**Paper:** `paper/main.pdf` (32 pages, NeurIPS 2026 Datasets & Benchmarks format)

---

## Quick Start

```bash
# 1. Proxy (required for internet on hulk)
source ~/proxy_on.sh
# If proxy session expired, re-authenticate:
cd ~ && python3 iitdproxy.py proxyAuth.txt
# Password: 6b4eb7b0 (IDP: cs5230804@iitd.ac.in, category: dual)
# Password: 6b4eb7b0 (IDP: cs5230804@iitd.ac.in, category: dual)

# 2. Conda environment
eval "$(/opt/anaconda/anaconda3/bin/conda shell.bash hook)"
conda activate ~/myenv

# 3. Run benchmark
python sc3 list                        # show all methods & status
python sc3 run --methods catboost_rdkit # run a specific method
python sc3 run --missing               # run only methods with no results
python sc3 run --all --quick           # smoke test (1 seed, 30 epochs)
python sc3 status                      # check results table
python sc3 collect                     # aggregate → CSV + LaTeX
```

### Proxy Details

The cluster uses IIT Delhi's proxy. Three files in `~/`:
- `proxy_on.sh` — sets `http_proxy`/`https_proxy` to `proxy62.iitd.ernet.in:3128`
- `iitdproxy.py` — authenticates with the proxy (asks for password interactively)
- `proxyAuth.txt` — contains `cs5230804 dual`

Conda and pip are pre-configured (`.condarc` and `.config/pip/pip.conf`) to use the proxy with trusted hosts.

---

## Directory Structure

```
sc3-benchmark/
│
├── sc3                          # Main CLI tool (single-file runner)
├── requirements.txt             # Python dependencies
├── setup_cluster.sh             # One-shot cluster setup script
│
├── data/                        # All data files
│   ├── raw/                     #   Raw BigSolDB v2.1 (112K rows)
│   ├── intermediate/
│   │   └── bigsoldb_cleaned.csv #   After cleaning pipeline (101K rows)
│   ├── clean/
│   │   ├── train.csv            #   Training pool (68,607 rows)
│   │   └── val.csv              #   Validation pool (12,034 rows)
│   ├── splits/
│   │   ├── bench_train.csv      #   Benchmarking train (61,752 rows, top-25 solvents)
│   │   ├── bench_eval.csv       #   Benchmarking eval (6,835 rows, held-out solutes)
│   │   └── bench_ood.csv        #   Solvent-OOD test (12,054 rows, 154 solvents)
│   ├── sc3/
│   │   ├── sc3_hard.csv         #   SC³-Hard tier (2,286 rows, MAE ≤ 0.1)
│   │   ├── sc3_medium.csv       #   SC³-Medium tier (3,126 rows, MAE ≤ 0.2)
│   │   └── sc3_easy.csv         #   SC³-Easy tier (4,092 rows, MAE ≤ 0.5)
│   └── dataset_statistics.json  #   Summary statistics for all splits
│
├── src/                         # Source code
│   ├── data/
│   │   ├── clean_bigsoldb.py    #   Phase 2: cleaning pipeline
│   │   └── build_sc3.py         #   Phase 4: SC³ tier construction
│   ├── analysis/
│   │   ├── source_analysis.py   #   Phase 3: copycat detection, DOI ranking
│   │   ├── aleatoric_deep.py    #   Phase 4: aleatoric deepdive (analyses A-O)
│   │   ├── eda_distributions.py #   Phase 2: EDA
│   │   ├── eda_sources.py       #   Phase 2: source coverage EDA
│   │   └── eda_smiles.py        #   Phase 2: SMILES analysis
│   ├── benchmarks/
│   │   ├── data_splits.py       #   Top-25 solvent split logic
│   │   ├── evaluate.py          #   RMSE, PS-RMSE, Z-RMSE computation
│   │   ├── pipeline.py          #   Unified training/eval pipeline
│   │   ├── featurizers/
│   │   │   ├── rdkit_featurizer.py    # ~158 RDKit 2D descriptors
│   │   │   ├── morgan_featurizer.py   # Morgan ECFP4 1024-bit
│   │   │   ├── mordred_featurizer.py  # ~1600 Mordred descriptors
│   │   │   └── dissolvr_featurizer.py # MOSE + Joback + Abraham + RDKit
│   │   └── methods/
│   │       ├── analytical.py      # GSE, ESOL
│   │       ├── sklearn_models.py  # RF, XGB, LGB, CatBoost, MLP, DT, Dissolvr
│   │       ├── abraham.py         # Abraham LFER, Abraham ML
│   │       ├── unifac_method.py   # UNIFAC, UNIFAC+ML
│   │       ├── fastprop.py        # FastProp (PyTorch MLP + BatchNorm)
│   │       ├── fastsolv.py        # FastSolv (Sobolev gradient regularization)
│   │       ├── tayyebi.py         # Tayyebi (Mordred + RF)
│   │       ├── gnn_models.py      # GCN, GAT, GIN, SolubNet, dual-encoder
│   │       ├── rilood.py          # RiLOOD (CIGIN + Set2Set)
│   │       ├── chemprop_method.py # Chemprop D-MPNN wrapper
│   │       ├── solvaformer.py     # Solvaformer (SE(3)-equivariant)
│   │       ├── soltrannet.py      # SolTranNet (char-level transformer)
│   │       ├── gp_tanimoto.py     # GP + Tanimoto kernel (GPyTorch)
│   │       ├── unimol_method.py   # Uni-Mol2 fine-tuning
│   │       ├── chemfm.py          # ChemFM (pending)
│   │       └── base.py            # BaseMethod with save/load
│   └── figures/
│       ├── style.py               # Shared figure style (colors, fonts, sizes)
│       └── paper/
│           └── generate_all.py    # Generate all main-body figures
│
├── results/                     # Per-method results (31 methods)
│   └── <method_name>/
│       ├── summary.json         #   Aggregated metrics (mean ± std)
│       ├── raw_results.json     #   Per-seed, per-split metrics
│       └── models/              #   Saved trained models (.pkl)
│           ├── seed_42.pkl
│           ├── seed_101.pkl
│           ├── seed_123.pkl
│           ├── seed_456.pkl
│           └── seed_789.pkl
│
├── figures/                     # All generated figures
│   ├── paper/                   #   Main-body figures (PDF + PNG)
│   │   ├── fig1_copycat.pdf     #     Copycat histogram
│   │   ├── fig34_aleatoric.pdf  #     Threshold sensitivity + per-solvent
│   │   ├── fig5_cross_db.pdf    #     Cross-database validation
│   │   ├── fig6_solvent_dist.pdf #    Solvent distribution shift
│   │   ├── fig_main_results.pdf #     Main results bar chart
│   │   └── fig_tier_degradation.pdf # Tier degradation (moved to appendix)
│   ├── aleatoric_deepdive/      #   Extended aleatoric analysis
│   │   ├── eps_vs_threshold.pdf #     ε vs θ sweep
│   │   ├── F_bootstrap_distributions.pdf
│   │   ├── H_distribution_qq_plots.pdf
│   │   ├── I_copycat_impact.pdf
│   │   ├── K_complexity_vs_error.pdf
│   │   ├── temperature_monotonicity.pdf
│   │   └── deepdive_results.json #    All numerical results
│   ├── eda/                     #   Exploratory data analysis (12 figures)
│   ├── source_analysis/         #   Source analysis (7 figures)
│   ├── aleatoric/               #   Aleatoric analysis (7 figures)
│   └── dataset_construction/    #   Dataset splits (4 figures)
│
├── reports/                     # Phase reports and analysis
│   ├── phase_01_data_inventory.md     # Phase 1: raw data audit
│   ├── phase_02_merging.md            # Phase 2: cleaning pipeline
│   ├── phase_02_eda_findings.md       # Phase 2: EDA findings
│   ├── phase_03_source_analysis.md    # Phase 3: copycats, DOI ranking
│   ├── phase_05_multimodality.md      # Phase 5: multimodality + metrics
│   ├── phase_06_literature_survey.md  # Phase 6: method taxonomy
│   ├── phase_07_analysis_plan.md      # Phase 7: analysis plan
│   ├── benchmark_results.md           # Aggregated benchmark results
│   ├── new_methods_report.md          # GP, Abraham, UNIFAC report
│   ├── phase_03_artifacts/            # Source analysis data
│   │   ├── pairwise_maes.csv          #   735 pairwise comparisons
│   │   ├── interlab_variability.csv   #   610 inter-lab pairs
│   │   ├── doi_reliability.csv        #   369 ranked DOIs
│   │   ├── exact_duplicates.csv       #   14 exact-duplicate pairs
│   │   ├── near_duplicates.csv        #   125 near-duplicate pairs
│   │   ├── aleatoric_limits.json      #   Stratified aleatoric limits
│   │   └── summary.json              #   All Phase 3 statistics
│   └── phase_04_aleatoric/            # Aleatoric analysis data
│       ├── aleatoric_analysis.json
│       ├── direct_comparisons.csv     #   5,042 direct comparisons
│       └── temperature_monotonicity.csv
│
├── paper/                       # NeurIPS 2026 LaTeX paper
│   ├── main.tex                 #   Master document
│   ├── main.pdf                 #   Compiled paper (32 pages)
│   ├── 00.packages.tex          #   LaTeX packages
│   ├── 00.macros.tex            #   Custom macros (SC³, ε_A, etc.)
│   ├── 00.metadata.tex          #   Title, authors
│   ├── references.bib           #   Bibliography
│   ├── neurips_2026.sty         #   NeurIPS style file
│   ├── sections/
│   │   ├── 00.abstract.tex      #   Abstract
│   │   ├── 01.introduction.tex  #   Introduction
│   │   ├── 02.related_work.tex  #   Related work
│   │   ├── 03.data_curation.tex #   §3: Data curation
│   │   ├── 04.aleatoric_theory.tex # §4: Aleatoric limit
│   │   ├── 05.benchmark_design.tex # §5: Benchmark design + metrics
│   │   ├── 06.metrics.tex       #   (empty — folded into §5)
│   │   ├── 07.experiments.tex   #   §7: Experiments
│   │   ├── 08.results.tex       #   §8: Results and discussion
│   │   ├── 09.discussion.tex    #   (empty — folded into §8)
│   │   ├── 10.conclusion.tex    #   §10: Conclusion
│   │   └── appendix.tex         #   Appendix A-H (662 lines)
│   └── tables/
│       └── main_results.tex     #   Main benchmark table (32 methods)
│
├── scripts/                     # Runner scripts
│   ├── run_all.sh               #   Run everything
│   ├── run_cpu_baselines.py     #   CPU methods
│   ├── run_gnn_baselines.py     #   GNN methods
│   ├── run_morgan_baselines.py  #   Morgan fingerprint variants
│   ├── run_transformer_baselines.py
│   ├── run_parallel.py          #   Parallel CPU execution
│   ├── collect_results.py       #   Aggregate results → CSV/LaTeX
│   └── analyze_results.py       #   Analysis and figures
│
└── Plan/                        # Project management
    ├── STATUS.md                #   Master status tracker
    ├── AGENT_START_HERE.md      #   Agent onboarding
    └── phase_01-09/             #   Per-phase plans and progress
```

---

## Benchmark Results (31 methods, 5 seeds each)

| Rank | Method | Family | Hard PS-RMSE |
|------|--------|--------|-------------|
| 1 | LightGBM (RDKit) | Desc+Tree | **0.659** |
| 2 | Abraham ML | Desc+Tree | 0.662 |
| 3 | CatBoost (RDKit) | Desc+Tree | 0.672 |
| 4 | Dissolvr | Domain+Tree | 0.680 |
| 5 | XGBoost (RDKit) | Desc+Tree | 0.682 |
| ... | | | |
| 10 | Chemprop | D-MPNN | 0.802 |
| 11 | Uni-Mol2 | Foundation | 0.805 |
| 15 | SolTranNet | Transformer | 0.864 |
| 20 | GCN | GNN | 0.888 |
| 26 | UNIFAC | Physics | 1.087 |
| 31 | GSE | Analytical | 1.980 |

Full results in `results/` and `paper/tables/main_results.tex`.

---

## Key Findings

1. **Aleatoric limit:** Median inter-lab MAE = 0.058 log S (after copycat exclusion at θ=0.01). The commonly cited 0.6–0.8 corresponds to P90–P95 of a heavy-tailed distribution, not the typical case.

2. **Representation > architecture:** RDKit 2D descriptors + gradient-boosted trees outperform Morgan fingerprints (19–27% gap), GNNs (22–58% gap), and billion-parameter foundation models (22% gap).

3. **All models far from solved:** Best Z-RMSE ≈ 49 on Hard tier (49× above measurement noise). Substantial room for improvement.

---

## Phase Overview

| Phase | Status | Output |
|-------|--------|--------|
| 1. Data Inventory | ✅ | `reports/phase_01_data_inventory.md` |
| 2. Cleaning + EDA | ✅ | `data/intermediate/bigsoldb_cleaned.csv`, `reports/phase_02_*.md` |
| 3. Source Analysis | ✅ | `reports/phase_03_artifacts/`, copycats, DOI rankings |
| 4. Dataset Construction | ✅ | `data/sc3/`, `data/splits/`, aleatoric analysis |
| 5. Multimodality | ✅ | `reports/phase_05_multimodality.md`, PS-RMSE justification |
| 6. Literature Survey | ✅ | `reports/phase_06_literature_survey.md` |
| 7. Benchmarking | ✅ | 31/32 methods complete (ChemFM pending) |
| 8. Interpretability | 🔲 | Planned |
| 9. Paper | 🟡 | Draft complete (`paper/main.pdf`, 32 pages) |

---

