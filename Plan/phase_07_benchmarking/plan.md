# Phase 7: Benchmarking — Training and Evaluation

**Goal:** Train all methods on clean training data and evaluate on SC3 challenge sets. Largest phase in compute time.

**Completion Criterion:** All methods trained and evaluated. All tables produced. Results reproducible (configs saved). No obvious pipeline errors.

**Output:** `sc3-benchmark/reports/phase_07_benchmarking.md`

**Depends on:** Phases 4, 5, 6 complete

---

## Step 7.1: Experimental Protocol (Define BEFORE running)

- [ ] Document exact protocol:
  - Training data: `sc3-benchmark/data/clean/train.csv`
  - Validation data: `sc3-benchmark/data/clean/val.csv`
  - Test data: SC3-Easy, SC3-Medium, SC3-Hard
  - Seeds: 5 random seeds per method, report mean ± std
  - Hyperparameter tuning: reasonable search on validation set, start from paper's recommended
  - Time tracking: wall-clock training + inference time, hardware specs
  - Metrics: Standard (RMSE, MAE, R², MAPE) + New (Z-RMSE, Z-MAE, Z-R², PS-RMSE) + Per-solvent RMSE (top 10 solvents)

## Step 7.2: Run Methods (in order, simplest first)

1. [ ] GSE (analytical, no training)
2. [ ] ESOL model (analytical/regression, very fast)
3. [ ] Random Forest on DISSOLVR features
4. [ ] XGBoost on DISSOLVR features
5. [ ] LightGBM on DISSOLVR features
6. [ ] CatBoost on DISSOLVR features (≈ DISSOLVR Regime I)
7. [ ] MLP on DISSOLVR features
8. [ ] DISSOLVR full (Interaction Layer for Regime II)
9. [ ] Chemprop
10. [ ] MolMerger
11. [ ] FastSolv (if CPU-feasible)
12. [ ] GCN, GAT, GIN, MPNN
13. [ ] Any additional methods from Phase 6

For each method:
- [ ] Data adapter script for expected format
- [ ] Train with validation-based early stopping
- [ ] Generate predictions on ALL test sets
- [ ] Compute all metrics
- [ ] Save predictions to `sc3-benchmark/tables/<method>_predictions.csv`
- [ ] Save trained model if possible
- [ ] Record issues/warnings/observations

## Step 7.3: Ablation Studies

**Feature ablation (best tree-based method):**
- [ ] Remove each feature category one at a time (Compositional, Topological, Energetic, Physicochemical)
- [ ] Use ONLY each category one at a time

**Training data quality ablation:**
- [ ] Train on full uncleaned BigSolDB → test on SC3
- [ ] Compare against cleaned dataset training
- [ ] Hypothesis: cleaned data → better SC3 performance despite smaller training set

**Solvent-specific fine-tuning:**
- [ ] For top 5 solvents in SC3, fine-tune best general model on that solvent's data
- [ ] Compare general vs. fine-tuned

**Transfer learning across solvents:**
- [ ] Train on top K solvents, test on rare solvents, vary K
- [ ] Show generalization gap

## Step 7.4: Results Tables

Save as CSV + LaTeX in `sc3-benchmark/tables/`:
- [ ] **Main results:** All methods × all SC3 sets × primary metrics (RMSE, Z-RMSE, R², Z-R²) + timing
- [ ] **Per-solvent results:** Top 10 solvents × RMSE per method
- [ ] **Feature ablation table**
- [ ] **Data quality ablation table**
- [ ] **Fine-tuning table**

## Step 7.5: Report

- [ ] All tables with analysis
- [ ] Key findings: best methods, deep vs. simple tradeoff, data cleaning impact
- [ ] Surprises/unexpected results
- [ ] Fair comparison commentary

---

## Key Notes for Agent

- Run simplest methods first to catch pipeline bugs
- CPU-only: budget time accordingly, some models may take hours
- Save all configs to `sc3-benchmark/configs/` for reproducibility
- Every metric computation should use the functions from `sc3-benchmark/src/metrics/` (Phase 5)
