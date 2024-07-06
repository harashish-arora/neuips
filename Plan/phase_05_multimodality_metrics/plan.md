# Phase 5: Multimodality Analysis and New Metrics

**Goal:** Demonstrate that solubility is a multimodal regression problem, that standard RMSE/R² are misleading, and propose better metrics.

**Completion Criterion:** Clear evidence that solubility is multimodal. At least 2-3 new metrics formally defined with justification. Can explain in one paragraph why standard RMSE is insufficient.

**Output:** `sc3-benchmark/reports/phase_05_multimodality_metrics.md`

**Depends on:** Phase 4 complete (needs the clean dataset)

---

## Step 5.1: Solubility Distribution by Solvent

- [ ] For each solvent, plot distribution of log S values across all solutes
- [ ] Compute mean and variance of log S per solvent
- [ ] Show different solvents have dramatically different mean solubilities
- [ ] Fit Gaussian Mixture Model to overall distribution — report number of components, BIC
- [ ] Check for bimodality WITHIN individual solvents (polar vs. nonpolar solutes)

## Step 5.2: Why RMSE and R² Are Misleading

Concrete demonstration:
- [ ] "Solvent-mean baseline": predict mean log S for each solvent. Compute RMSE and R² — show deceptively high R²
- [ ] Show a model can have excellent overall RMSE but terrible per-solvent performance
- [ ] Compare two hypothetical models: uniform 0.5 error vs. 0.1 on common / 2.0 on rare — similar RMSE, wildly different utility

## Step 5.3: Solvent Similarity and Solubility Correlation

- [ ] For solvent pairs tested on the same solute, compute log S correlation
- [ ] Build solvent-solvent similarity matrix from solubility correlation
- [ ] Compare against chemical similarity (Tanimoto on fingerprints, Hansen parameter distance)
- [ ] Quantify "like dissolves like" at scale
- [ ] Assess feasibility of solvent-transfer learning

## Step 5.4: Define New Metrics

Implement in `sc3-benchmark/src/metrics/`:

- [ ] **Z-RMSE** (Solvent-Normalized RMSE): `z_i = (ŷ_i - y_i) / σ_s`, `Z-RMSE = √(mean(z_i²))`
- [ ] **Z-MAE**: `Z-MAE = mean(|z_i|)`
- [ ] **Z-R²**: R² after z-score normalization within each solvent
- [ ] **PS-RMSE** (Per-Solvent Averaged RMSE): `(1/|S|) Σ_s RMSE_s`
- [ ] **MAPE**: `mean(|ŷ_i - y_i| / |y_i|) × 100` — handle near-zero carefully (exclude |log S| < 0.1 or use SMAPE)
- [ ] **W-RMSE** (Weighted by aleatoric limit): `√(Σ w_i(ŷ_i - y_i)² / Σ w_i)`, `w_i = 1/ε_i²`

## Step 5.5: Validate New Metrics

- [ ] Demonstrate on synthetic data that Z-RMSE and PS-RMSE distinguish good from bad models better than RMSE
- [ ] Show two models with similar RMSE can have very different Z-RMSE
- [ ] Recommend PRIMARY metric for SC3 challenge

## Step 5.6: Report

- [ ] Multimodality analyses with figures
- [ ] RMSE/R² failure demonstration
- [ ] Solvent similarity analysis
- [ ] Formal metric definitions
- [ ] Validation on synthetic data
- [ ] Primary metric recommendation

---

## Key Notes for Agent

- Metrics code should be modular and reusable — Phase 7 will call these functions for every model
- The RMSE failure demonstration is critical for the paper's narrative
- Colorblind-friendly palettes for all figures
