# Phase 8: Model Analysis and Interpretability

**Goal:** Understand WHAT the models learn, not just HOW WELL they predict. Adds scientific depth beyond mere benchmarking.

**Completion Criterion:** At least 3 distinct interpretability analyses completed across 2+ model families. Figures produced. Insights documented.

**Output:** `sc3-benchmark/reports/phase_08_interpretability.md`

**Depends on:** Phase 7 complete (needs trained models and predictions)

---

## Step 8.1: Feature Importance Analysis (Tree-Based)

- [ ] Extract SHAP values from best tree-based method (likely CatBoost/DISSOLVR)
- [ ] Plot SHAP summary: which features matter most?
- [ ] Does alignment with chemical intuition hold?
- [ ] Compare feature importance across solvents (polar vs. nonpolar)
- [ ] Compare feature importance on SC3-Easy vs. SC3-Hard

## Step 8.2: Graph Explainability (GNN)

- [ ] For best GNN: use gradient-based or attention-based explainability
- [ ] GNNExplainer or similar for important substructures
- [ ] Attention weights from GAT (if applicable)
- [ ] Select 5-10 representative molecules across solubility spectrum
- [ ] Do highlighted substructures match known solubility-relevant functional groups?

## Step 8.3: Error Analysis Across Model Families

- [ ] Systematic bias check: consistently too high/low for a solvent?
- [ ] High-error outliers (> 2 log S error) — what's special about those molecules?
- [ ] Correlation with molecular properties: error vs. MW, rotatable bonds, structural novelty
- [ ] Tanimoto similarity of SC3 molecules to nearest training neighbor → plot error vs. similarity (OOD degradation)

## Step 8.4: Physics-Consistency Checks

- [ ] For temperature-aware models: does predicted solubility increase monotonically with T? (endothermic dissolution)
- [ ] Plot predicted vs. actual solubility vs. temperature for 10 representative systems
- [ ] Note: only DISSOLVR enforces monotonic constraints — show whether others violate this

## Step 8.5: Report

- [ ] SHAP analysis figures and discussion
- [ ] GNN explainability visualizations
- [ ] Error analysis across model families
- [ ] Physics-consistency results
- [ ] Synthesis: general lessons about model design

---

## Key Notes for Agent

- SHAP can be slow on large datasets — sample if needed but document the sampling
- The Dissolvr `explainer/` module may have relevant code
- Physics-consistency is a strong selling point of the paper
