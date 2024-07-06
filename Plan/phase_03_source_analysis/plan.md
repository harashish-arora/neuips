# Phase 3: Source Analysis and Inter-Lab Variability

**Goal:** Understand which sources are reliable, which are duplicates, and quantify the true inter-lab variability. This is the scientific heart of the paper.

**Completion Criterion:** Clear, quantified statement of the aleatoric limit for different data quality tiers. Source reliability scores computed for all DOIs. Confident in numbers before moving on.

**Output:** `sc3-benchmark/reports/phase_03_source_analysis.md`

**Depends on:** Phase 2 complete

---

## Step 3.1: Group by Solute-Solvent-Source

- [ ] For each unique (solute, solvent) pair, identify all sources (DOIs)
- [ ] For each (solute, solvent, source) triple, list all temperature-solubility data points
- [ ] Statistics: how many pairs have ≥2 independent sources? ≥3? ≥5?

## Step 3.2: Detect Intra-Lab Duplication ("Copycat Problem")

**This is a critical discovery — one of the paper's key contributions.**

- [ ] Check if any two sources report identical values (to 3 decimal places) at identical temperatures → flag as exact duplicates
- [ ] Check if sources share authors (CrossRef API via `habanero` or direct API) → same-author = intra-lab, not inter-lab
- [ ] Compute pairwise MAE between all sources per (solute, solvent) pair
- [ ] If MAE < 0.01 log S units → flag as suspected duplicate
- [ ] Count: how many multi-source groups have variance < 0.01?
- [ ] Document the fraction of apparent inter-lab agreement that is artificial

## Step 3.3: Fit Thermodynamic Curves per Source

For each (solute, solvent, source) triple with sufficient temperature data:
- [ ] ≥3 temperature points → fit Apelblat equation: `ln S = A + B/T + C·ln(T)` using `scipy.optimize.curve_fit`
- [ ] = 2 temperature points → fit van't Hoff equation: `ln S = A + B/T` (exact with 2 points)
- [ ] = 1 temperature point → mark as "isolated" (still valuable for training, can't compare at arbitrary T)
- [ ] For each fit, compute 95% confidence interval from covariance matrix

**Note:** Dissolvr's `apelblat/` folder likely has existing curve-fitting code to reuse.

## Step 3.4: Compute True Inter-Lab Variability

For each (solute, solvent) pair with ≥2 truly independent sources:
- [ ] Choose reference temperatures T_ref where multiple sources have data or reliable interpolation (CI width < 0.3 log S)
- [ ] Only compare sources with temperature gap ≥ 2.0 K between actual measurements
- [ ] Compute inter-lab MAE, RMSE, std dev at each T_ref
- [ ] Aggregate into global aleatoric limit: `ε_aleatoric = √(ε_direct² + 2δ_interp²)`

## Step 3.5: Source Reliability Ranking

- [ ] For each DOI, compute average deviation from consensus across all overlapping (solute, solvent) pairs
- [ ] Rank DOIs by reliability
- [ ] Identify "Hall of Shame" — DOIs with > 0.6 log S MAE from consensus
- [ ] Identify "Hall of Fame" — consistently low-error DOIs
- [ ] Produce reliability histogram and ranked table

## Step 3.6: Stratified Aleatoric Limit

Compute aleatoric limit separately for:
- [ ] Easy pairs (≥5 independent sources, good fits, small CIs) — expect ~0.3 log S
- [ ] Medium pairs (3-4 independent sources) — expect ~0.5 log S
- [ ] Hard pairs (2 independent sources, wider CIs) — expect 0.5-0.8 log S
- [ ] Compare against literature: Palmer & Mitchell (2014) 0.6-0.7, Attia et al. (2025) 0.5-1.0, Llompart et al. (2024) curation issues

**Key insight:** The commonly cited 0.6-0.8 is an AVERAGE. Well-curated data from reliable sources has a much lower floor (~0.3).

## Step 3.7: Report

- [ ] Copycat/duplication discovery with hard numbers
- [ ] Distribution of inter-lab MAE for truly independent sources
- [ ] Source reliability ranking (top 20 best, top 20 worst)
- [ ] Stratified aleatoric limit estimates
- [ ] Literature comparison
- [ ] Detailed methodology (equations, thresholds, justifications)
- [ ] All figures → `sc3-benchmark/figures/`

---

## Key Notes for Agent

- This is the most scientifically important phase — take time to get it right
- The copycat detection is novel and must be documented carefully
- Reuse Apelblat fitting code from `Dissolvr/apelblat/` where possible
- The aleatoric limit numbers will directly inform Phase 4's cleaning thresholds
