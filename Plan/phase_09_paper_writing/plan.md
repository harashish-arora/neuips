# Phase 9: Paper Writing

**Goal:** Organize all findings into a coherent NeurIPS Datasets & Benchmarks paper.

**Completion Criterion:** Complete paper draft with all figures and tables. Coherent story. Every number traceable to a report.

**Output:** `sc3-benchmark/paper/sc3_paper.tex`

**Depends on:** All prior phases complete

---

## Step 9.1: NeurIPS D&B Format

- [ ] Look up submission guidelines (page limit, required sections, review criteria)
- [ ] Typical structure:
  1. Introduction (problem significance, gap)
  2. Related Work (datasets, methods, evaluations)
  3. The Dataset (construction, statistics, quality guarantees)
  4. Benchmark Protocol (methods, metrics, setup)
  5. Results and Analysis (tables, ablations, interpretability)
  6. Discussion and Recommendations
  7. Broader Impact / Limitations

## Step 9.2: Draft the Paper Narrative

Write `sc3-benchmark/paper/narrative_outline.md`:
1. Solubility prediction matters but is limited by data quality
2. The field has a benchmarking crisis (inflated performance, leakage, misleading metrics)
3. We rigorously quantify the problem (copycats, inter-lab variability, stratified aleatoric limits)
4. Solubility is multimodal — standard metrics fail — we propose better ones
5. SC3: quality-stratified challenge with known ground truth quality
6. Fair benchmarking: X methods, identical conditions, key finding: [whatever data shows]
7. What models actually learn: interpretability insights
8. Recommendations for dataset creators, model builders, benchmark designers

## Step 9.3: Prepare All Figures

Publication-quality (matplotlib, consistent style, 300 DPI, colorblind-friendly):
- [ ] Dataset construction pipeline (flowchart)
- [ ] Copycat discovery (before/after cleaning)
- [ ] Inter-lab variability distribution
- [ ] Aleatoric limit by data quality tier
- [ ] Multimodality: distributions across solvents
- [ ] Why RMSE fails
- [ ] Main results (radar plot or grouped bar chart)
- [ ] Per-solvent performance heatmap
- [ ] Data quality ablation
- [ ] SHAP feature importance
- [ ] GNN explainability
- [ ] Temperature-consistency check

Save to `sc3-benchmark/figures/paper/` as .png (300 DPI) and .pdf

## Step 9.4: Prepare All Tables

- [ ] Compile final tables in LaTeX format in `sc3-benchmark/tables/paper/`

## Step 9.5: Write the Paper

Using NeurIPS LaTeX template:
- [ ] Main paper within page limit (~9 pages for D&B)
- [ ] Appendix: supplementary tables, additional figures, full method descriptions, hyperparameters, per-solvent results, DOI reliability ranking
- [ ] Every claim supported by a number
- [ ] Every citation verified
- [ ] Cross-reference with DISSOLVR paper

## Step 9.6: Final Check

- [ ] Re-run key experiments for reproducibility
- [ ] Check all numbers against raw results
- [ ] Verify no data leakage
- [ ] Proofread

---

## Key Notes for Agent

- The paper is the deliverable — everything else serves this
- "Dataset and benchmark ARE the contribution" — not a new model
- Cross-reference DISSOLVR carefully (same research group)
