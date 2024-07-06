# SC3: The Third Solubility Challenge — Agent Research Protocol

## Document Purpose

This document is the complete operating manual for an autonomous research agent. The agent will execute a multi-week research program to produce a NeurIPS Datasets & Benchmarks track paper. The paper establishes a new, rigorously cleaned solubility dataset, quantifies the true aleatoric limit of solubility prediction, proposes new evaluation metrics for multimodal regression, and benchmarks all major solubility prediction methods under fair conditions.

The agent should treat this as a research expedition, not a script. At each phase, it must observe, analyze, form hypotheses, and record findings before moving on. Quality and honesty are paramount — this is a benchmarking paper, and every number must be defensible.

---

## Operating Principles

### How the Agent Works

1. **Sequential phases.** The agent works through numbered phases in order. It does NOT skip ahead. Each phase has a clear completion criterion.
2. **Markdown reports.** Every phase produces a report in `/reports/phase_XX_<name>.md`. Reports contain: what was done, what was found, all numerical results in tables, all figures saved to `/figures/`, and a clear statement of completion or uncertainty.
3. **Pass-on reports.** If the agent is underconfident about any result, or if the session ends mid-phase, it writes a pass-on report to `/pass-ons/pass_on_NNN.md` explaining exactly where it stopped, what remains, and what the next agent instance should do first. The next agent instance MUST read the latest pass-on before doing anything else.
4. **Code quality.** All code lives in `/src/` with clear module structure. Code must be runnable from the terminal. No notebooks — only `.py` scripts with argparse interfaces. Every script has a docstring explaining what it does and how to run it.
5. **Honesty over results.** If a method performs poorly, report it honestly. If the agent cannot reproduce a paper's numbers, document what was tried and what went wrong. Never fabricate or cherry-pick.
6. **Figures.** All figures are saved as both `.png` (300 DPI) and `.pdf` to `/figures/`. Every figure has a descriptive filename and a caption written into the phase report.

### Folder Structure

```
/sc3-benchmark/
├── data/
│   ├── raw/                    # Original downloaded datasets
│   ├── intermediate/           # Intermediate cleaning stages
│   ├── clean/                  # Final cleaned training/validation data
│   └── sc3/                    # The SC3 challenge test sets
│       ├── sc3_easy.csv
│       ├── sc3_medium.csv
│       └── sc3_hard.csv
├── src/
│   ├── data/                   # Data loading, cleaning, curation scripts
│   ├── analysis/               # Dataset analysis, aleatoric limit, multimodality
│   ├── metrics/                # New metric implementations (Z-RMSE, Z-MAE, etc.)
│   ├── models/                 # Model training/evaluation wrappers
│   ├── baselines/              # Cloned baseline repos and adapter scripts
│   ├── interpretability/       # Model analysis and explainability
│   └── utils/                  # Shared utilities
├── reports/                    # Phase completion reports (markdown)
├── pass-ons/                   # Handoff reports between agent sessions
├── figures/                    # All generated figures
├── tables/                     # All generated tables (as .csv and .tex)
├── configs/                    # Hyperparameter configs for each method
└── paper/                      # LaTeX paper drafts
```

Create this folder structure as the very first action.

---

## PHASE 1: Data Acquisition and Initial Inventory

**Goal:** Download all required datasets, understand their structure, and produce an initial inventory of what we have.

### Step 1.1: Acquire BigSolDB

- Search for and download BigSolDB 2.0 from its official source. The paper by Krasnov et al. (2025) "BigSolDB 2.0, dataset of solubility values for organic compounds in different solvents at various temperatures" published in Scientific Data provides the dataset. Check Zenodo, Figshare, or the paper's data availability statement for download links.
- If BigSolDB 1.0 is separately available (Krasnov et al. 2023, ChemRxiv), download it too.
- Place raw files in `/data/raw/bigsoldb_v1/` and `/data/raw/bigsoldb_v2/`.
- Record: number of rows, columns, column names, data types, any metadata files.

### Step 1.2: Acquire Additional Datasets

- Download AqSolDB (Sorkun et al. 2019) — aqueous solubility dataset (~9,982 compounds). Available on Harvard Dataverse or the paper's repository.
- Download ESOL (Delaney 2004) — small aqueous benchmark (~1,128 compounds). Available from Chemprop's GitHub or MoleculeNet.
- Download the Second Solubility Challenge dataset (Llinas et al. 2020) — 100 compounds, external validation. Search JCIM supplementary.
- Download the Leeds dataset (Boobier et al. 2020) — organic solvent solubility with physicochemical descriptors. Check Nature Communications supplementary.
- Place each in `/data/raw/<dataset_name>/`.

### Step 1.3: Initial Inventory Report

Write `/reports/phase_01_data_inventory.md` containing:
- For each dataset: source URL, citation, number of entries, number of unique solutes, number of unique solvents, temperature range, solubility unit, any known issues from the literature.
- A comparison table showing overlap potential between datasets.
- Note which datasets have DOI/source information per measurement (critical for later source analysis).

**Completion criterion:** All datasets downloaded, inventory report written with hard numbers.

---

## PHASE 2: Data Merging and Structural Cleaning

**Goal:** Combine BigSolDB 1.0 and 2.0 into a single unified dataset, perform structural cleaning (not yet content-based filtering).

### Step 2.1: Understand the Schema

- Load both BigSolDB versions. Understand column semantics: solute identifiers (SMILES, names), solvent identifiers, temperature, solubility values, units, source DOIs.
- Determine if v2.0 is a strict superset of v1.0 or if they have unique entries.
- Document any schema differences between versions.

### Step 2.2: Merge and Deduplicate

- Concatenate v1.0 and v2.0.
- Deduplicate: identify rows that appear in both versions (same solute SMILES, same solvent, same temperature, same solubility value). Keep one copy.
- Canonicalize all SMILES using RDKit (install if needed: `pip install rdkit-pypi`). Strip stereochemistry for consistency with 2D methods.
- Standardize solvent names using a solvent alias map (e.g., "THF" → "tetrahydrofuran", "DMSO" → "dimethyl sulfoxide"). Build this map from the data itself — look at all unique solvent names and group synonyms.

### Step 2.3: Unit Standardization

- Check what units solubility is reported in. BigSolDB typically uses mole fraction (x). Convert to log₁₀(S) in mol/L using the formula:
  ```
  log S (mol/L) = log₁₀(x · ρ_solvent(T) / MW_solvent)
  ```
- For density at temperature T, use the `thermo` Python library or a lookup table. If `thermo` is not available, `pip install thermo`.
- Cache density calculations to avoid repeated computation.
- Flag any entries where conversion fails (unknown solvent, missing density data).

### Step 2.4: Basic Structural Filters

- Remove entries with "." in SMILES (salts, mixtures, multi-component systems).
- Remove polymeric solvents (PEG variants, Span, PEGDME, etc.) — their molar volume is ill-defined.
- Remove entries with molecular weight > 1000 Da (non-drug-like).
- Apply solubility range filter: keep entries where -15 ≤ log S ≤ 2. Values outside this are likely errors or edge cases.
- Standardize tautomers using RDKit's `rdMolStandardize.TautomerEnumerator`.

### Step 2.5: Remove Known Bad DOIs

Remove all entries from the following DOIs, identified as problematic by the BigSolDB creators:
```
10.1021/acs.jced.4c00179
10.1021/acs.jced.9b00728
10.1021/acs.jced.6b00009
10.1016/j.molliq.2022.119759
10.1016/j.fluid.2011.09.033
10.1016/j.fluid.2013.09.018
10.1016/j.molliq.2013.06.011
10.1016/j.molliq.2020.113867
10.1016/j.fluid.2015.07.038
```

### Step 2.6: Report

Write `/reports/phase_02_merging.md`:
- Number of entries at each cleaning stage (waterfall chart data).
- Number of unique solutes, solvents, DOIs after cleaning.
- Temperature distribution histogram.
- Solubility distribution histogram.
- Any anomalies discovered during cleaning.

**Completion criterion:** A single clean CSV at `/data/intermediate/bigsoldb_merged_clean.csv` with standardized SMILES, standardized solvents, log S in mol/L, temperature in K, and source DOI. Report written.

---

## PHASE 3: Source Analysis and Inter-Lab Variability

**Goal:** Understand which sources are reliable, which are duplicates, and quantify the true inter-lab variability. This is the scientific heart of the paper.

### Step 3.1: Group by Solute-Solvent-Source

- For each unique (solute, solvent) pair, identify all sources (DOIs) that have measured it.
- For each (solute, solvent, source) triple, list all temperature-solubility data points from that source.
- Produce statistics: how many (solute, solvent) pairs have ≥2 independent sources? ≥3? ≥5?

### Step 3.2: Detect Intra-Lab Duplication ("Copycat Problem")

This is a critical discovery. Many apparent "independent" measurements are actually the same data re-reported.

- For each (solute, solvent) pair with multiple sources, check if any two sources report identical values (to 3 decimal places) at identical temperatures. Flag these as exact duplicates.
- Check if sources share authors (if author metadata is available via CrossRef API — use `pip install habanero` or direct API calls). Same-author papers measuring the same system at nearly identical conditions are intra-lab, not inter-lab.
- Compute pairwise MAE between all sources for each (solute, solvent) pair. If MAE < 0.01 log S units, flag as suspected duplicate.
- Count: how many of the multi-source groups have variance < 0.01? What fraction of apparent inter-lab agreement is artificial?
- Document this finding in detail — it is one of the paper's key contributions.

### Step 3.3: Fit Thermodynamic Curves per Source

For each (solute, solvent, source) triple with sufficient temperature-dependent data:

- **≥ 3 temperature points:** Fit the Apelblat equation: `ln S = A + B/T + C·ln(T)`. Use `scipy.optimize.curve_fit`. Record fit parameters and covariance matrix.
- **= 2 temperature points:** Fit the van't Hoff equation: `ln S = A + B/T`. This is exact with 2 points.
- **= 1 temperature point:** Cannot fit a curve. These are "isolated" measurements — still valuable for training but cannot contribute to inter-lab comparison at arbitrary temperatures.

For each fitted curve, compute the 95% confidence interval at each temperature using the covariance matrix from the fit. This replaces arbitrary temperature bounds.

### Step 3.4: Compute True Inter-Lab Variability

For each (solute, solvent) pair where ≥2 TRULY INDEPENDENT sources have fitted curves:

- Choose reference temperatures T_ref where multiple sources have data or reliable interpolation (confidence interval width < threshold, e.g., < 0.3 log S units).
- Only compare sources where the temperature gap between actual measurements is ≥ 2.0 K (to ensure independence — same-day same-equipment measurements often differ by < 1K).
- At each T_ref, compute the inter-lab MAE, RMSE, and standard deviation across independent sources.
- Aggregate these into a global aleatoric limit estimate: ε_aleatoric = √(ε_direct² + 2δ_interp²), where ε_direct is the direct inter-lab disagreement and δ_interp is the interpolation uncertainty from the curve fits.

### Step 3.5: Source Reliability Ranking

- For each DOI, compute its average deviation from the consensus value across all (solute, solvent) pairs where it overlaps with other sources.
- Rank all DOIs by reliability (low deviation = high reliability).
- Identify a "Hall of Shame" — DOIs with consistently high error (> 0.6 log S MAE from consensus). These should be flagged or removed.
- Identify a "Hall of Fame" — DOIs with consistently low error. These are anchor sources.
- Produce a source reliability histogram and a ranked table.

### Step 3.6: The Aleatoric Limit — Stratified Analysis

The aleatoric limit is NOT a single number. It depends on the quality of data.

- Compute the aleatoric limit separately for:
  - Easy pairs (≥5 independent sources, good curve fits, small confidence intervals) — expect ε ≈ 0.3 log S
  - Medium pairs (3-4 independent sources) — expect ε ≈ 0.5 log S
  - Hard pairs (2 independent sources, wider confidence intervals) — expect ε ≈ 0.5-0.8 log S
- Compare your findings against the literature: Palmer & Mitchell (2014) cite 0.6-0.7 for aqueous data from heterogeneous sources. Attia et al. (2025) cite 0.5-1.0 for organic solvents. Llompart et al. (2024) showed that curation issues inflate apparent performance.
- The key insight: the commonly cited 0.6-0.8 is an AVERAGE across all data quality levels. Well-curated data from reliable sources has a much lower floor (~0.3), while poorly curated data has a higher floor.

### Step 3.7: Report

Write `/reports/phase_03_source_analysis.md`:
- The copycat/duplication discovery with hard numbers.
- Distribution of inter-lab MAE for truly independent sources.
- Source reliability ranking (top 20 best, top 20 worst DOIs).
- Stratified aleatoric limit estimates.
- Comparison with literature claims.
- Detailed methodology description (equations, thresholds, justifications).
- All figures saved to `/figures/`.

**Completion criterion:** A clear, quantified statement of the aleatoric limit for different data quality tiers. Source reliability scores computed for all DOIs. The agent is confident in these numbers before moving on.

---

## PHASE 4: Dataset Construction — Clean Train/Val and SC3 Challenge

**Goal:** Using the source analysis from Phase 3, construct (a) a clean training/validation dataset and (b) the SC3 held-out challenge sets.

### Step 4.1: The Two-Pointer Cleaning Algorithm

For each (solute, solvent) group with multiple measurements (across sources and temperatures):

1. Sort all measurements by their predicted value at a reference temperature (using fitted curves or raw values).
2. Use a two-pointer / sliding-window approach to find the LARGEST subset where max(values) - min(values) < threshold ε₀.
3. The threshold ε₀ should be informed by the aleatoric limit from Phase 3 — e.g., ε₀ = the median inter-lab MAE for that data quality tier.
4. Measurements outside this maximal consensus window are outliers. Flag them but don't delete yet — record which source they came from (feeds back into source reliability).
5. For the consensus window, take the median value as the "ground truth."

This is the error-minimizing maximal set: it retains as much data as possible while ensuring internal consistency.

### Step 4.2: Construct the Clean Training Dataset

- Start with all (solute, solvent, temperature, log S) tuples from the merged dataset.
- For tuples from Hall-of-Shame DOIs, remove entirely.
- For tuples with multiple sources, use the consensus value from the two-pointer algorithm.
- For tuples with a single source, keep IF that source has a reliability score above a threshold (e.g., sources whose average inter-lab MAE < 0.5 when they DO overlap with others).
- Apply the anti-leakage protocol: remove any molecules that will appear in the SC3 test sets (defined below).
- Split into train (85%) and validation (15%) using a random split, stratified by solvent to ensure representation.
- Save to `/data/clean/train.csv` and `/data/clean/val.csv`.

### Step 4.3: Construct the SC3 Challenge Sets

The SC3 (Third Solubility Challenge) consists of three difficulty tiers, all held out from training:

**SC3-Easy (~500 data points):**
- Select (solute, solvent) pairs with the MOST inter-lab agreement (≥5 independent sources, consensus MAE < 0.3 log S units).
- These have a very well-defined ground truth. A model's deviation from the consensus is almost entirely model error, not data noise.
- Sample diverse solutes and solvents — avoid over-representing any single chemical family.

**SC3-Medium (~200 data points):**
- Select pairs with moderate agreement (3-4 sources, consensus MAE 0.3-0.5 log S).
- Ground truth is less certain but still defensible.

**SC3-Hard (~100 data points):**
- Select pairs where the aleatoric limit is lowest (best-characterized, highest-agreement data with the tightest error bars).
- This sounds paradoxical — "hard" here means that the dataset has the TIGHTEST ground truth, so it's hardest for a MODEL to get away with sloppy predictions. There's no noise to hide behind.
- Alternatively: include pairs with novel solutes not seen in training (OOD generalization challenge) or rare solvents.

Decide which framing of "hard" is more scientifically interesting. Document the choice.

- Ensure ZERO overlap between SC3 test sets and the training/validation data (at the solute level for OOD tests, or at least at the data-point level).
- Save to `/data/sc3/sc3_easy.csv`, `/data/sc3/sc3_medium.csv`, `/data/sc3/sc3_hard.csv`.
- Each file should contain: solute SMILES, solvent SMILES, temperature (K), ground truth log S, uncertainty estimate, number of independent sources, list of source DOIs.

### Step 4.4: Dataset Statistics Report

Write `/reports/phase_04_dataset_construction.md`:
- Training set: size, number of unique solutes/solvents, solubility distribution, temperature distribution, top 20 solvents by frequency.
- Validation set: same statistics.
- SC3-Easy/Medium/Hard: same statistics plus the aleatoric limit per set.
- Overlap analysis: confirm zero leakage.
- Comparison with existing benchmarks (AqSolDB size, ESOL size, BigSolDB raw size vs. our cleaned size).
- A "data quality certificate" — what the agent is confident about and what caveats exist.

**Completion criterion:** All dataset files created. Statistics verified. The agent is confident that training data is clean and SC3 test sets have well-characterized ground truths.

---

## PHASE 5: Multimodality Analysis and New Metrics

**Goal:** Demonstrate that solubility is a multimodal regression problem, that standard RMSE/R² are misleading, and propose better metrics.

### Step 5.1: Solubility Distribution by Solvent

- For each solvent in the dataset, plot the distribution of log S values across all solutes measured in that solvent.
- Compute the mean and variance of log S for each solvent.
- Show that different solvents have dramatically different mean solubilities (e.g., water vs. hexane vs. DMSO). This is the multimodality — the overall distribution of log S is a mixture of many Gaussians, one per solvent.
- Compute the overall distribution of log S across all solvents and show it is NOT a single Gaussian — fit a Gaussian Mixture Model and report the number of components, BIC, etc.
- Check for bimodality WITHIN individual solvents — for solvents with enough data, test whether polar vs. nonpolar solutes create a bimodal distribution. If so, this is an additional layer of multimodality.

### Step 5.2: Why RMSE and R² Are Misleading

Construct a concrete demonstration:

- Take a hypothetical model that simply predicts the mean log S for each solvent (the "solvent-mean baseline"). Compute its RMSE and R² on the full dataset. Show that R² can appear deceptively high because the model captures inter-solvent variance (which is trivial) without capturing intra-solvent variance (which is the actual prediction challenge).
- Show that a model can have excellent overall RMSE but terrible performance on specific solvents where the mean is far from the global mean.
- Compare two hypothetical models: one with uniform 0.5 log S error across all solvents, and one with 0.1 error on the most common solvent and 2.0 error on rare solvents. They might have similar overall RMSE but wildly different utility.

### Step 5.3: Solvent Similarity and Solubility Correlation

- For pairs of solvents that have been tested on the same solute, compute the correlation of log S values.
- Build a solvent-solvent similarity matrix based on solubility correlation.
- Compare this against chemical similarity (e.g., Tanimoto similarity of solvent fingerprints, or Hansen solubility parameter distance).
- Show that chemically similar solvents produce correlated solubility profiles — this is a known principle ("like dissolves like") but quantifying it on this scale is novel.
- This analysis also informs whether solvent-transfer learning is feasible.

### Step 5.4: Define New Metrics

**Z-RMSE (Solvent-Normalized RMSE):**
For each solvent s, compute the z-score of each prediction error:
```
z_i = (ŷ_i - y_i) / σ_s
```
where σ_s is the standard deviation of log S values for solvent s in the training data. Then:
```
Z-RMSE = √(mean(z_i²)) across all predictions
```
This normalizes by the inherent spread of each solvent's solubility distribution, making errors comparable across solvents.

**Z-MAE:** Same idea but with absolute values: `Z-MAE = mean(|z_i|)`.

**Z-R²:** Compute R² after z-score normalization within each solvent. This measures how well the model captures intra-solvent variance, not inter-solvent variance.

**Per-Solvent Averaged RMSE (PS-RMSE):**
```
PS-RMSE = (1/|S|) Σ_s RMSE_s
```
where RMSE_s is the RMSE computed only on data points from solvent s. This gives equal weight to each solvent regardless of data frequency.

**MAPE (Mean Absolute Percentage Error):**
```
MAPE = mean(|ŷ_i - y_i| / |y_i|) × 100
```
Note: MAPE is problematic when y_i is near zero. Handle this carefully (e.g., exclude data with |log S| < 0.1, or use SMAPE).

**Weighted RMSE by Aleatoric Limit:**
For each data point, weight the squared error by the inverse of the local aleatoric uncertainty:
```
W-RMSE = √(Σ w_i (ŷ_i - y_i)² / Σ w_i),  w_i = 1/ε_i²
```
This downweights predictions where the ground truth is itself uncertain.

### Step 5.5: Validate New Metrics

- Demonstrate on synthetic data that Z-RMSE and PS-RMSE are better at distinguishing good models from bad models in the multimodal setting.
- Show that two models with similar RMSE can have very different Z-RMSE if one is biased toward common solvents.
- Argue for which metric(s) should be the PRIMARY metric for the SC3 challenge.

### Step 5.6: Report

Write `/reports/phase_05_multimodality_metrics.md`:
- All multimodality analyses with figures.
- The concrete RMSE/R² failure demonstration.
- Solvent similarity analysis.
- Formal definitions of all new metrics.
- Validation experiments on synthetic data.
- Recommendation for primary SC3 metric.

**Completion criterion:** Clear evidence that solubility is multimodal. At least 2-3 new metrics formally defined with justification. The agent can explain in one paragraph why standard RMSE is insufficient.

---

## PHASE 6: Literature Survey and Method Selection

**Goal:** Identify all methods to benchmark, understand their architectures, and plan the experimental setup.

### Step 6.1: Taxonomy of Methods

Classify solubility prediction methods into families:

1. **Classical/Empirical:** General Solubility Equation (GSE), ESOL model. These are baselines that any ML method should beat.
2. **Descriptor-based ML (Trees):** Random Forest, XGBoost, LightGBM, CatBoost (DISSOLVR's backbone). These use explicit molecular descriptors.
3. **Descriptor-based ML (Neural):** MLP / feedforward networks on molecular descriptors.
4. **Graph Neural Networks:** GCN, GAT, GIN, MPNN. These learn directly from molecular graphs.
5. **Specialized Solubility Architectures:** MolMerger (Ramani & Karmakar 2024), Chemprop (Heid et al. 2024), FastSolv (Attia et al. 2025), SolTranNet, AqSolPred, DISSOLVR.
6. **Property Prediction Frameworks (Adjacent Fields):** Search for lightweight models from molecular property prediction that can be adapted. Look for recent methods in the Journal of Cheminformatics, JCIM, or ML venues that handle multi-property prediction on molecular graphs.

For each method, record: paper citation, GitHub repo URL, required dependencies, expected compute requirements, whether it can handle multi-solvent prediction, and whether it requires GPU.

### Step 6.2: Feasibility Check

For each candidate method:
- Clone the repo (into `/src/baselines/<method_name>/`).
- Check if it runs on CPU (no GPU available).
- Check if dependencies are compatible with the current environment.
- Try a minimal test run on 100 data points to verify the code works.
- If the code is broken, spend ONE session trying to fix it. If it remains broken, document the issue in `/reports/phase_06_method_survey.md` and move on.
- If a method requires GPU and cannot run on CPU in reasonable time, document this and skip it.

### Step 6.3: Feature Engineering for Fair Comparison

For descriptor-based methods (trees, MLP):
- Use the DISSOLVR featurizer as the primary feature set (176 features per molecule). Implement this from the DISSOLVR paper's Table 11 using RDKit.
- Also implement a simpler feature set (RDKit 2D descriptors, Morgan fingerprints) for comparison.
- For multi-solvent methods, implement the Refined Feature Set (24 features, Table 1 from DISSOLVR) for the interaction terms.

For graph-based methods:
- Use standard molecular graph construction (atoms as nodes, bonds as edges, atom/bond features from RDKit).
- For multi-solvent methods that need both solute and solvent graphs, implement the graph construction for both.

### Step 6.4: Read NeurIPS Benchmarking Papers

- Search for and read 2-3 well-regarded NeurIPS Datasets & Benchmarks papers to understand:
  - How they structure their paper (introduction of the problem, description of the dataset, benchmark protocol, results, analysis).
  - How they present tables and figures.
  - What level of detail they provide in the appendix.
  - How they handle the "contribution" framing (the dataset and analysis ARE the contribution, not a new model).
- Record insights in the report.

### Step 6.5: Report

Write `/reports/phase_06_method_survey.md`:
- Taxonomy table with all methods, repos, status (works/broken/skipped).
- Feature sets defined.
- NeurIPS benchmarking paper insights.
- Final list of methods that WILL be benchmarked.

**Completion criterion:** All repos cloned, feasibility tested, final method list decided. Feature engineering code written and tested.

---

## PHASE 7: Benchmarking — Training and Evaluation

**Goal:** Train all methods on the clean training data and evaluate on SC3 challenge sets. This is the largest phase in terms of compute time.

### Step 7.1: Experimental Protocol

Define and document the exact protocol BEFORE running anything:

- **Training data:** `/data/clean/train.csv`
- **Validation data:** `/data/clean/val.csv` (for hyperparameter tuning and early stopping)
- **Test data:** SC3-Easy, SC3-Medium, SC3-Hard
- **Seeds:** Run each method with 5 random seeds. Report mean ± std.
- **Hyperparameter tuning:** For each method, do a reasonable hyperparameter search on the validation set. Document the search space and the chosen hyperparameters. Use the original paper's recommended hyperparameters as the starting point.
- **Time tracking:** Record wall-clock training time and inference time for every method. Use `time.time()` wrappers. Record the hardware used (CPU model, RAM).
- **Metrics:** For each evaluation, compute ALL of the following:
  - Standard: RMSE, MAE, R², MAPE
  - New: Z-RMSE, Z-MAE, Z-R², PS-RMSE
  - Per-solvent RMSE for the top 10 most frequent solvents

### Step 7.2: Run Each Method

For each method in the final list:

1. **Prepare data in the method's expected format.** Some methods expect SMILES strings, others expect pre-computed features, others expect graph objects. Write a data adapter script for each.
2. **Train on training data with validation-based early stopping or hyperparameter selection.**
3. **Generate predictions on ALL test sets (SC3-Easy, Medium, Hard).**
4. **Compute all metrics.**
5. **Save predictions to `/tables/<method_name>_predictions.csv`.**
6. **Save the trained model if possible (for later interpretability analysis).**
7. **Record any issues, warnings, or observations.**

Run methods in this order (simplest first, to catch data pipeline bugs early):
1. GSE (analytical, no training)
2. ESOL model (analytical/regression, very fast)
3. Random Forest on DISSOLVR features
4. XGBoost on DISSOLVR features
5. LightGBM on DISSOLVR features
6. CatBoost on DISSOLVR features (this is essentially DISSOLVR's Regime I)
7. MLP on DISSOLVR features
8. DISSOLVR full (with Interaction Layer for Regime II)
9. Chemprop
10. MolMerger
11. FastSolv (if CPU-feasible)
12. GCN, GAT, GIN, MPNN (on molecular graphs)
13. Any additional methods from the literature survey

### Step 7.3: Ablation Studies

Run the following ablations:

**Feature ablation (on the best tree-based method):**
- Remove each feature category one at a time (Compositional, Topological, Energetic, Physicochemical) and re-evaluate.
- Use ONLY each feature category one at a time and evaluate.

**Training data quality ablation:**
- Train on the full (uncleaned) BigSolDB and test on SC3. Compare against training on the cleaned dataset. The hypothesis: training on cleaned data gives better SC3 performance even though the training set is smaller.

**Solvent-specific fine-tuning:**
- Take the best general model. For each of the top 5 most frequent solvents in SC3, fine-tune on data from THAT specific solvent and evaluate. Show that solvent-specific fine-tuning improves accuracy.
- This demonstrates the "lack of data" problem — there's not enough data per solvent to build truly specialized models, but general models lose accuracy by averaging across diverse chemical environments.

**Transfer learning across solvents:**
- Train on data from the top K solvents, test on rare solvents. Vary K. Show the generalization gap.

### Step 7.4: Results Tables

Produce the following tables (saved as both CSV and LaTeX in `/tables/`):

1. **Main results table:** All methods × all SC3 test sets × primary metrics (RMSE, Z-RMSE, R², Z-R²). Highlight best and second-best. Include training time and inference time columns.
2. **Per-solvent results table:** For the top 10 solvents, RMSE per method. This shows which methods are strong/weak on which solvent types.
3. **Ablation table:** Feature ablation results.
4. **Data quality ablation table:** Clean vs. unclean training data.
5. **Fine-tuning table:** General vs. solvent-specific models.

### Step 7.5: Report

Write `/reports/phase_07_benchmarking.md`:
- All tables with analysis.
- Key findings: which methods are best? Do deep methods justify their compute cost? Does data cleaning help?
- Surprises or unexpected results.
- Fair comparison commentary: where methods had advantages (e.g., more features, different training regimes) and how this was controlled for.

**Completion criterion:** All methods trained and evaluated. All tables produced. Results are reproducible (configs saved in `/configs/`). The agent has verified that no obvious errors exist in the pipeline.

---

## PHASE 8: Model Analysis and Interpretability

**Goal:** Understand WHAT the models learn, not just HOW WELL they predict. This adds scientific depth beyond mere benchmarking.

### Step 8.1: Feature Importance Analysis (Tree-Based Methods)

- For the best tree-based method (likely CatBoost/DISSOLVR), extract SHAP values for all features.
- Plot SHAP summary: which features matter most? Does this align with chemical intuition?
- Compare feature importance across solvents: do different features matter for polar vs. nonpolar solvents?
- Compare feature importance on SC3-Easy vs. SC3-Hard: does the model rely on different features for "easy" vs. "hard" predictions?

### Step 8.2: Graph Explainability (GNN Methods)

- For the best-performing GNN, use gradient-based or attention-based explainability:
  - GNNExplainer or similar method to identify important substructures.
  - Attention weights from GAT to visualize which atoms/bonds the model focuses on.
- Select 5-10 representative molecules across the solubility spectrum and visualize the explanations.
- Do the highlighted substructures align with known solubility-relevant functional groups (hydroxyl for water solubility, aromatic rings for organic solvents, etc.)?

### Step 8.3: Error Analysis Across Model Families

- For each model family, categorize the types of errors:
  - Systematic bias (consistently predicting too high or too low for a solvent)?
  - High-error outliers (a few molecules with > 2 log S error — what's special about them)?
  - Correlation with molecular properties (do errors increase with molecular weight? With number of rotatable bonds? With structural novelty vs. training set?).
- Compute the Tanimoto similarity of each SC3 molecule to its nearest neighbor in the training set. Plot error vs. similarity — this shows OOD degradation.

### Step 8.4: Physics-Consistency Checks

- For models that accept temperature as input, test temperature consistency: for a given (solute, solvent), does predicted solubility increase monotonically with temperature? (It should, for endothermic dissolution.)
- Plot predicted solubility vs. temperature for 10 representative systems and overlay experimental data. Do the models capture the correct trend shape?
- Only DISSOLVR enforces monotonic constraints — show whether other methods violate this.

### Step 8.5: Report

Write `/reports/phase_08_interpretability.md`:
- SHAP analysis figures and discussion.
- GNN explainability visualizations.
- Error analysis across model families.
- Physics-consistency results.
- A synthesis: what general lessons about model design emerge from this analysis?

**Completion criterion:** At least 3 distinct interpretability analyses completed across 2+ model families. Figures produced. Insights documented.

---

## PHASE 9: Paper Writing Preparation

**Goal:** Organize all findings into a coherent paper structure with pre-written sections, figures, and tables ready for LaTeX.

### Step 9.1: Read and Internalize NeurIPS D&B Format

- Look up the NeurIPS Datasets & Benchmarks track submission guidelines (page limit, required sections, review criteria).
- The typical structure is:
  1. Introduction (problem significance, gap in existing benchmarks)
  2. Related Work (existing datasets, existing methods, existing evaluations)
  3. The Dataset (construction methodology, statistics, quality guarantees)
  4. Benchmark Protocol (methods evaluated, metrics used, experimental setup)
  5. Results and Analysis (main tables, ablations, interpretability)
  6. Discussion and Recommendations (what we learned, guidelines for future work)
  7. Broader Impact / Limitations

### Step 9.2: Draft the Paper Narrative

Write a narrative outline (not the full paper, but the STORY) in `/paper/narrative_outline.md`:

The story should flow as:
1. **Solubility prediction matters** (pharma, environment) but is fundamentally limited by data quality.
2. **The field has a benchmarking crisis**: reported performance is inflated by data leakage, non-standardized curation, and misleading metrics. Many claims of "beating the aleatoric limit" are artifacts. (Cite Llompart et al. 2024, Attia et al. 2025.)
3. **We rigorously quantify the problem**: copycat duplication, true inter-lab variability, stratified aleatoric limits.
4. **Solubility is multimodal**: standard metrics fail. We propose better ones.
5. **SC3**: a challenge with known ground truth quality, stratified by difficulty.
6. **Fair benchmarking**: X methods evaluated under identical conditions. Key finding: [whatever the data shows].
7. **What models actually learn**: interpretability reveals [insights].
8. **Recommendations**: for dataset creators, model builders, and benchmark designers.

### Step 9.3: Prepare All Figures

Compile a final figure list. For each figure:
- Produce publication-quality versions (matplotlib with consistent style, proper font sizes, colorblind-friendly palettes).
- Write a detailed caption.
- Save to `/figures/paper/`.

Expected figures (adapt based on actual findings):
1. Dataset construction pipeline (flowchart).
2. Copycat/duplication discovery (before/after cleaning comparison).
3. Inter-lab variability distribution.
4. Aleatoric limit by data quality tier.
5. Multimodality: solubility distributions across solvents.
6. Why RMSE fails: concrete demonstration.
7. Main results: radar plot or grouped bar chart of methods × metrics.
8. Per-solvent performance heatmap.
9. Data quality ablation (clean vs. unclean training).
10. Feature importance / SHAP analysis.
11. GNN explainability visualizations.
12. Temperature-consistency check.

### Step 9.4: Prepare All Tables

Same for tables — compile final versions in LaTeX format in `/tables/paper/`.

### Step 9.5: Write the Paper

Using the NeurIPS LaTeX template, write the full paper in `/paper/sc3_paper.tex`.

- Main paper: within the page limit (typically 9 pages for D&B track).
- Appendix: unlimited. Include all supplementary tables, additional figures, full method descriptions, hyperparameter tables, per-solvent results, and the complete DOI reliability ranking.
- Every claim must be supported by a number from the reports.
- Every citation must be verified (correct author, year, venue).
- Cross-reference with DISSOLVR paper where appropriate (same research group, builds on that work).

### Step 9.6: Final Check

- Re-run key experiments to verify reproducibility.
- Check all numbers in tables against the raw results files.
- Ensure no data leakage (SC3 test molecules not in training data).
- Proofread for clarity and consistency.

**Completion criterion:** Complete paper draft with all figures and tables. The paper tells a coherent story. Every number is traceable to a report.

---

## Appendix A: Key References the Agent Should Be Aware Of

These are the most important papers in this space. The agent should search for and read these (or at least their abstracts and key results) during Phase 6:

1. **Palmer & Mitchell (2014)** — "Is experimental data quality the limiting factor in predicting the aqueous solubility of druglike molecules?" Mol. Pharm. — The foundational paper on aleatoric limits.
2. **Llompart et al. (2024)** — "Will we ever be able to accurately predict solubility?" Sci. Data. — Critical analysis of data quality and benchmarking issues.
3. **Attia et al. (2025)** — "Data-driven organic solubility prediction at the limit of aleatoric uncertainty." Nature Communications. — FastSolv and the multi-solvent aleatoric limit.
4. **Krasnov et al. (2023, 2025)** — BigSolDB and BigSolDB 2.0 — The primary datasets.
5. **Sorkun et al. (2019)** — AqSolDB — The primary aqueous dataset.
6. **Boobier et al. (2020)** — Machine learning with physicochemical relationships — Leeds dataset and descriptor-based approach.
7. **Ramani & Karmakar (2024)** — MolMerger — Physics-informed GNN for multi-solvent.
8. **Heid et al. (2024)** — Chemprop — D-MPNN architecture.
9. **Delaney (2004)** — ESOL — The classic baseline.
10. **The DISSOLVR paper (Ramani, Arora, Kuchhal, Ranu, Karmakar 2026)** — Our own prior work. The agent should understand its methodology deeply as it shares authors with this paper.

## Appendix B: Critical Reminders

- **The agent does not have GPU access.** All methods must run on CPU. Skip anything that requires GPU and cannot reasonably run on CPU within a few hours.
- **The agent should not rush.** Each phase should be thoroughly completed before moving on. If something doesn't look right, investigate. The worst outcome is a paper with wrong numbers.
- **Pass-on reports are mandatory.** When a session ends or the agent is uncertain, write to `/pass-ons/`. The next agent instance MUST check this folder first.
- **Every script should be runnable independently.** Write code so that `python src/data/clean_bigsoldb.py --input data/raw/bigsoldb_v2/ --output data/intermediate/` works from the terminal.
- **Commit messages (if using git) should be descriptive.** Better yet, maintain a `/CHANGELOG.md` that logs what was done in each session.
- **The agent should maintain a running `/STATUS.md`** that always reflects the current state: which phases are complete, which are in progress, what the next action should be.

## Appendix C: The SC3 Framing

The "Third Solubility Challenge" name is deliberate — it follows the First Solubility Challenge (Llinas et al. 2008) and the Second Solubility Challenge (Llinas et al. 2020), which were aqueous-only and contained 32 and 100 compounds respectively. SC3 is:

1. **Multi-solvent** (not just aqueous).
2. **Much larger** (hundreds of data points per difficulty tier).
3. **Quality-stratified** (easy/medium/hard based on ground truth certainty).
4. **Metric-aware** (evaluated with Z-RMSE and other multimodal metrics, not just RMSE).
5. **Open and reproducible** (dataset, code, and evaluation SDK will be publicly released).

This framing positions the paper as the natural next step in a lineage of community challenges, while addressing the fundamental issues that previous challenges did not: data quality quantification, multimodal metrics, and multi-solvent coverage.
