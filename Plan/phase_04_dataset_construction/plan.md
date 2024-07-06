# Phase 4: Dataset Construction — Clean Train/Val and SC3 Challenge

**Goal:** Using the source analysis from Phase 3, construct (a) a clean training/validation dataset and (b) the SC3 held-out challenge sets.

**Completion Criterion:** All dataset files created. Statistics verified. Training data is clean and SC3 test sets have well-characterized ground truths. Zero leakage confirmed.

**Outputs:**
- `sc3-benchmark/data/clean/train.csv`
- `sc3-benchmark/data/clean/val.csv`
- `sc3-benchmark/data/sc3/sc3_easy.csv`
- `sc3-benchmark/data/sc3/sc3_medium.csv`
- `sc3-benchmark/data/sc3/sc3_hard.csv`
- `sc3-benchmark/reports/phase_04_dataset_construction.md`

**Depends on:** Phases 2 and 3 complete

---

## Step 4.1: Two-Pointer Cleaning Algorithm

For each (solute, solvent) group with multiple measurements:
- [ ] Sort measurements by predicted value at reference temperature
- [ ] Sliding-window to find LARGEST subset where max-min < threshold ε₀
- [ ] ε₀ informed by Phase 3's aleatoric limit per data quality tier
- [ ] Flag outliers outside consensus window (record source DOI)
- [ ] Take median of consensus window as "ground truth"

## Step 4.2: Construct Clean Training Dataset

- [ ] Start with all (solute, solvent, temperature, log S) tuples
- [ ] Remove tuples from Hall-of-Shame DOIs
- [ ] For multi-source tuples, use consensus value from two-pointer
- [ ] For single-source tuples, keep IF source reliability score above threshold (avg MAE < 0.5)
- [ ] Apply anti-leakage protocol: remove molecules that appear in SC3 test sets
- [ ] Split: train (85%) / validation (15%), stratified by solvent
- [ ] Save to `sc3-benchmark/data/clean/`

## Step 4.3: Construct SC3 Challenge Sets

**SC3-Easy (~500 data points):**
- [ ] Select pairs with ≥5 independent sources, consensus MAE < 0.3 log S
- [ ] Sample diverse solutes and solvents
- [ ] Well-defined ground truth — deviation is almost entirely model error

**SC3-Medium (~200 data points):**
- [ ] Select pairs with 3-4 sources, consensus MAE 0.3-0.5 log S
- [ ] Ground truth less certain but defensible

**SC3-Hard (~100 data points):**
- [ ] Decide framing: tightest ground truth (hardest for model to hide behind noise) OR novel OOD solutes/rare solvents
- [ ] Document the choice and rationale

**For all sets:**
- [ ] Ensure ZERO overlap with training/validation data (at solute level for OOD, at least data-point level otherwise)
- [ ] Each row: solute SMILES, solvent SMILES, temperature (K), ground truth log S, uncertainty estimate, number of independent sources, source DOIs
- [ ] Save to `sc3-benchmark/data/sc3/`

## Step 4.4: Dataset Statistics Report

- [ ] Training set: size, unique solutes/solvents, solubility distribution, temperature distribution, top 20 solvents
- [ ] Validation set: same
- [ ] SC3-Easy/Medium/Hard: same + aleatoric limit per set
- [ ] Overlap analysis: confirm zero leakage
- [ ] Compare with existing benchmarks (AqSolDB, ESOL, BigSolDB raw vs. cleaned)
- [ ] "Data quality certificate" — confidence and caveats

---

## Key Notes for Agent

- The SC3-Hard framing decision is important — document the reasoning
- Anti-leakage must be verified computationally, not just assumed
- Keep the full provenance chain: for every data point, you should be able to trace back to which DOIs contributed
