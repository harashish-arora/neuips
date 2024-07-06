# Phase 2: Data Merging and Structural Cleaning

**Goal:** Combine BigSolDB 1.0 and 2.0 into a single unified dataset, perform structural cleaning.

**Completion Criterion:** A single clean CSV at `sc3-benchmark/data/intermediate/bigsoldb_merged_clean.csv` with standardized SMILES, standardized solvents, log S in mol/L, temperature in K, and source DOI. Report written.

**Output:** `sc3-benchmark/reports/phase_02_merging.md`

**Depends on:** Phase 1 complete

---

## Step 2.1: Understand the Schema

- [ ] Load both BigSolDB versions
- [ ] Document column semantics: solute identifiers (SMILES, names), solvent identifiers, temperature, solubility values, units, source DOIs
- [ ] Determine if v2.0 is a strict superset of v1.0 or if they have unique entries
- [ ] Document schema differences

## Step 2.2: Merge and Deduplicate

- [ ] Concatenate v1.0 and v2.0
- [ ] Deduplicate: same solute SMILES + same solvent + same temperature + same solubility = keep one
- [ ] Canonicalize all SMILES using RDKit (`pip install rdkit-pypi` if needed)
- [ ] Strip stereochemistry for consistency with 2D methods
- [ ] Build solvent alias map from data (e.g., "THF" → "tetrahydrofuran", "DMSO" → "dimethyl sulfoxide")
- [ ] Standardize solvent names using the alias map

## Step 2.3: Unit Standardization

- [ ] Check what units solubility is reported in (BigSolDB uses mole fraction x)
- [ ] Convert to log₁₀(S) in mol/L: `log S = log₁₀(x · ρ_solvent(T) / MW_solvent)`
- [ ] Use `thermo` library for density at temperature T (`pip install thermo` if needed)
- [ ] Cache density calculations
- [ ] Flag entries where conversion fails (unknown solvent, missing density)

## Step 2.4: Basic Structural Filters

- [ ] Remove entries with "." in SMILES (salts, mixtures, multi-component)
- [ ] Remove polymeric solvents (PEG variants, Span, PEGDME, etc.)
- [ ] Remove entries with MW > 1000 Da
- [ ] Apply solubility range filter: keep -15 ≤ log S ≤ 2
- [ ] Standardize tautomers using RDKit's `rdMolStandardize.TautomerEnumerator`

## Step 2.5: Remove Known Bad DOIs

Remove all entries from these DOIs:
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

## Step 2.6: Report

Write report with:
- [ ] Entry counts at each cleaning stage (waterfall)
- [ ] Unique solutes, solvents, DOIs after cleaning
- [ ] Temperature distribution histogram → `sc3-benchmark/figures/`
- [ ] Solubility distribution histogram → `sc3-benchmark/figures/`
- [ ] Anomalies discovered

---

## Key Notes for Agent

- Script should be runnable: `python src/data/clean_bigsoldb.py --input ... --output ...`
- Check if Dissolvr already has cleaning/featurization code that can be reused
- The `Dissolvr/apelblat/` folder likely has relevant temperature-handling code
