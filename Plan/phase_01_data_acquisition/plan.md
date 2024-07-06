# Phase 1: Data Acquisition and Initial Inventory

**Goal:** Download all required datasets, understand their structure, and produce an initial inventory of what we have.

**Completion Criterion:** All datasets downloaded, inventory report written with hard numbers.

**Output:** `sc3-benchmark/reports/phase_01_data_inventory.md`

---

## Step 1.1: Acquire BigSolDB

- [ ] Locate BigSolDB 2.0 (Krasnov et al. 2025, Scientific Data) — check Zenodo, Figshare, paper's data availability
- [ ] Locate BigSolDB 1.0 (Krasnov et al. 2023, ChemRxiv) if separately available
- [ ] **CHECK FIRST:** Both may already exist in `Dissolvr/regime-ii/all_datasets/bigsol1.0/` and `bigsol2.0/`
- [ ] Copy/symlink raw files into `sc3-benchmark/data/raw/bigsoldb_v1/` and `bigsoldb_v2/`
- [ ] Record: row count, columns, column names, data types, metadata files

## Step 1.2: Acquire Additional Datasets

- [ ] AqSolDB (Sorkun et al. 2019) — ~9,982 compounds. **Check:** `Dissolvr/regime-i/all_datasets/aqsoldb/`
- [ ] ESOL (Delaney 2004) — ~1,128 compounds. **Check:** `Dissolvr/regime-i/all_datasets/esol/`
- [ ] Second Solubility Challenge (Llinas et al. 2020) — 100 compounds. **Check:** `Dissolvr/regime-i/all_datasets/sc2/`
- [ ] Leeds dataset (Boobier et al. 2020) — organic solvent solubility. **Check:** `Dissolvr/regime-ii/all_datasets/leeds/`
- [ ] Place each in `sc3-benchmark/data/raw/<dataset_name>/`

## Step 1.3: Initial Inventory Report

Write `sc3-benchmark/reports/phase_01_data_inventory.md` containing:
- For each dataset: source URL, citation, number of entries, unique solutes, unique solvents, temperature range, solubility unit, known issues
- Comparison table showing overlap potential
- Note which datasets have DOI/source info per measurement

---

## Key Notes for Agent

- Most datasets likely already exist in Dissolvr — inventory first, download only what's missing
- Don't skip any dataset even if it seems redundant — we need the full picture for the paper
- Record exact file paths so downstream phases can find them
