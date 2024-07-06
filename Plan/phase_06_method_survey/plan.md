# Phase 6: Literature Survey and Method Selection

**Goal:** Identify all methods to benchmark, understand their architectures, and plan the experimental setup.

**Completion Criterion:** All repos cloned, feasibility tested, final method list decided. Feature engineering code written and tested.

**Output:** `sc3-benchmark/reports/phase_06_method_survey.md`

**Depends on:** Phase 4 complete (needs the clean dataset for test runs)

---

## Step 6.1: Taxonomy of Methods

Classify into families:

1. **Classical/Empirical:** GSE, ESOL model
2. **Descriptor-based ML (Trees):** Random Forest, XGBoost, LightGBM, CatBoost (DISSOLVR backbone)
3. **Descriptor-based ML (Neural):** MLP on molecular descriptors
4. **Graph Neural Networks:** GCN, GAT, GIN, MPNN
5. **Specialized Solubility Architectures:** MolMerger, Chemprop, FastSolv, SolTranNet, AqSolPred, DISSOLVR
6. **Property Prediction Frameworks:** lightweight models from adjacent fields

For each: paper citation, GitHub repo, dependencies, compute requirements, multi-solvent capability, GPU needed?

**Check Dissolvr repo first:** Many baselines already implemented in `Dissolvr/baselines/`

## Step 6.2: Feasibility Check

For each candidate:
- [ ] Clone repo (or locate in `Dissolvr/baselines/`) → `sc3-benchmark/src/baselines/<method>/`
- [ ] Check CPU compatibility (NO GPU available)
- [ ] Check dependency compatibility
- [ ] Test run on 100 data points
- [ ] If broken: spend ONE session fixing. If still broken, document and move on
- [ ] If GPU-only and unreasonable on CPU: document and skip

## Step 6.3: Feature Engineering

For descriptor-based methods:
- [ ] DISSOLVR featurizer (176 features) — implement from Table 11 using RDKit (or reuse from Dissolvr repo)
- [ ] Simpler feature set: RDKit 2D descriptors + Morgan fingerprints
- [ ] Refined Feature Set (24 features, Table 1) for interaction terms

For graph-based methods:
- [ ] Standard molecular graph construction (atoms=nodes, bonds=edges, RDKit features)
- [ ] Dual graph construction for multi-solvent (solute + solvent)

## Step 6.4: Read NeurIPS Benchmarking Papers

- [ ] Find and read 2-3 well-regarded NeurIPS D&B papers
- [ ] Note: paper structure, table/figure presentation, appendix detail, contribution framing
- [ ] Record insights

## Step 6.5: Report

- [ ] Taxonomy table: all methods, repos, status (works/broken/skipped)
- [ ] Feature sets defined
- [ ] NeurIPS D&B insights
- [ ] Final method list for benchmarking

---

## Key Notes for Agent

- The Dissolvr repo has many baselines pre-implemented — inventory those first
- CPU-only constraint is hard — document any methods skipped because of this
- Feature engineering code must be modular for Phase 7 to use
