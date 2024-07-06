# SC³: A Multi-Solvent Solubility Benchmark with Calibrated Aleatoric Ground Truth

## Premise

Solubility prediction has been a cornerstone ML-for-chemistry task for two decades, but every existing benchmark is aqueous-only, treats labels as noise-free, and has a field-wide consensus that models have "hit the wall" at 0.6–0.8 log-unit error. We argue two of these three assumptions are wrong, and build a benchmark that enables new kinds of research that previous benchmarks couldn't support.

## Core idea

Calibrated ground truth changes what you can study. By forensically auditing a large multi-solvent dataset (BigSolDB), detecting copycat publications, and recovering per-point aleatoric uncertainty from multi-source consensus, we produce a benchmark where (a) the noise floor is an order of magnitude lower than folklore, (b) σ is known per measurement, and (c) the tree-vs-GNN gap can finally be decomposed mechanistically.

## Paper structure

1. **Introduction — solubility as a computational problem.** Why it matters (drug formulation, crystallization, green chemistry), why aqueous-only is insufficient, why the noise-floor argument is under-examined.

2. **Related work — history and datasets.** Solubility Challenges I/II, ESOL, AqSolDB, MoleculeNet, BigSolDB, SolProp. Note that none of these: (i) cover organic solvents at scale, (ii) audit source integrity, or (iii) calibrate per-measurement uncertainty.

3. **Data curation.** From 112K raw BigSolDB rows to 101,535 cleaned measurements. Structural filters (salts, polymeric solvents, MW, LogS range), SMILES canonicalization, LogS recovery from mole fractions, bad-DOI removal. Produces 1,311 solutes × 204 solvents × 1,494 DOIs.

4. **Aleatoric limit and SC³ construction.** Copycat detection via pairwise MAE (139 copycat pairs identified). DOI reliability ranking (369 ranked sources). Apelblat-interpolated multi-source consensus. Corrected aleatoric floor: median ε_direct ≈ 0.06 (vs folklore 0.6–0.8, which is P90–P95 of a heavy-tailed distribution). Three nested tiers (Hard/Medium/Easy) with per-point σ, plus solvent-OOD split.

5. **Benchmark design and metrics.** Molecule-level anti-leakage. PS-RMSE to strip between-solvent variance inflation. Z-RMSE (error normalized by σ) as the first aleatoric-aware chemistry metric. 2.1 log-unit shift across solvents motivates per-solvent evaluation.

6. **Experiments and benchmarking.** 31 methods across 8 paradigms, 5 seeds. LightGBM + RDKit wins (Hard PS-RMSE 0.659), beating Morgan fingerprints by 27%, GNNs by 22–34%, foundation models by 22%. All methods operate ~50× above the aleatoric floor.

7. **New analyses enabled by calibrated ground truth** (the additional experiments):
   - **Aleatoric-aware training:** heteroscedastic weighting gives LightGBM a significant 3.7% Z-RMSE improvement (n=20, p = 1.9e-6), with a symmetric OOD cost — evidence of capacity reallocation under epistemic-bounded regime.
   - **Feature attribution:** SHAP reveals 73/20/7 solute/solvent/temperature split; the model learns sophisticated complexity and shape descriptors (BertzCT, FractionCSP3) beyond naive pharmacophore counts.
   - **Solvent representation:** three Hansen parameters (δD, δP, δH) match 158 RDKit descriptors on in-distribution Hard PS-RMSE. Coverage-matched OOD shows principled solvation parameters generalize better per-solvent than RDKit — the field's default of generic descriptors is overkill.
   - **Transfer learning:** pretraining on ~1M CombiSolv-QM solvation free energies gives 10× data efficiency (FastProp transfer at 10% SC³ beats scratch at 100%). QM→Exp two-stage recipe closes 55% of the FastProp-LightGBM gap. GCN transfer closes only 22%, showing handcrafted features benefit more from pretraining than learned representations.
   - **Scaling and mechanism:** flat/wrong-signed scaling exponents confirm we're in a plateau regime. Descriptor injection into the GCN recovers 19% of the gap. GCN-on-scrambled-graphs finding (pending verification) — if it holds, evidence that message-passing is actively harmful at this scale.

8. **Discussion.** The remaining gap is (a) architectural (trees > MLPs on tabular solvation), (b) featurization coverage (principled parameters generalize but lookup tables are sparse), and (c) data scarcity for learned representations. Not fundamental limits — actionable research directions.

9. **Conclusion.** SC³ is not just a better benchmark but infrastructure: calibrated per-point σ, source-level provenance, and tiered ground truth enable training recipes (heteroscedastic weighting), evaluation recipes (Z-RMSE), and interpretability analyses that previous benchmarks could not support.

---

Three sentences if you need them shorter:

> *Multi-solvent solubility prediction has been stuck at 0.6–0.8 log-unit error for a decade, but the noise floor is actually ~0.06 once copycat publications and heavy-tailed statistics are corrected for. We build SC³, a rigorously audited benchmark with per-measurement aleatoric uncertainty across 101K measurements, and show it enables new analyses — aleatoric-aware training, principled solvent featurization, and controlled transfer learning decomposition — that previous benchmarks could not support. Trees still win, but we now know why: better features, better interaction modeling, and a coverage bottleneck on principled solvent parameters that defines the next research direction.*
