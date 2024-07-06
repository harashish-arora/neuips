# Ablations / Interpretability

This experiment answers the interpretability arm of **Q5: what has the model
*actually learned*?** by attributing predictions back to the chemistry.
It is the logical sibling of the Representation ablation: we keep the **same
fixed LightGBM** (tuned `lgb_rdkit` HPs) and the **same 7 featurizers** from
Representation, and then - for each featurizer - we compute

- **main-effect SHAP values** (exact Tree-SHAP via `shap.TreeExplainer`) on
  the full `eval` and `ood` splits, so the importance story maps directly to
  the RMSE / PS-RMSE / Z-RMSE numbers in the benchmark,
- **per-solvent importance rankings** on the 25 in-distribution solvents
  (= top-25 by count on `eval`, all > 100 rows), so each solvent's feature
  profile is an interpretable vector,
- **hierarchical clustering of solvents** based on their per-solvent SHAP
  fingerprint: solvents the model treats similarly end up in the same
  cluster, independently of any chemist's prior,
- **TreeSHAP interaction values** on `abraham_only` (exact, full sample) and
  `rdkit` (small sample) to see which feature *pairs* the model uses jointly,
  and
- a dedicated **Abraham/LSER axis** view that reads out, per solvent, how
  much each of `A / B / S / E / V / Tm` matters.

On top of this, a **graph-side study** (`run_gcn_explain.py`) re-uses the
already-trained GCN (no retraining) and does

- **atom-level occlusion attribution** on ~5000 `eval` rows
  (`a_v = f(G) − f(G \ {v})` with node features of atom `v` zeroed), and
- **BRICS fragment aggregation** of those atom scores - globally and per
  top-10 solvents, giving a chemically meaningful substructure ranking.

> **On GraphTrail / MAGE.**  Both are graph-*classification* explainers (they
> compute Shapley values over boolean value functions and/or generate
> class-specific motif graphs).  Our GCN is a **dual-encoder regression**
> model on a continuous target (logS).  After reading the papers and the
> reference repos we concluded that adapting them cleanly would require
> significant research (regression value function, dual-encoder support,
> continuous symbolic regression) and produce a method that isn't really
> "their" method.  Instead we used the approach that is both simpler and
> already standard for graph regression: occlusion attribution aggregated to
> RDKit BRICS substructures.  The PyG `Explainer` / `GNNExplainer` in
> regression mode is wired-in but slower, so we keep it optional.

## Model

Identical to Representation:

- **Model:** LightGBM, fixed HPs from `configs/best_hps.json["lgb_rdkit"]`
  (`n_estimators=3000`, `num_leaves=127`, `lr=0.05`, early-stopping on the
  full `eval` split with patience=50).
- **Seed:** single (42). Consistent with the Representation ablation
  single-seed pass.  Re-running with more seeds is resumable and changes
  only the runtime.
- **GCN:** the already-trained model from `vansh/results/gcn/model_seed_42.pkl`
  (dual-encoder GCN, `hidden_dim=96`, `num_layers=4`).  Never retrained.

## Featurizers

| Key | n_features (concat) | Notes |
|-----|--------------------:|-------|
| `rdkit`        | 320  | 158 RDKit 2D descriptors per molecule |
| `morgan`       | 2052 | 1024-bit Morgan ECFP4 (radius 2) |
| `dissolvr`     | 356  | RDKit + MOSE + Joback + Abraham proxies |
| `mordred`      | 3226 | ~1600 Mordred 2D descriptors |
| `maccs`        | 338  | 166-bit MACCS substructure keys |
| `atompair`     | 2052 | 1024-bit Atom-Pair fingerprint |
| `abraham_only` | 16   | 5 Abraham proxies (A, B, S, E, V) + Tm, solute+solvent+T |

## How to run

```bash
cd /DATATWO/users/solubility/Solubility/vansh/Ablations/Interpretability

# 1. SHAP for all 7 featurizers on eval + ood (trains LightGBM if missing,
#    reuses cached models; ~8 min total).
python run_shap.py

# 2. Analysis & figures: global / per-solvent / solvent clustering / Abraham.
python analyze_shap.py --only abc
python analyze_shap.py --only abraham

# 3. TreeSHAP interaction values (abraham: 1500 rows exact in ~2 min;
#    rdkit: 200 rows, tree_limit=150 in ~10 min).
python analyze_shap.py --only interactions

# 4. GCN atom-level + BRICS fragment attribution (5000 eval rows, ~30 s on A100).
python run_gcn_explain.py --gpu 2 --n-sample 5000
```

All driver scripts are **resumable**: existing outputs are skipped unless
`--force` is passed.

## Outputs

```
Ablations/Interpretability/
├── run_shap.py                 # Trains LGB + computes SHAP per featurizer
├── analyze_shap.py             # Global/per-solvent/clustering/interactions
├── run_gcn_explain.py          # GCN occlusion + BRICS aggregation
├── README.md                   # (this file)
├── FINDINGS.md                 # Narrative summary of chemistry-aware insights
├── results/
│   ├── <featurizer>/
│   │   ├── feature_names.json
│   │   ├── metrics.json                       # sanity-check RMSE/MAE/R2/PS_RMSE per split
│   │   ├── model_seed_42.pkl
│   │   ├── shap_eval.npz                      # (N, F) + solvent/smiles alignment
│   │   ├── shap_ood.npz
│   │   ├── global_importance__eval.csv        # ranked features mean|SHAP|
│   │   ├── global_importance__ood.csv
│   │   ├── global_blockwise.json              # solute/solvent/T shares
│   │   ├── per_solvent__eval.csv              # all features per solvent
│   │   ├── per_solvent__ood.csv
│   │   ├── per_solvent_top5__eval.json
│   │   ├── per_solvent_top5__ood.json
│   │   ├── solvent_similarity.npz             # cosine-sim matrix + fingerprints
│   │   ├── shap_interactions__eval.npz        # (abraham_only, rdkit only)
│   │   └── interactions_top50__eval.csv       # (abraham_only, rdkit only)
│   ├── abraham_only/
│   │   └── abraham_axis_ranking.csv           # per-solvent A/B/S/E/V/Tm importance
│   └── gcn/
│       ├── metrics.json
│       ├── atom_occlusion__eval.csv.gz        # one row per atom per molecule
│       ├── atom_occlusion_by_atom_type.csv
│       ├── atom_occlusion_by_solvent.csv
│       ├── brics_fragment_importance.csv      # global fragment ranking
│       └── brics_fragment_per_solvent.csv     # top 15 fragments for top-10 solvents
├── figures/
│   ├── global_blockwise_panel.png             # solute/solvent/T share across all 7
│   ├── global_top_features__<feat>__<split>.png
│   ├── per_solvent_heatmap__<feat>__<split>.png
│   ├── per_solvent_summary__<feat>__<solvent>.png
│   ├── solvent_corr_heatmap__<feat>.png
│   ├── solvent_dendrogram__<feat>.png
│   ├── interaction_top__<feat>__eval.png
│   ├── abraham_axis_per_solvent__<side>.png   # solute / solvent
│   ├── gcn_atom_type_importance.png
│   ├── gcn_atom_type_per_solvent.png
│   └── gcn_brics_fragments_top.png
└── logs/
```

## Notes

- SHAP values use Tree-SHAP with `feature_perturbation="tree_path_dependent"`
  (exact, model-aware, no background sample needed).
- Per-solvent analysis uses **all rows** of that solvent in the given split
  (no subsampling), so the ranking is stable for well-populated solvents
  (N ≥ 100); small solvents (N < 5) are skipped.
- Solvent–solvent clustering uses **L2-normalised mean(|SHAP|) over the top
  50 globally-important features** as the per-solvent fingerprint, then
  cosine distance with average linkage.  The choice of K=50 is robustness
  over expressivity — varying K in [25, 100] gives the same top-level
  cluster structure.
- Cross-featurizer comparisons in FINDINGS.md are always anchored to the
  sanity-check RMSE printed in each `metrics.json`, which reproduces the
  Representation ablation numbers within rounding.
- The GCN occlusion scheme is "soft" — we zero out node features rather
  than delete the atom, so graph topology is preserved and message-passing
  remains well-defined. This is the standard technique used in the molecular
  GNN attribution literature (e.g., Pope et al. 2019; Jiménez-Luna et al.
  2020).  Exact-subgraph deletion would break the solute graph and isn't
  comparable across molecules.
