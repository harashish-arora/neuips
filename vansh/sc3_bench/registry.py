"""
Method and hyperparameter registries for SC3 benchmark.

METHOD_REGISTRY maps method_key -> metadata dict.
load_hps() returns the best hyperparameters from configs/best_hps.json.
"""

import json
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# model_type: "tree" | "descriptor_nn" | "gnn"
METHOD_REGISTRY = {
    # Descriptor + tree ensembles (RDKit)
    "lgb_rdkit":        {"display": "LightGBM (RDKit)",  "family": "Desc+Tree",  "featurizer": "rdkit",    "model_type": "tree",  "tree_method": "lgb"},
    "catboost_rdkit":   {"display": "CatBoost (RDKit)",  "family": "Desc+Tree",  "featurizer": "rdkit",    "model_type": "tree",  "tree_method": "catboost"},
    "xgb_rdkit":        {"display": "XGBoost (RDKit)",   "family": "Desc+Tree",  "featurizer": "rdkit",    "model_type": "tree",  "tree_method": "xgb"},
    "rf_rdkit":         {"display": "RF (RDKit)",        "family": "Desc+Tree",  "featurizer": "rdkit",    "model_type": "tree",  "tree_method": "rf"},
    "dt_rdkit":         {"display": "DT (RDKit)",        "family": "Desc+Tree",  "featurizer": "rdkit",    "model_type": "tree",  "tree_method": "dt"},
    # Domain
    "lgb_dissolvr":     {"display": "Dissolvr",          "family": "Domain",     "featurizer": "dissolvr", "model_type": "tree",  "tree_method": "lgb"},
    # Mordred + RF
    "tayyebi_mordred":  {"display": "Tayyebi (Mordred)", "family": "Desc+Tree",  "featurizer": "mordred",  "model_type": "tree",  "tree_method": "tayyebi"},
    # Fingerprint + tree
    "lgb_morgan":       {"display": "LightGBM (Morgan)", "family": "FP+Tree",    "featurizer": "morgan",   "model_type": "tree",  "tree_method": "lgb"},
    "catboost_morgan":  {"display": "CatBoost (Morgan)", "family": "FP+Tree",    "featurizer": "morgan",   "model_type": "tree",  "tree_method": "catboost"},
    "xgb_morgan":       {"display": "XGBoost (Morgan)",  "family": "FP+Tree",    "featurizer": "morgan",   "model_type": "tree",  "tree_method": "xgb"},
    "rf_morgan":        {"display": "RF (Morgan)",       "family": "FP+Tree",    "featurizer": "morgan",   "model_type": "tree",  "tree_method": "rf"},
    # Fingerprint + GP
    "gp_morgan":        {"display": "GP (Tanimoto)",     "family": "FP+GP",      "featurizer": "morgan",   "model_type": "tree",  "tree_method": "gp"},
    # Descriptor NNs
    "fastprop":         {"display": "FastProp",          "family": "Deep Desc",  "featurizer": "rdkit",    "model_type": "descriptor_nn", "nn_arch": "fastprop"},
    "fastprop_big":     {"display": "FastProp-Big",      "family": "Deep Desc",  "featurizer": "rdkit",    "model_type": "descriptor_nn", "nn_arch": "fastprop"},
    "fastprop_xl":      {"display": "FastProp-XL",       "family": "Deep Desc",  "featurizer": "rdkit",    "model_type": "descriptor_nn", "nn_arch": "fastprop"},
    "fastsolv":         {"display": "FastSolv",          "family": "Deep Desc",  "featurizer": "rdkit",    "model_type": "descriptor_nn", "nn_arch": "fastsolv"},
    "mlp":              {"display": "MLP (RDKit)",       "family": "Deep Desc",  "featurizer": "rdkit",    "model_type": "descriptor_nn", "nn_arch": "mlp"},
    # GNNs
    "gcn":              {"display": "GCN",               "family": "GNN",        "featurizer": "graph",    "model_type": "gnn",  "gnn_type": "GCN"},
    "gat":              {"display": "GAT",               "family": "GNN",        "featurizer": "graph",    "model_type": "gnn",  "gnn_type": "GAT"},
    "gin":              {"display": "GIN",               "family": "GNN",        "featurizer": "graph",    "model_type": "gnn",  "gnn_type": "GIN"},
    # Merged-graph GNN
    "molmerger":        {"display": "MolMerger",         "family": "Merged GNN", "featurizer": "merged",   "model_type": "molmerger"},
}

METHOD_ORDER = list(METHOD_REGISTRY.keys())

DEFAULT_SEEDS = [42, 101, 123, 456, 789]
EVAL_SPLITS = ["eval", "ood", "sc3_gold", "sc3_silver", "sc3_bronze"]
NJOBS = 16


def load_hps() -> dict:
    """Load best hyperparameters from configs/best_hps.json."""
    hp_file = CONFIGS_DIR / "best_hps.json"
    if not hp_file.exists():
        raise FileNotFoundError(f"HP config not found: {hp_file}")
    with open(hp_file) as f:
        return json.load(f)


def get_hp(method_key: str) -> dict:
    """Get best HP dict for a single method."""
    all_hps = load_hps()
    if method_key not in all_hps:
        raise KeyError(f"No HPs for '{method_key}'. Available: {list(all_hps.keys())}")
    return all_hps[method_key]
