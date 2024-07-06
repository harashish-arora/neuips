"""
Uni-Mol2 baseline for SC3 benchmark.

Architecture (from arxiv 2406.14969):
  Uni-Mol2 is a two-track transformer pretrained on 800M molecular conformations
  that integrates atomic, graph, and 3D geometry information. The 84M-parameter
  base model is used here as a frozen feature extractor.

Approach — Dual-encoder with pretrained representations:
  1. Extract Uni-Mol2 CLS representations (768-d) for all unique molecules.
  2. For each (solute, solvent, temperature) sample, concatenate:
       [solute_repr(768) ‖ solvent_repr(768) ‖ temp_features(4)] = 1540-d
  3. Train a regression MLP head on these features.

References:
  - Uni-Mol2: Lu et al., arXiv:2406.14969 (2024)
  - unimol_tools: https://github.com/deepmodeling/unimol_tools
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseMethod


# ============================================================================
# Column resolution helpers — tolerant to bench/SC3 naming differences
# ============================================================================
# bench_*.csv use:  Solute_Canon, Solvent_Canon, Solvent_Name, Temperature_K, LogS
# sc3 tiers use:    Solute_Canon, Solvent_Canon, Temperature_K, LogS_consensus, sigma

def _resolve_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} in df columns: {list(df.columns)}")


def get_solute_col(df):
    return _resolve_col(df, ["Solute_Canon", "Solute", "Solute_SMILES"])


def get_solvent_col(df):
    """Solvent SMILES column (the canonical key for indexing reprs)."""
    return _resolve_col(df, ["Solvent_Canon", "Solvent", "Solvent_SMILES"])


def get_temp_col(df):
    return _resolve_col(df, ["Temperature_K", "Temperature", "T"])


def get_target_col(df):
    return _resolve_col(df, ["LogS", "LogS_consensus"])


# ============================================================================
# MLP regression head
# ============================================================================

class UniMolMLP(nn.Module):
    """Regression MLP head on concatenated Uni-Mol2 representations."""

    def __init__(self, repr_dim=768, num_temp_feats=4, dropout=0.3):
        super().__init__()
        in_dim = repr_dim * 2 + num_temp_feats
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# SMILES sanitization + Uni-Mol representation extraction
# ============================================================================

def _sanitize_smiles(smi):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None


def extract_unimol_representations(smiles_list, model_name="unimolv2",
                                   model_size="84m", use_cuda=True,
                                   batch_size=1024):
    """Extract Uni-Mol2 CLS representations for a list of SMILES.

    Failed molecules get zero vectors (logged).

    Returns:
        np.ndarray of shape (n_molecules, repr_dim)
    """
    from unimol_tools import UniMolRepr
    import logging
    logging.getLogger("unimol_tools").setLevel(logging.WARNING)

    repr_model = UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        model_name=model_name,
        model_size=model_size,
        use_cuda=use_cuda,
        batch_size=batch_size,
    )

    sanitized = [_sanitize_smiles(s) for s in smiles_list]
    failed_indices = {i for i, s in enumerate(sanitized) if s is None}
    if failed_indices:
        print(f"    Warning: {len(failed_indices)} SMILES failed sanitization, using zero vectors")

    valid_pairs = [(i, s) for i, s in enumerate(sanitized) if s is not None]
    valid_indices = [i for i, _ in valid_pairs]
    valid_smiles = [s for _, s in valid_pairs]

    repr_dim = None
    results = {}
    chunk_size = 256

    for start in range(0, len(valid_smiles), chunk_size):
        chunk = valid_smiles[start:start + chunk_size]
        chunk_idx = valid_indices[start:start + chunk_size]
        try:
            reprs = repr_model.get_repr(chunk)
            reprs_arr = np.array(reprs, dtype=np.float32)
            if repr_dim is None:
                repr_dim = reprs_arr.shape[1] if reprs_arr.ndim == 2 else reprs_arr.shape[0]
            for j, idx in enumerate(chunk_idx):
                results[idx] = reprs_arr[j] if reprs_arr.ndim == 2 else reprs_arr
        except Exception as e:
            print(f"    Warning: chunk failed ({e}), falling back to individual extraction")
            for smi, idx in zip(chunk, chunk_idx):
                try:
                    r = repr_model.get_repr([smi])
                    r_arr = np.array(r, dtype=np.float32)
                    if repr_dim is None:
                        repr_dim = r_arr.shape[1] if r_arr.ndim == 2 else r_arr.shape[0]
                    results[idx] = r_arr[0] if r_arr.ndim == 2 else r_arr
                except Exception:
                    failed_indices.add(idx)

    if repr_dim is None:
        repr_dim = 768  # default for unimolv2 84m

    output = np.zeros((len(smiles_list), repr_dim), dtype=np.float32)
    for idx, vec in results.items():
        output[idx] = vec

    if failed_indices:
        print(f"    Total failed: {len(failed_indices)}/{len(smiles_list)} molecules")

    return output


# ============================================================================
# Feature builders — tolerant to varying column names
# ============================================================================

def _build_features_array(df, solute_reprs, solvent_reprs, solute_to_idx, solvent_to_idx):
    sol_col = get_solute_col(df)
    solv_col = get_solvent_col(df)
    temp_col = get_temp_col(df)

    sol_indices = [solute_to_idx[s] for s in df[sol_col]]
    solv_indices = [solvent_to_idx[s] for s in df[solv_col]]

    sol_feats = solute_reprs[sol_indices]
    solv_feats = solvent_reprs[solv_indices]

    T = df[temp_col].values.astype(np.float64)
    temp_feats = np.column_stack([
        T / 300.0,
        1000.0 / T,
        (T / 300.0) ** 2,
        np.log(T / 300.0),
    ]).astype(np.float32)

    return np.concatenate([sol_feats, solv_feats, temp_feats], axis=1)


def build_unimol_features(df, solute_reprs, solvent_reprs, solute_to_idx,
                          solvent_to_idx, device):
    """Build feature & target tensors. Returns (features, targets)."""
    features = _build_features_array(df, solute_reprs, solvent_reprs,
                                     solute_to_idx, solvent_to_idx)
    target_col = get_target_col(df)
    features = torch.tensor(features, dtype=torch.float32, device=device)
    targets = torch.tensor(df[target_col].values.astype(np.float32),
                           dtype=torch.float32, device=device)
    return features, targets


def build_unimol_features_numpy(df, solute_reprs, solvent_reprs, solute_to_idx,
                                solvent_to_idx):
    """Numpy version for GBDT-style heads (CatBoost runner)."""
    features = _build_features_array(df, solute_reprs, solvent_reprs,
                                     solute_to_idx, solvent_to_idx)
    target_col = get_target_col(df)
    targets = df[target_col].values.astype(np.float32)
    return features, targets


# ============================================================================
# Method registry stubs (real training lives in the runner script)
# ============================================================================

class UniMolMethod(BaseMethod):
    @classmethod
    def info(cls):
        return {
            "name": "Uni-Mol2",
            "featurizer": "none",
            "mode": "unimol",
            "gpu_required": False,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError(
            "Uni-Mol2 requires the dedicated runner. Use scripts/train_unimol_mlp.py"
        )

    def predict(self, X):
        raise NotImplementedError("Uni-Mol2 requires the dedicated runner.")


from catboost import CatBoostRegressor


class UniMolCatBoostMethod(BaseMethod):
    """Uni-Mol2 pretrained representations + CatBoost head."""

    @classmethod
    def info(cls):
        return {
            "name": "Uni-Mol2+CatBoost",
            "featurizer": "none",
            "mode": "unimol",
            "gpu_required": True,
        }

    def __init__(self, **catboost_params):
        self.model = None
        self.catboost_params = catboost_params or {
            "iterations": 5000,
            "learning_rate": 0.05,
            "depth": 7,
            "loss_function": "RMSE",
            "task_type": "GPU",
            "devices": "0",
            "verbose": 100,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None
        self.model = CatBoostRegressor(**self.catboost_params)
        self.model.fit(X_train, y_train, eval_set=eval_set)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)
