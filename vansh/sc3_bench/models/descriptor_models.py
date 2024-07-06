"""
Descriptor-based neural network models for SC3 solubility prediction.

Three model variants:
  - FastPropNet:  Deep MLP with BatchNorm + Dropout (descriptor-based)
  - FastSolvNet:  Same architecture + Sobolev gradient regularization
  - SimpleMLP:    Vanilla MLP with ReLU + Dropout (no BatchNorm)

All models take pre-computed feature vectors (RDKit descriptors ⊕ temperature
features) as input and predict LogS.  Featurization is handled externally by
the runner script.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Model architectures
# =============================================================================


class FastPropNet(nn.Module):
    """Deep descriptor-based network with BatchNorm (FastProp-style).

    Architecture: [Linear → BatchNorm → ReLU → Dropout] × N → Linear(1)
    """

    def __init__(self, in_dim, hidden_dims=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FastSolvNet(nn.Module):
    """Deep descriptor-based network with BatchNorm (FastSolv architecture).

    Identical to FastPropNet in structure; the Sobolev training loss is
    applied externally in the training loop.
    """

    def __init__(self, in_dim, hidden_dims=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SimpleMLP(nn.Module):
    """Vanilla MLP with ReLU + Dropout (no BatchNorm).

    Architecture: [Linear → ReLU → Dropout] × N → Linear(1)
    """

    def __init__(self, in_dim, hidden_dims=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# Featurization helpers  (used by the runner scripts)
# =============================================================================


def build_feature_cache(splits, featurizer):
    """Pre-compute RDKit descriptor vectors for all unique SMILES across splits.

    Returns
    -------
    cache : dict[str, np.ndarray]
        SMILES → 1-D feature vector.
    feature_names : list[str]
        Descriptor column names (same order across all vectors).
    """
    all_smiles = set()
    for _name, df in splits.items():
        all_smiles.update(df["Solute"].unique())
        all_smiles.update(
            df["Solvent"].unique()
            if "Solvent" in df.columns
            else df["Solvent_Name"].unique()
        )

    print(f"Building feature cache for {len(all_smiles)} unique molecules...")
    smiles_list = sorted(all_smiles)
    feat_df = featurizer.transform(smiles_list)
    feature_names = list(feat_df.columns)

    cache = {}
    for smi, row in zip(smiles_list, feat_df.values):
        cache[smi] = row.astype(np.float32)

    print(f"  Cached {len(cache)} molecules × {len(feature_names)} descriptors")
    return cache, feature_names


def featurize_split(df, feature_cache, feature_names):
    """Build feature matrix X and target vector y for a data split.

    Features: [solute_desc, solvent_desc, T/300, 1000/T, (T/300)^2, ln(T/300)]

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Solute, Solvent (or Solvent_Name), Temperature, LogS.
    feature_cache : dict[str, np.ndarray]
        Pre-computed descriptor cache from ``build_feature_cache``.
    feature_names : list[str]
        Descriptor column names (used to determine dimension).

    Returns
    -------
    X : np.ndarray  shape (n, 2*n_desc + 4)
    y : np.ndarray  shape (n,)
    """
    solute_col = "Solute"
    solvent_col = "Solvent" if "Solvent" in df.columns else "Solvent_Name"
    temp_col = "Temperature"
    target_col = "LogS"

    n = len(df)
    n_desc = len(feature_names)
    X = np.zeros((n, 2 * n_desc + 4), dtype=np.float32)
    y = df[target_col].values.astype(np.float32)

    for i in range(n):
        solute_smi = df.iloc[i][solute_col]
        solvent_smi = df.iloc[i][solvent_col]
        T = float(df.iloc[i][temp_col])

        solute_feat = feature_cache.get(solute_smi, np.zeros(n_desc, dtype=np.float32))
        solvent_feat = feature_cache.get(solvent_smi, np.zeros(n_desc, dtype=np.float32))

        T_scaled = T / 300.0
        temp_feats = np.array([
            T_scaled,
            1000.0 / max(T, 1e-6),
            T_scaled ** 2,
            np.log(max(T_scaled, 1e-6)),
        ], dtype=np.float32)

        X[i, :n_desc] = solute_feat
        X[i, n_desc:2*n_desc] = solvent_feat
        X[i, 2*n_desc:] = temp_feats

    return X, y


# =============================================================================
# Sobolev gradient target computation  (FastSolv only)
# =============================================================================


SOBOLEV_SCALE = 10.0
GRAD_MAG_CAP = 1.0


def compute_sobolev_targets(desc_raw, t_std, y_std):
    """Compute empirical d(y_std)/d(t_std) via finite differences.

    Groups training data by identical descriptor vectors (same solute+solvent
    pair). Within each group, sorts by standardised temperature and computes
    np.gradient finite differences.

    Physics filter (endothermic dissolution prior):
      - Only keeps positive gradients (solubility increases with T)
      - Caps at GRAD_MAG_CAP in standardised space
      - NaN for all others; Sobolev loss uses masked mean to ignore them

    Parameters
    ----------
    desc_raw : np.ndarray  shape (n, d)
        Raw (un-normalised) descriptor features.
    t_std : np.ndarray  shape (n,)
        Standardised temperature values.
    y_std : np.ndarray  shape (n,)
        Standardised target values.

    Returns
    -------
    grads : np.ndarray  shape (n,)
        Per-sample gradient targets (NaN where invalid).
    """
    n = len(y_std)
    grads = np.full(n, np.nan, dtype=np.float32)

    desc_r = np.round(desc_raw, decimals=5)
    groups: dict[bytes, list[int]] = {}
    for i in range(n):
        k = desc_r[i].tobytes()
        groups.setdefault(k, []).append(i)

    for idx_list in groups.values():
        if len(idx_list) < 2:
            continue
        idx = np.array(idx_list)
        t_g = t_std[idx]
        y_g = y_std[idx]

        order = np.argsort(t_g)
        t_sorted = t_g[order]
        y_sorted = y_g[order]
        idx_sorted = idx[order]

        unique_t, inv = np.unique(t_sorted, return_inverse=True)
        if len(unique_t) < 2:
            continue

        y_avg = np.array(
            [y_sorted[inv == j].mean() for j in range(len(unique_t))]
        )
        fd = np.gradient(y_avg, unique_t)

        for i_local, orig_idx in enumerate(idx_sorted):
            g = float(fd[inv[i_local]])
            if np.isfinite(g) and 0.0 < g < GRAD_MAG_CAP:
                grads[orig_idx] = g

    return grads
