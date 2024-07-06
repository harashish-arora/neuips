"""
MolMerger: GNN with Gasteiger charge-based graph merging for solubility prediction.

Ramani & Karmakar, arXiv 2402.11340 / J. Chem. Theory Comput. 2024.
GitHub: VanshRamani/Molmerger-Solubility-Prediction

Core idea: merge solute + solvent into a single graph by adding two virtual
"interaction bonds" between the most electron-dense atom in solute and the
least electron-dense in solvent (and vice versa), based on Gasteiger partial
charges.  Feeds merged graph into AttentiveFP (PyG) for regression.

Temperature features are appended to every node in the merged graph so the
GNN sees thermal context at every message-passing step.

Node features (35-d):
    atom_type one-hot (10) + formal_charge (1) + hybridization one-hot (3)
    + H-bond acceptor/donor (2) + aromatic (1) + degree one-hot (7)
    + total_num_Hs one-hot (5) + chirality (2)
    + temperature features (4): T/300, 1000/T, (T/300)^2, ln(T/300)

Edge features (14-d):
    bond_type one-hot (5, includes INTERACTION for virtual bonds)
    + same_ring (1) + conjugated (1) + stereo one-hot (6) + graph_distance (1)
"""

import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski
from torch_geometric.data import Data, Batch
from torch_geometric.nn.models import AttentiveFP as PyGAttentiveFP

BaseMethod = object

# ============================================================================
# Constants
# ============================================================================

ATOM_TYPE_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si"]
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

CHEM_NODE_DIM = 31   # pure chemical features (no temperature)
NODE_DIM = 31 + 4    # 31 chem features + 4 temperature features = 35
EDGE_DIM = 14

# ============================================================================
# Featurization helpers
# ============================================================================

def _one_hot(val, allowed, with_other=True):
    enc = [int(val == a) for a in allowed]
    if with_other:
        enc.append(int(val not in allowed))
    return enc


def _atom_features(atom, h_bond_info):
    """31-dim atom feature vector matching the MolMerger paper."""
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_TYPE_LIST, with_other=False)  # 10
    feats.append(float(atom.GetFormalCharge()))  # 1
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST, with_other=False)  # 3
    idx = atom.GetIdx()
    feats.append(float(idx in h_bond_info["acceptors"]))  # 1
    feats.append(float(idx in h_bond_info["donors"]))  # 1
    feats.append(float(atom.GetIsAromatic()))  # 1
    feats += _one_hot(atom.GetTotalDegree(), list(range(7)), with_other=False)  # 7
    feats += _one_hot(atom.GetTotalNumHs(), list(range(5)), with_other=False)  # 5
    tag = str(atom.GetChiralTag())
    feats.append(float("CW" in tag))  # 1
    feats.append(float("CCW" in tag))  # 1
    return feats


def _bond_features(bond, dist_matrix):
    """14-dim edge feature vector."""
    feats = []
    bt = bond.GetBondType()
    feats += _one_hot(bt, BOND_TYPE_LIST, with_other=True)  # 5
    feats.append(float(bond.IsInRing()))  # 1
    feats.append(float(bond.GetIsConjugated()))  # 1
    feats += _one_hot(bond.GetStereo(), STEREO_LIST, with_other=False)  # 6
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    if dist_matrix is not None and i < len(dist_matrix) and j < len(dist_matrix):
        feats.append(float(dist_matrix[i][j]))  # 1
    else:
        feats.append(0.0)
    return feats


_INTERACTION_BOND_FEATS = [
    0, 0, 0, 0, 1,        # other/INTERACTION slot
    0.0,                   # not in ring
    0.0,                   # not conjugated
    1, 0, 0, 0, 0, 0,     # STEREONONE
    1.0,                   # graph distance = 1
]


def _get_h_bond_info(mol):
    """Get H-bond acceptor and donor atom indices."""
    acceptors = set()
    donors = set()
    acc_smarts = Lipinski.HAcceptorSmarts
    if acc_smarts is not None:
        for match in mol.GetSubstructMatches(acc_smarts):
            acceptors.update(match)
    don_smarts = Lipinski.HDonorSmarts
    if don_smarts is not None:
        for match in mol.GetSubstructMatches(don_smarts):
            donors.update(match)
    return {"acceptors": acceptors, "donors": donors}


def _safe_gasteiger_charges(mol):
    """Compute Gasteiger charges, replacing NaN with 0."""
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        c = atom.GetDoubleProp("_GasteigerCharge")
        if math.isnan(c) or math.isinf(c):
            c = 0.0
        charges.append(c)
    return charges


# ============================================================================
# MolMerger graph construction — temperature-free skeleton
# ============================================================================

def molmerger_skeleton(solute_smi, solvent_smi):
    """Build a merged PyG Data *without* temperature (31-d nodes).

    Returns None if either molecule can't be parsed.  The returned Data
    has x of shape [N, 31] — temperature is stamped on at dataset-read time
    to avoid duplicating graphs across temperatures.
    """
    mol1 = Chem.MolFromSmiles(solute_smi)
    mol2 = Chem.MolFromSmiles(solvent_smi)
    if mol1 is None or mol2 is None:
        return None

    n1 = mol1.GetNumAtoms()
    n2 = mol2.GetNumAtoms()
    if n1 == 0 or n2 == 0:
        return None

    charges1 = _safe_gasteiger_charges(mol1)
    charges2 = _safe_gasteiger_charges(mol2)

    max_idx1 = int(np.argmax(charges1))
    min_idx1 = int(np.argmin(charges1))
    max_idx2 = int(np.argmax(charges2))
    min_idx2 = int(np.argmin(charges2))

    # --- Node features (no temperature) ---
    h_bond1 = _get_h_bond_info(mol1)
    h_bond2 = _get_h_bond_info(mol2)

    node_feats = []
    for atom in mol1.GetAtoms():
        node_feats.append(_atom_features(atom, h_bond1))
    for atom in mol2.GetAtoms():
        node_feats.append(_atom_features(atom, h_bond2))

    x = torch.tensor(node_feats, dtype=torch.float32)  # [N, 31]

    # --- Edge features ---
    dist1 = Chem.GetDistanceMatrix(mol1)
    dist2 = Chem.GetDistanceMatrix(mol2)

    src_list, dst_list, edge_feat_list = [], [], []

    for bond in mol1.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond, dist1)
        src_list += [i, j]
        dst_list += [j, i]
        edge_feat_list += [bf, bf]

    for bond in mol2.GetBonds():
        i, j = bond.GetBeginAtomIdx() + n1, bond.GetEndAtomIdx() + n1
        bf = _bond_features(bond, dist2)
        src_list += [i, j]
        dst_list += [j, i]
        edge_feat_list += [bf, bf]

    ibf = _INTERACTION_BOND_FEATS
    s1, d1 = max_idx1, min_idx2 + n1
    src_list += [s1, d1]
    dst_list += [d1, s1]
    edge_feat_list += [ibf, ibf]

    s2, d2 = min_idx1, max_idx2 + n1
    src_list += [s2, d2]
    dst_list += [d2, s2]
    edge_feat_list += [ibf, ibf]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_feat_list, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def stamp_temperature(skeleton: Data, temperature: float) -> Data:
    """Append 4 temperature features to every node in a skeleton graph.

    Returns a *new* Data object (skeleton is not mutated).
    """
    n = skeleton.x.size(0)
    T = temperature
    tf = torch.tensor(
        [T / 300.0, 1000.0 / T, (T / 300.0) ** 2, math.log(T / 300.0)],
        dtype=torch.float32,
    ).unsqueeze(0).expand(n, -1)         # [N, 4]
    x_full = torch.cat([skeleton.x, tf], dim=1)   # [N, 35]
    return Data(x=x_full, edge_index=skeleton.edge_index,
                edge_attr=skeleton.edge_attr)


# ============================================================================
# MolMerger Model (wraps PyG AttentiveFP)
# ============================================================================

class MolMergerNet(nn.Module):
    """AttentiveFP on merged solute-solvent graphs."""

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 200,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gnn = PyGAttentiveFP(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        graph_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return self.pred_head(graph_emb).squeeze(-1)


# ============================================================================
# Method wrapper (for registry, though runner is standalone)
# ============================================================================

class MolMergerMethod(BaseMethod):
    GNN_TYPE = "AttentiveFP"

    @classmethod
    def info(cls):
        return {
            "name": "MolMerger",
            "featurizer": "none",
            "mode": "merged_graph",
            "gpu_required": False,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError(
            "MolMerger requires the graph-based runner. "
            "Use scripts/run_molmerger.py"
        )

    def predict(self, X):
        raise NotImplementedError(
            "MolMerger requires the graph-based runner."
        )
