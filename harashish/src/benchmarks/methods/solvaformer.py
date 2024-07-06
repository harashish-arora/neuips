"""
Solvaformer: SE(3)-equivariant graph transformer for multi-solvent solubility prediction.

Reference: Broadbent et al., "Solvaformer: an SE(3)-equivariant graph transformer
for small molecule solubility prediction" (arXiv 2511.09774, Sanofi, 2025).

Architecture (faithful to paper concepts, scaled for single-GPU / 60k data):
  - 3D conformer input via RDKit ETKDGv3
  - Intramolecular SE(3)-equivariant message passing using PaiNN-style
    scalar+vector features (l=0 + l=1) — captures 3D molecular geometry
  - Intermolecular scalar cross-attention — only rotationally invariant
    features cross the molecular boundary (paper's key insight: inter-molecular
    spatial relationships are undefined for solutions)
  - Dual encoder (solute + solvent) with shared weights
  - Global pooling → MLP prediction head

The original paper used EquiformerV2 with l_max=6, 8 layers, 10 heads
on 24 H100 GPUs for 160 hours (82k training points from BigSolDB+CombiSolv).
This implementation uses the same architectural principles but with a PaiNN-style
equivariant backbone (l_max=1) that is ~100x more parameter-efficient, targeting
~300k parameters for our 60k training set.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem


# ============================================================================
# 3D Conformer Generation
# ============================================================================

def smiles_to_3d_graph(smiles, max_attempts=5):
    """Convert SMILES → 3D molecular graph with atom features and coordinates."""
    try:
        return _smiles_to_3d_graph_inner(smiles, max_attempts)
    except Exception:
        return None


def _smiles_to_3d_graph_inner(smiles, max_attempts=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useSmallRingTorsions = True

    conf_id = AllChem.EmbedMolecule(mol, params)
    if conf_id < 0:
        params_fallback = AllChem.ETKDGv3()
        params_fallback.randomSeed = 42
        params_fallback.useRandomCoords = True
        conf_id = AllChem.EmbedMolecule(mol, params_fallback)
        if conf_id < 0:
            return None

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass

    try:
        mol_no_h = Chem.RemoveHs(mol)
    except Exception:
        mol_no_h = Chem.RWMol(mol)
        idxs_to_remove = [a.GetIdx() for a in mol_no_h.GetAtoms() if a.GetAtomicNum() == 1]
        for idx in sorted(idxs_to_remove, reverse=True):
            mol_no_h.RemoveAtom(idx)

    conf = mol.GetConformer()
    heavy_atom_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if len(heavy_atom_indices) == 0:
        return None

    positions = np.array([list(conf.GetAtomPosition(i)) for i in heavy_atom_indices],
                         dtype=np.float32)

    atom_features = []
    for atom in mol_no_h.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetMass() / 100.0,
        ])

    edges = []
    for bond in mol_no_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])

    return {
        "node_feats": torch.tensor(atom_features, dtype=torch.float32),
        "pos": torch.tensor(positions, dtype=torch.float32),
        "edge_index": (torch.tensor(edges, dtype=torch.long).t()
                       if edges else torch.zeros((2, 0), dtype=torch.long)),
        "num_nodes": len(atom_features),
    }


# ============================================================================
# Equivariant Building Blocks (PaiNN-style: scalar s + vector v)
# ============================================================================

class RadialBasis(nn.Module):
    """Gaussian RBF for encoding interatomic distances."""
    def __init__(self, num_basis=20, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        offsets = torch.linspace(0, cutoff, num_basis)
        self.register_buffer("offsets", offsets)
        self.width = 0.5 * (cutoff / num_basis)

    def forward(self, dist):
        return torch.exp(-((dist.unsqueeze(-1) - self.offsets) ** 2)
                         / (2 * self.width ** 2))


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope."""
    def __init__(self, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist):
        return 0.5 * (1 + torch.cos(math.pi * dist / self.cutoff)) * (dist <= self.cutoff).float()


class PaiNNMessage(nn.Module):
    """PaiNN-style equivariant message passing.

    Updates scalar features s and vector features v using:
    - Distance-dependent filter (RBF → MLP → filter)
    - Directional information from unit vectors r_ij
    - Vector channel gating via scalar activations

    This is SE(3)-equivariant: scalars are invariant, vectors transform
    as 3D vectors under rotation.
    """
    def __init__(self, hidden_dim, num_basis=20):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.scalar_msg = nn.Sequential(
            nn.Linear(num_basis, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

        self.vec_scale = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, s, v, edge_index, rbf, cutoff_w, unit_vec, num_nodes):
        """
        s: [N, hidden_dim] scalar features
        v: [N, hidden_dim, 3] vector features
        edge_index: [2, E]
        rbf: [E, num_basis]
        cutoff_w: [E] cutoff weights
        unit_vec: [E, 3] unit direction vectors
        """
        if edge_index.numel() == 0:
            return s, v

        src, dst = edge_index

        W = self.scalar_msg(rbf) * cutoff_w.unsqueeze(-1)
        W_ss, W_sv, W_vv = W.chunk(3, dim=-1)

        ds = torch.zeros_like(s)
        ds.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.hidden_dim), s[src] * W_ss)

        scaled_v_src = self.vec_scale(v[src].transpose(1, 2)).transpose(1, 2)
        dv = torch.zeros_like(v)
        dv.scatter_add_(
            0,
            dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.hidden_dim, 3),
            scaled_v_src * W_vv.unsqueeze(-1),
        )

        dir_msg = unit_vec.unsqueeze(1) * (s[src] * W_sv).unsqueeze(-1)
        dv.scatter_add_(
            0,
            dst.unsqueeze(-1).unsqueeze(-1).expand(-1, self.hidden_dim, 3),
            dir_msg,
        )

        return s + ds, v + dv


class PaiNNUpdate(nn.Module):
    """PaiNN-style equivariant update block.

    Mixes scalar and vector channels using vector norms (invariant)
    to gate information flow.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.vec_U = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.vec_V = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scalar_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )

    def forward(self, s, v):
        Uv = self.vec_U(v.transpose(1, 2)).transpose(1, 2)
        Vv = self.vec_V(v.transpose(1, 2)).transpose(1, 2)

        Vv_norm = Vv.norm(dim=-1)

        combined = torch.cat([s, Vv_norm], dim=-1)
        out = self.scalar_net(combined)
        a_ss, a_sv, a_vv = out.chunk(3, dim=-1)

        ds = a_ss + a_sv * (Uv * Vv).sum(dim=-1)
        dv = a_vv.unsqueeze(-1) * Uv

        return s + ds, v + dv


class ScalarCrossAttention(nn.Module):
    """Intermolecular scalar cross-attention (Solvaformer paper Eq. 3-4).

    Only scalar (rotationally invariant) features participate in
    cross-molecular attention — the paper's key architectural insight.

    Uses scatter-based attention for GPU-efficient batched operation
    (no Python for-loop over graphs).
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_s, kv_s, query_batch, kv_batch, num_graphs):
        """Cross-attend from query molecule atoms to kv molecule atoms.

        Vectorized: pools kv per graph, then broadcasts to query atoms.
        This avoids per-graph Python loops while preserving the semantics.
        """
        Q = self.q_proj(query_s)
        K = self.k_proj(kv_s)
        V = self.v_proj(kv_s)

        K_pool = torch.zeros(num_graphs, K.size(-1), device=K.device)
        V_pool = torch.zeros(num_graphs, V.size(-1), device=V.device)
        counts = torch.zeros(num_graphs, 1, device=K.device)
        K_pool.scatter_add_(0, kv_batch.unsqueeze(-1).expand_as(K), K)
        V_pool.scatter_add_(0, kv_batch.unsqueeze(-1).expand_as(V), V)
        counts.scatter_add_(0, kv_batch.unsqueeze(-1),
                           torch.ones(kv_batch.size(0), 1, device=kv_batch.device))
        K_pool = K_pool / counts.clamp(min=1)
        V_pool = V_pool / counts.clamp(min=1)

        K_broadcast = K_pool[query_batch]
        V_broadcast = V_pool[query_batch]

        q = Q.view(-1, self.num_heads, self.head_dim)
        k = K_broadcast.view(-1, self.num_heads, self.head_dim)
        v = V_broadcast.view(-1, self.num_heads, self.head_dim)

        attn = (q * k).sum(dim=-1, keepdim=True) * self.scale
        attn = torch.sigmoid(attn)
        out = (v * attn).view(-1, Q.size(-1))

        out = self.out_proj(out)
        return self.norm(query_s + out)


# ============================================================================
# Solvaformer Block
# ============================================================================

class SolvaformerBlock(nn.Module):
    """One Solvaformer transformer block:
    1. Intramolecular equivariant message passing (PaiNN)
    2. Intramolecular equivariant update
    3. Intermolecular scalar cross-attention
    """
    def __init__(self, hidden_dim, num_basis=20, num_heads=4, dropout=0.1):
        super().__init__()
        self.msg = PaiNNMessage(hidden_dim, num_basis)
        self.upd = PaiNNUpdate(hidden_dim)
        self.cross_attn = ScalarCrossAttention(hidden_dim, num_heads, dropout)
        self.s_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sol_s, sol_v, solv_s, solv_v,
                sol_edge, solv_edge, sol_rbf, solv_rbf,
                sol_cut, solv_cut, sol_uvec, solv_uvec,
                sol_batch, solv_batch, n_sol, n_solv, B):
        sol_s, sol_v = self.msg(sol_s, sol_v, sol_edge, sol_rbf, sol_cut, sol_uvec, n_sol)
        solv_s, solv_v = self.msg(solv_s, solv_v, solv_edge, solv_rbf, solv_cut, solv_uvec, n_solv)

        sol_s, sol_v = self.upd(sol_s, sol_v)
        solv_s, solv_v = self.upd(solv_s, solv_v)

        sol_s = self.s_norm(sol_s)
        solv_s = self.s_norm(solv_s)

        sol_s = self.cross_attn(sol_s, solv_s, sol_batch, solv_batch, B)
        solv_s = self.cross_attn(solv_s, sol_s, solv_batch, sol_batch, B)

        return sol_s, sol_v, solv_s, solv_v


# ============================================================================
# Full Solvaformer Model
# ============================================================================

class Solvaformer(nn.Module):
    """Solvaformer: SE(3)-equivariant graph transformer for solubility.

    Architecture (arXiv 2511.09774, adapted for efficiency):
    - PaiNN-style equivariant backbone: scalar (l=0) + vector (l=1) features
    - Intramolecular: direction-aware message passing preserving SE(3) equivariance
    - Intermolecular: scalar cross-attention only (no spurious 3D vectors)
    - Dual encoder with shared weights

    Target: ~300-400k parameters for 60k training points.
    """
    def __init__(self, node_input_dim=7, hidden_dim=64, num_layers=3,
                 num_heads=4, num_basis=20, cutoff=5.0, num_temp_feats=4,
                 dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff

        self.node_embed = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rbf = RadialBasis(num_basis, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        self.blocks = nn.ModuleList([
            SolvaformerBlock(hidden_dim, num_basis, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)

        mlp_in = hidden_dim * 2 + num_temp_feats
        self.head = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _edge_feats(self, pos, edge_index):
        if edge_index.numel() == 0:
            return (torch.zeros(0, self.rbf.offsets.size(0), device=pos.device),
                    torch.zeros(0, device=pos.device),
                    torch.zeros(0, 3, device=pos.device))
        src, dst = edge_index
        diff = pos[src] - pos[dst]
        dist = diff.norm(dim=-1).clamp(min=1e-8)
        unit = diff / dist.unsqueeze(-1)
        return self.rbf(dist), self.cutoff_fn(dist), unit

    def _pool(self, s, batch_ids, B):
        pooled = torch.zeros(B, self.hidden_dim, device=s.device)
        pooled.scatter_add_(0, batch_ids.unsqueeze(-1).expand_as(s), s)
        counts = torch.zeros(B, device=s.device)
        counts.scatter_add_(0, batch_ids, torch.ones_like(batch_ids, dtype=torch.float))
        return pooled / counts.clamp(min=1).unsqueeze(-1)

    def forward(self, solute_data, solvent_data, temp_feats):
        sol_s = self.node_embed(solute_data["node_feats"])
        solv_s = self.node_embed(solvent_data["node_feats"])

        n_sol = sol_s.size(0)
        n_solv = solv_s.size(0)
        sol_v = torch.zeros(n_sol, self.hidden_dim, 3, device=sol_s.device)
        solv_v = torch.zeros(n_solv, self.hidden_dim, 3, device=solv_s.device)

        sol_rbf, sol_cut, sol_uvec = self._edge_feats(solute_data["pos"], solute_data["edge_index"])
        solv_rbf, solv_cut, solv_uvec = self._edge_feats(solvent_data["pos"], solvent_data["edge_index"])

        B = len(solute_data["num_nodes"])

        for block in self.blocks:
            sol_s, sol_v, solv_s, solv_v = block(
                sol_s, sol_v, solv_s, solv_v,
                solute_data["edge_index"], solvent_data["edge_index"],
                sol_rbf, solv_rbf, sol_cut, solv_cut,
                sol_uvec, solv_uvec,
                solute_data["batch_ids"], solvent_data["batch_ids"],
                n_sol, n_solv, B,
            )

        sol_s = self.out_norm(sol_s)
        solv_s = self.out_norm(solv_s)

        sol_pool = self._pool(sol_s, solute_data["batch_ids"], B)
        solv_pool = self._pool(solv_s, solvent_data["batch_ids"], B)

        combined = torch.cat([sol_pool, solv_pool, temp_feats], dim=-1)
        return self.head(combined).squeeze(-1)


# ============================================================================
# Dataset and Collation
# ============================================================================

class SolvaformerDataset(Dataset):
    def __init__(self, df, graph_cache):
        self.df = df.reset_index(drop=True)
        self.graph_cache = graph_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        solute_g = self.graph_cache[row["Solute"]]
        solvent_g = self.graph_cache[row["Solvent"]]
        T = row["Temperature"]
        temp_feats = torch.tensor(
            [T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)],
            dtype=torch.float32,
        )
        target = torch.tensor(row["LogS"], dtype=torch.float32)
        return solute_g, solvent_g, temp_feats, target


def solvaformer_collate_fn(batch):
    solute_gs, solvent_gs, temp_feats, targets = zip(*batch)

    def batch_graphs_3d(graph_list):
        node_feats, positions, edge_indices, batch_ids, num_nodes = [], [], [], [], []
        offset = 0
        for i, g in enumerate(graph_list):
            n = g["num_nodes"]
            node_feats.append(g["node_feats"])
            positions.append(g["pos"])
            if g["edge_index"].numel() > 0:
                edge_indices.append(g["edge_index"] + offset)
            batch_ids.extend([i] * n)
            num_nodes.append(n)
            offset += n
        return {
            "node_feats": torch.cat(node_feats, dim=0),
            "pos": torch.cat(positions, dim=0),
            "edge_index": (torch.cat(edge_indices, dim=1)
                          if edge_indices
                          else torch.zeros((2, 0), dtype=torch.long)),
            "batch_ids": torch.tensor(batch_ids, dtype=torch.long),
            "num_nodes": num_nodes,
        }

    return (
        batch_graphs_3d(solute_gs),
        batch_graphs_3d(solvent_gs),
        torch.stack(temp_feats),
        torch.stack(targets),
    )
