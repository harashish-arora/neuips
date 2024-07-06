"""
SolTranNet — Molecule Attention Transformer for solubility prediction.

Faithfully implements the MAT (Molecule Attention Transformer) architecture from:
  Maziarka et al. (2020) "Molecule Attention Transformer"
  https://github.com/ardigen/MAT

As adapted by SolTranNet (Francoeur & Koes, 2021) for solubility prediction.

Extended here for multi-solvent prediction with dual MAT encoders
(solute + solvent) fused with temperature features.

Architecture:
  - Atom-level featurization: 28-dim (27 atom features + 1 dummy node flag)
  - Adjacency-augmented multi-head attention: linearly interpolates between
    learned self-attention and normalized adjacency, controlled by lambda_attention
  - Pre-norm residual encoder blocks with LayerNorm
  - Global mean pooling over atoms (masked)
  - Dual encoders: separate MAT for solute and solvent
  - Fusion MLP: [solute_emb || solvent_emb || temp_feats] -> LogS
"""

import math
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit.Chem import MolFromSmiles

from .base import BaseMethod


# ============================================================================
# Column resolution helpers — tolerant to bench/SC3 naming differences
# ============================================================================

def _resolve_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} in df columns: {list(df.columns)}")
    return None


def get_solute_col(df):
    return _resolve_col(df, ["Solute_Canon", "Solute", "Solute_SMILES"])


def get_solvent_col(df):
    return _resolve_col(df, ["Solvent_Canon", "Solvent", "Solvent_SMILES"])


def get_temp_col(df):
    return _resolve_col(df, ["Temperature_K", "Temperature", "T"])


def get_target_col(df, required=True):
    return _resolve_col(df, ["LogS", "LogS_consensus"], required=required)


# ============================================================================
# Atom Featurization (from SolTranNet / MAT data_utils.py)
# ============================================================================

# Element one-hot lookup: B, C, N, O, F, P, S, Cl, Br, I -> indices 0-9
# Index 10 = "Other", index 11 = "Dummy"
_ANUM_MAP = {5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
_ANUM_TABLE = np.full(128, 10, dtype=np.int32)
for _anum, _idx in _ANUM_MAP.items():
    _ANUM_TABLE[_anum] = _idx

# Feature dimensions:
#   Element identity:    12 (B,C,N,O,F,P,S,Cl,Br,I,Other,Dummy)
#   Heavy neighbors:      6 (0-5)
#   H atoms:              5 (0-4)
#   Formal charge:        3 (-1, 0, +1)
#   In ring:              1
#   Aromatic:             1
# Total: 28
D_ATOM = 28


def _get_atom_features(atom):
    """Compute 27-dim atom feature vector (without dummy flag)."""
    attrs = np.zeros(27, dtype=np.float32)

    anum = atom.GetAtomicNum()
    anum_idx = _ANUM_TABLE[anum] if anum < 128 else 10
    attrs[anum_idx] = 1.0

    n_neighbors = min(len(atom.GetNeighbors()), 5)
    attrs[11 + n_neighbors] = 1.0

    n_hs = min(atom.GetTotalNumHs(), 4)
    attrs[17 + n_hs] = 1.0

    charge = atom.GetFormalCharge()
    if charge == 0:
        attrs[23] = 1.0
    elif charge < 0:
        attrs[22] = 1.0
    else:
        attrs[24] = 1.0

    attrs[25] = float(atom.IsInRing())
    attrs[26] = float(atom.GetIsAromatic())
    return attrs


def featurize_mol(smiles, add_dummy_node=True):
    """Convert SMILES to (node_features, adjacency_matrix).

    Returns:
        node_features: (n_atoms, D_ATOM) float32 array
        adj_matrix: (n_atoms, n_atoms) float32 binary adjacency with self-loops
        Returns None if SMILES is invalid.
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    node_features = np.array([_get_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    adj = np.eye(n, dtype=np.float32)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    if add_dummy_node:
        nf_padded = np.zeros((n + 1, D_ATOM), dtype=np.float32)
        nf_padded[1:, 1:] = node_features
        nf_padded[0, 0] = 1.0
        node_features = nf_padded

        adj_padded = np.zeros((n + 1, n + 1), dtype=np.float32)
        adj_padded[1:, 1:] = adj
        adj_padded[0, 0] = 1.0
        adj = adj_padded

    return node_features, adj


# ============================================================================
# Graph Transformer (MAT) Architecture
# ============================================================================

def _clones(module, n):
    """Produce N independent copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    """Pre-norm residual connection: x + dropout(sublayer(norm(x)))."""

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

def mat_attention(query, key, value, adj_matrix,
                  mask=None, dropout=None, lambdas=(0.5, 0.5),
                  eps=1e-6):
    """Molecule Attention Transformer attention.

    Linearly interpolates between standard scaled dot-product attention
    and row-normalized adjacency matrix:
        p_weighted = lambda_attn * softmax(QK^T / sqrt(d)) + lambda_adj * norm(A)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(
            mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1) == 0,
            mask_value,
        )

    p_attn = F.softmax(scores, dim=-1)

    # Row-normalize adjacency and broadcast over heads
    adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + eps)
    adj_norm = adj_norm.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

    lambda_attn, lambda_adj = lambdas
    p_weighted = lambda_attn * p_attn + lambda_adj * adj_norm

    if dropout is not None:
        p_weighted = dropout(p_weighted)

    out = torch.matmul(p_weighted, value)
    return out, p_weighted


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lambda_attention=0.5):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.lambdas = (lambda_attention, 1.0 - lambda_attention)
        self.linears = _clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, adj_matrix, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, _ = mat_attention(
            query,
            key,
            value,
            adj_matrix,
            mask=mask,
            dropout=self.dropout,
            lambdas=self.lambdas,
        )

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0,
                 dense_output_nonlinearity="relu"):
        super().__init__()
        self.N_dense = N_dense
        self.linears = _clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = _clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope

        if dense_output_nonlinearity == "relu":
            self.output_fn = lambda x: F.leaky_relu(x, negative_slope=leaky_relu_slope)
        elif dense_output_nonlinearity == "tanh":
            self.output_fn = torch.tanh
        else:
            self.output_fn = lambda x: x

    def forward(self, x):
        if self.N_dense == 0:
            return x
        for i in range(len(self.linears) - 1):
            x = self.dropout[i](F.leaky_relu(
                self.linears[i](x),
                negative_slope=self.leaky_relu_slope,
            ))
        return self.dropout[-1](self.output_fn(self.linears[-1](x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = _clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super().__init__()
        self.layers = _clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, adj_matrix):
        for layer in self.layers:
            x = layer(x, mask, adj_matrix)
        return self.norm(x)


class Embeddings(nn.Module):
    """Linear projection from atom features to d_model."""

    def __init__(self, d_model, d_atom, dropout):
        super().__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


class MATEncoder(nn.Module):
    """Complete Molecule Attention Transformer encoder.

    Takes (node_features, mask, adj_matrix) and returns a single
    d_model-dimensional molecular embedding via masked mean pooling.
    """

    def __init__(self, d_atom=D_ATOM, N=4, d_model=64, h=4, dropout=0.1,
                 lambda_attention=0.5, N_dense=1, leaky_relu_slope=0.0,
                 dense_output_nonlinearity="relu", aggregation_type="mean"):
        super().__init__()
        attn = MultiHeadedAttention(h, d_model, dropout, lambda_attention)
        ff = PositionwiseFeedForward(
            d_model,
            N_dense,
            dropout,
            leaky_relu_slope,
            dense_output_nonlinearity,
        )
        layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout)
        self.encoder = Encoder(layer, N)
        self.src_embed = Embeddings(d_model, d_atom, dropout)
        self.aggregation_type = aggregation_type
        self.d_model = d_model

    def forward(self, src, src_mask, adj_matrix):
        x = self.encoder(self.src_embed(src), src_mask, adj_matrix)

        mask = src_mask.unsqueeze(-1).float()
        out_masked = x * mask

        if self.aggregation_type == "mean":
            return out_masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if self.aggregation_type == "sum":
            return out_masked.sum(dim=1)
        return out_masked[:, 0]


# ============================================================================
# Dual-Encoder Model for Multi-Solvent Solubility
# ============================================================================

class DualSolTranNet(nn.Module):
    """Dual MAT encoder for solute-solvent solubility prediction."""

    def __init__(self, d_atom=D_ATOM, N=4, d_model=64, h=4, dropout=0.1,
                 lambda_attention=0.5, N_dense=1, num_temp_feats=4,
                 n_generator_layers=2):
        super().__init__()
        self.solute_enc = MATEncoder(
            d_atom=d_atom,
            N=N,
            d_model=d_model,
            h=h,
            dropout=dropout,
            lambda_attention=lambda_attention,
            N_dense=N_dense,
        )
        self.solvent_enc = MATEncoder(
            d_atom=d_atom,
            N=N,
            d_model=d_model,
            h=h,
            dropout=dropout,
            lambda_attention=lambda_attention,
            N_dense=N_dense,
        )

        mlp_in = d_model * 2 + num_temp_feats
        if n_generator_layers == 1:
            self.mlp = nn.Linear(mlp_in, 1)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_in, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, solute_feats, solute_mask, solute_adj,
                solvent_feats, solvent_mask, solvent_adj, temp_feats):
        sol_emb = self.solute_enc(solute_feats, solute_mask, solute_adj)
        solv_emb = self.solvent_enc(solvent_feats, solvent_mask, solvent_adj)
        combined = torch.cat([sol_emb, solv_emb, temp_feats], dim=1)
        return self.mlp(combined).squeeze(-1)


# ============================================================================
# Dataset and Collation
# ============================================================================

class SolTranNetDataset(Dataset):
    """Dataset that converts benchmark rows to graph features for SolTranNet."""

    def __init__(self, df, graph_cache=None, target_col=None):
        self.df = df.reset_index(drop=True)
        self.graph_cache = graph_cache if graph_cache is not None else {}

        self.solute_col = get_solute_col(self.df)
        self.solvent_col = get_solvent_col(self.df)
        self.temp_col = get_temp_col(self.df)
        self.target_col = target_col or get_target_col(self.df, required=False)

    def __len__(self):
        return len(self.df)

    def _get_graph(self, smiles):
        if smiles not in self.graph_cache:
            result = featurize_mol(smiles, add_dummy_node=True)
            if result is None:
                logging.warning(f"Invalid SMILES: {smiles}, using dummy graph")
                nf = np.zeros((1, D_ATOM), dtype=np.float32)
                nf[0, 0] = 1.0
                adj = np.ones((1, 1), dtype=np.float32)
                result = (nf, adj)
            self.graph_cache[smiles] = result
        return self.graph_cache[smiles]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        solute_nf, solute_adj = self._get_graph(row[self.solute_col])
        solvent_nf, solvent_adj = self._get_graph(row[self.solvent_col])

        T = float(row[self.temp_col])
        temp_feats = np.array(
            [T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)],
            dtype=np.float32,
        )

        if self.target_col is None:
            target = np.float32(0.0)
        else:
            target = np.float32(row[self.target_col])

        return solute_nf, solute_adj, solvent_nf, solvent_adj, temp_feats, target


def soltrannet_collate_fn(batch):
    """Collate variable-size molecular graphs into padded batches."""
    sol_nfs, sol_adjs, solv_nfs, solv_adjs, temp_list, tgt_list = zip(*batch)

    def _pad_graphs(node_feats_list, adj_list):
        max_n = max(nf.shape[0] for nf in node_feats_list)
        d = node_feats_list[0].shape[1]
        batch_size = len(node_feats_list)

        feats_padded = np.zeros((batch_size, max_n, d), dtype=np.float32)
        adj_padded = np.zeros((batch_size, max_n, max_n), dtype=np.float32)
        masks = np.zeros((batch_size, max_n), dtype=np.float32)

        for i, (nf, adj) in enumerate(zip(node_feats_list, adj_list)):
            n = nf.shape[0]
            feats_padded[i, :n, :] = nf
            adj_padded[i, :n, :n] = adj
            masks[i, :n] = 1.0

        return (
            torch.from_numpy(feats_padded),
            torch.from_numpy(adj_padded),
            torch.from_numpy(masks),
        )

    sol_feats, sol_adj, sol_mask = _pad_graphs(sol_nfs, sol_adjs)
    solv_feats, solv_adj, solv_mask = _pad_graphs(solv_nfs, solv_adjs)

    temp_feats = torch.from_numpy(np.stack(temp_list))
    targets = torch.tensor(tgt_list, dtype=torch.float32)

    return (
        sol_feats,
        sol_adj,
        sol_mask,
        solv_feats,
        solv_adj,
        solv_mask,
        temp_feats,
        targets,
    )


# ============================================================================
# Method Wrapper
# ============================================================================

class SolTranNetMethod(BaseMethod):
    """Benchmark wrapper for SolTranNet.

    The dedicated runner should be used for full SC3 reporting, but this wrapper
    keeps the method registry functional and supports DataFrame-based fit/predict.
    """

    @classmethod
    def info(cls):
        return {
            "name": "SolTranNet",
            "featurizer": "none",
            "mode": "transformer",
            "gpu_required": True,
        }

    def __init__(
        self,
        seed=42,
        device=None,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        patience=15,
        d_model=64,
        N=4,
        h=4,
        dropout=0.1,
        lambda_attention=0.5,
        N_dense=1,
        num_workers=0,
    ):
        super().__init__(seed=seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.num_workers = num_workers
        self.model_kwargs = {
            "d_model": d_model,
            "N": N,
            "h": h,
            "dropout": dropout,
            "lambda_attention": lambda_attention,
            "N_dense": N_dense,
        }
        self.graph_cache = {}
        self.model = None

    def _make_loader(self, df, shuffle):
        ds = SolTranNetDataset(df, graph_cache=self.graph_cache)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=soltrannet_collate_fn,
        )

    def _forward_batch(self, batch):
        batch = [x.to(self.device) for x in batch]
        sol_feats, sol_adj, sol_mask, solv_feats, solv_adj, solv_mask, temp_feats, targets = batch
        preds = self.model(
            sol_feats,
            sol_mask,
            sol_adj,
            solv_feats,
            solv_mask,
            solv_adj,
            temp_feats,
        )
        return preds, targets

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        train_df = X_train.copy()
        if y_train is not None and get_target_col(train_df, required=False) is None:
            train_df["LogS"] = np.asarray(y_train, dtype=np.float32)

        val_df = None
        if X_val is not None:
            val_df = X_val.copy()
            if y_val is not None and get_target_col(val_df, required=False) is None:
                val_df["LogS"] = np.asarray(y_val, dtype=np.float32)

        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader = self._make_loader(val_df, shuffle=False) if val_df is not None else None

        self.model = DualSolTranNet(**self.model_kwargs).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            factor=0.5,
            min_lr=1e-6,
        )

        best_loss = float("inf")
        best_state = None
        wait = 0

        for _epoch in range(self.epochs):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                preds, targets = self._forward_batch(batch)
                loss = F.mse_loss(preds, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            losses = []
            with torch.no_grad():
                for batch in val_loader:
                    preds, targets = self._forward_batch(batch)
                    losses.append(F.mse_loss(preds, targets).item() * targets.numel())

            val_loss = float(np.sum(losses) / len(val_df))
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        loader = self._make_loader(X, shuffle=False)
        self.model.eval()

        preds_all = []
        with torch.no_grad():
            for batch in loader:
                preds, _targets = self._forward_batch(batch)
                preds_all.append(preds.detach().cpu().numpy())

        return np.concatenate(preds_all)


_SolTranNetMethod = SolTranNetMethod
