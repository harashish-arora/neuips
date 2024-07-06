"""
Graph Neural Network models for SC3 benchmark.

Implements dual-encoder GCN, GAT, GIN, MPNN, AttentiveFP.
Requires: torch, torch_geometric (PyG) or dgl.

These models are GPU-preferred but can run on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem

BaseMethod = object


# ============================================================================
# Graph Construction (no dependency on PyG/DGL — pure torch)
# ============================================================================

def smiles_to_graph(smiles):
    """Convert SMILES to a simple graph representation (adjacency + node features)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: [atomic_num, degree, formal_charge, num_Hs, aromatic, in_ring]
    node_feats = []
    for atom in mol.GetAtoms():
        node_feats.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetMass() / 100.0,
        ])
    node_feats = torch.tensor(node_feats, dtype=torch.float32)

    # Edge index (COO format)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {"node_feats": node_feats, "edge_index": edge_index, "num_nodes": len(node_feats)}


# ============================================================================
# Simple GNN Layers (pure PyTorch, no external GNN library needed)
# ============================================================================

class GCNLayer(nn.Module):
    """Basic Graph Convolutional layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, num_nodes, precomputed_norm=None):
        # Transform first
        h = self.linear(x)

        if precomputed_norm is None:
            row, col, norm = GNNEncoder.precompute_gcn_norm(edge_index, num_nodes, x.device)
        else:
            row, col, norm = precomputed_norm

        # Message passing: D^{-1/2} A D^{-1/2} X W (with cached topology norm)
        out = torch.zeros_like(h)
        msg = h[col] * norm.unsqueeze(-1)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        return torch.relu(out)


class GATLayer(nn.Module):
    """Basic Graph Attention layer."""
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, num_nodes):
        h = self.W(x).view(-1, self.num_heads, self.head_dim)
        row, col = edge_index

        # Attention scores
        alpha_src = (h[row] * self.a_src).sum(-1)
        alpha_dst = (h[col] * self.a_dst).sum(-1)
        alpha = self.leaky_relu(alpha_src + alpha_dst)

        # Softmax per node
        alpha_max = torch.zeros(num_nodes, self.num_heads, device=x.device)
        alpha_max.scatter_reduce_(0, row.unsqueeze(-1).expand_as(alpha), alpha, reduce="amax")
        alpha = torch.exp(alpha - alpha_max[row])
        alpha_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        alpha_sum.scatter_add_(0, row.unsqueeze(-1).expand_as(alpha), alpha)
        alpha = alpha / (alpha_sum[row] + 1e-8)

        # Aggregate
        msg = h[col] * alpha.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).unsqueeze(-1).expand_as(msg), msg)

        return torch.relu(out.view(num_nodes, -1))


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index
        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), x[col])
        out = self.mlp((1 + self.eps) * x + agg)
        return out


class TAGConvLayer(nn.Module):
    """Topology Adaptive Graph Convolutional layer (Du et al., 2017).

    Implements: H^K = Σ_{k=0}^K (D^{-1/2} A D^{-1/2})^k X Θ_k

    Each hop k propagates features through the normalized adjacency.
    All K+1 feature matrices are concatenated and projected through a
    single linear layer of size [in_dim * (K+1), out_dim].

    Unlike GCN, no self-loops are added — the k=0 term (raw features)
    serves that role. This polynomial filter adapts to the graph topology.
    """
    def __init__(self, in_dim, out_dim, k=2):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_dim * (k + 1), out_dim)
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, x, edge_index, num_nodes):
        if edge_index.numel() == 0:
            fstack = [x] * (self.k + 1)
            return torch.relu(self.linear(torch.cat(fstack, dim=-1)))

        row, col = edge_index

        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5).unsqueeze(-1)

        fstack = [x]
        for _ in range(self.k):
            h = fstack[-1] * deg_inv_sqrt
            out = torch.zeros_like(h)
            out.scatter_add_(0, row.unsqueeze(-1).expand_as(h[col]), h[col])
            out = out * deg_inv_sqrt
            fstack.append(out)

        return torch.relu(self.linear(torch.cat(fstack, dim=-1)))


# ============================================================================
# GNN Encoder
# ============================================================================

class GNNEncoder(nn.Module):
    """Multi-layer GNN encoder with global mean pooling."""
    def __init__(self, in_dim, hidden_dim, num_layers, gnn_type="GCN", tagconv_k=2):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_map = {
            "GCN": lambda i, o: GCNLayer(i, o),
            "GAT": lambda i, o: GATLayer(i, o),
            "GIN": lambda i, o: GINLayer(i, o),
            "TAGConv": lambda i, o: TAGConvLayer(i, o, k=tagconv_k),
        }
        if gnn_type not in layer_map:
            raise ValueError(f"Unknown gnn_type: {gnn_type}. Choose from {list(layer_map.keys())}")
        make_layer = layer_map[gnn_type]
        self.layers.append(make_layer(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_layer(hidden_dim, hidden_dim))
        self.gnn_type = gnn_type

    @staticmethod
    def precompute_gcn_norm(edge_index, num_nodes, device):
        """Precompute normalized adjacency terms shared across all GCN layers."""
        self_loops = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).to(dtype=torch.float32)
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return row, col, norm

    def forward(self, node_feats, edge_index, batch_ids, num_nodes_per_graph):
        h = node_feats
        gcn_norm = None
        if self.gnn_type == "GCN":
            gcn_norm = self.precompute_gcn_norm(edge_index, h.size(0), h.device)

        for layer in self.layers:
            if self.gnn_type == "GCN":
                h = layer(h, edge_index, h.size(0), precomputed_norm=gcn_norm)
            else:
                h = layer(h, edge_index, h.size(0))
        # Global mean pooling
        num_graphs = len(num_nodes_per_graph)
        graph_embeds = torch.zeros(num_graphs, h.size(1), device=h.device)
        graph_embeds.scatter_add_(0, batch_ids.unsqueeze(-1).expand_as(h), h)
        counts = torch.tensor(num_nodes_per_graph, dtype=torch.float32, device=h.device).unsqueeze(-1)
        return graph_embeds / counts


# ============================================================================
# Dual-Encoder Model
# ============================================================================

class DualGNNSolubility(nn.Module):
    """Dual GNN for solute-solvent solubility prediction."""
    def __init__(self, node_dim=7, hidden_dim=64, num_layers=3, gnn_type="GCN", num_temp_feats=4, tagconv_k=2):
        super().__init__()
        self.solute_enc = GNNEncoder(node_dim, hidden_dim, num_layers, gnn_type, tagconv_k=tagconv_k)
        self.solvent_enc = GNNEncoder(node_dim, hidden_dim, num_layers, gnn_type, tagconv_k=tagconv_k)
        mlp_in = hidden_dim * 2 + num_temp_feats
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, solute_data, solvent_data, temp_feats):
        sol_emb = self.solute_enc(
            solute_data["node_feats"], solute_data["edge_index"],
            solute_data["batch_ids"], solute_data["num_nodes"]
        )
        solv_emb = self.solvent_enc(
            solvent_data["node_feats"], solvent_data["edge_index"],
            solvent_data["batch_ids"], solvent_data["num_nodes"]
        )
        combined = torch.cat([sol_emb, solv_emb, temp_feats], dim=1)
        return self.mlp(combined).squeeze(-1)


# ============================================================================
# Dataset and Collation
# ============================================================================

class SolubilityGraphDataset(Dataset):
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
        temp_feats = torch.tensor([T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)], dtype=torch.float32)
        target = torch.tensor(row["LogS"], dtype=torch.float32)
        return solute_g, solvent_g, temp_feats, target


def batch_graph_list(graph_list, device=None):
    """Batch a list of graph dicts into a single batched graph dict."""
    node_feats = []
    edge_indices = []
    batch_ids = []
    num_nodes = []
    offset = 0
    for i, g in enumerate(graph_list):
        n = g["num_nodes"]
        node_feats.append(g["node_feats"])
        if g["edge_index"].numel() > 0:
            edge_indices.append(g["edge_index"] + offset)
        batch_ids.extend([i] * n)
        num_nodes.append(n)
        offset += n
    result = {
        "node_feats": torch.cat(node_feats, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long),
        "batch_ids": torch.tensor(batch_ids, dtype=torch.long),
        "num_nodes": num_nodes,
    }
    if device is not None:
        result = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in result.items()}
    return result


def collate_graphs(batch):
    """Collate a batch of (solute_graph, solvent_graph, temp, target) tuples."""
    solute_gs, solvent_gs, temp_feats, targets = zip(*batch)
    return (
        batch_graph_list(solute_gs),
        batch_graph_list(solvent_gs),
        torch.stack(temp_feats),
        torch.stack(targets),
    )


# ============================================================================
# GNN Method Wrappers
# ============================================================================

class _GNNBaseMethod(BaseMethod):
    """Base class for GNN methods — not directly instantiated."""
    GNN_TYPE = "GCN"
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    EPOCHS = 100
    LR = 0.001
    BATCH_SIZE = 64
    PATIENCE = 15

    @classmethod
    def info(cls):
        return {
            "name": f"GNN-{cls.GNN_TYPE}",
            "featurizer": "none",  # GNNs featurize from SMILES directly
            "mode": "graph",
            "gpu_required": False,  # can run on CPU, just slower
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        NOTE: For GNN models, X_train is ignored. Instead, we read SMILES
        from the DataFrame. The pipeline must pass the DataFrames directly.
        This is handled by the GNN-specific runner script (scripts/run_gnn_baselines.py).
        """
        raise NotImplementedError("GNN models require the graph-based runner. Use scripts/run_gnn_baselines.py")

    def predict(self, X):
        raise NotImplementedError("GNN models require the graph-based runner.")


class GCNMethod(_GNNBaseMethod):
    GNN_TYPE = "GCN"

class GATMethod(_GNNBaseMethod):
    GNN_TYPE = "GAT"

class GINMethod(_GNNBaseMethod):
    GNN_TYPE = "GIN"

class SolubNetMethod(_GNNBaseMethod):
    """SolubNet: TAGConv GNN adapted from Du et al. (2017) / SolubNet.

    Uses Topology Adaptive Graph Convolution with K=2 hops per layer.
    Each layer concatenates [h^0, h^1, h^2] (original + 1-hop + 2-hop
    propagated features) and projects through a linear layer.
    """
    GNN_TYPE = "TAGConv"
    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    EPOCHS = 200
    LR = 0.001
    BATCH_SIZE = 64
    PATIENCE = 25

    @classmethod
    def info(cls):
        return {
            "name": "SolubNet",
            "featurizer": "none",
            "mode": "graph",
            "gpu_required": False,
        }
