"""
RILOOD — Relational Invariant Learning for OOD Solubility Prediction.

Implements the RILOOD framework (Chen et al., ICML 2025) for multi-solvent
solubility prediction. The architecture comprises:

  1. Dual MPNN encoders with CIGIN-style atom/bond featurization
  2. Bidirectional atomic interaction maps (solute <-> solvent)
  3. Set2Set graph-level readout over interaction-enriched node embeddings
  4. MCVAE: Mixup-enhanced Conditional VAE for environment-invariant latent codes
  5. MCAR: Multi-granularity Context-Aware Refinement (global-local Hadamard fusion)
  6. MI contrastive loss for invariant representation learning

Extended for SC3 benchmark:
  - Temperature features [T/300, 1000/T, (T/300)^2, ln(T/300)] concatenated
    into the predictor head alongside the MCAR-refined latent code
  - Solvent identity encoded via learned embeddings instead of one-hot

Requires: torch, torch_geometric, rdkit
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, Set2Set
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from typing import List, Optional, Dict, Tuple

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# ============================================================================
# CIGIN-style atom and bond featurization (Tables 1 & 2 from CIGIN paper)
# ============================================================================

ATOM_SYMBOLS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "unknown"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D"]
CHIRALITIES = ["R", "S", "None"]
NUM_HS = ["0", "1", "2", "3", "4", "unknown"]

ATOM_FEAT_DIM = 11 + 4 + 3 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # 31

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_STEREOS = ["E", "Z", "None"]
BOND_FEAT_DIM = 4 + 1 + 1 + 3  # 9

NUM_TEMP_FEATS = 4


def _one_hot(value, choices):
    enc = [0] * len(choices)
    if value in choices:
        enc[choices.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


def get_atom_features(atom) -> List[float]:
    """CIGIN Table 1 compliant 31-dim atom feature vector."""
    feats = []

    feats += _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)

    feats.append(1.0 if (atom.GetTotalValence() - atom.GetExplicitValence()) > 0 else 0.0)
    feats.append(1.0 if atom.GetNumRadicalElectrons() > 0 else 0.0)

    try:
        chi = str(atom.GetChiralTag())
        if "CW" in chi:
            feats += _one_hot("R", CHIRALITIES)
        elif "CCW" in chi:
            feats += _one_hot("S", CHIRALITIES)
        else:
            feats += _one_hot("None", CHIRALITIES)
    except Exception:
        feats += _one_hot("None", CHIRALITIES)

    feats += _one_hot(str(min(atom.GetTotalNumHs(), 4)), NUM_HS)

    hyb = str(atom.GetHybridization())
    hyb_val = hyb.split(".")[-1] if "." in hyb else hyb
    if hyb_val in ("S", "SP"):
        hyb_val = "SP"
    feats += _one_hot(hyb_val, HYBRIDIZATIONS)

    is_acidic = (
        atom.GetSymbol() == "O"
        and atom.GetTotalNumHs() > 0
        and any(n.GetSymbol() in ("C", "S", "P") for n in atom.GetNeighbors())
    )
    feats.append(1.0 if is_acidic else 0.0)

    is_basic = (
        atom.GetSymbol() == "N"
        and not atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
    )
    feats.append(1.0 if is_basic else 0.0)

    feats.append(1.0 if atom.GetIsAromatic() else 0.0)

    is_donor = atom.GetSymbol() in ("N", "O") and atom.GetTotalNumHs() > 0
    feats.append(1.0 if is_donor else 0.0)

    is_acceptor = atom.GetSymbol() in ("N", "O", "F")
    feats.append(1.0 if is_acceptor else 0.0)

    return feats


def get_bond_features(bond) -> List[float]:
    """CIGIN Table 2 compliant 9-dim bond feature vector."""
    feats = []
    feats += _one_hot(str(bond.GetBondType()).split(".")[-1], BOND_TYPES)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    feats.append(1.0 if bond.IsInRing() else 0.0)

    stereo = str(bond.GetStereo())
    if "Z" in stereo or "CIS" in stereo:
        feats += _one_hot("Z", BOND_STEREOS)
    elif "E" in stereo or "TRANS" in stereo:
        feats += _one_hot("E", BOND_STEREOS)
    else:
        feats += _one_hot("None", BOND_STEREOS)
    return feats


def smiles_to_pyg(smiles: str) -> Optional[Data]:
    """Convert SMILES to a PyG Data object with CIGIN features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = [get_atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edge_index_list, edge_attr_list = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list.extend([[i, j], [j, i]])
        bf = get_bond_features(bond)
        edge_attr_list.extend([bf, bf])

    if len(edge_index_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================================================================
# MPNN Layer (message passing with edge features)
# ============================================================================

class MPNNLayer(MessagePassing):
    """MPNN layer: message = MLP([h_i || h_j || e_ij]), update = MLP([h_i || agg])."""

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__(aggr="add", node_dim=0)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


# ============================================================================
# GNN Encoder with Set2Set readout
# ============================================================================

class GNNEncoder(nn.Module):
    """Multi-layer MPNN encoder. Returns node-level representations.

    Set2Set readout and final projection are exposed as separate methods
    so that we can call them after the interaction map enriches the node
    embeddings (see RILOOD.forward).
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

        # Set2Set operates on 2*hidden_dim input (after interaction concatenation)
        self.readout = Set2Set(hidden_dim * 2, processing_steps=3)
        # Set2Set output is 2 * input_dim = 4 * hidden_dim; project to hidden_dim
        self.final_proj = nn.Linear(2 * (hidden_dim * 2), hidden_dim)

    def forward_nodes(self, data: Data) -> torch.Tensor:
        """Return node-level representations [total_nodes, hidden_dim]."""
        x = self.node_embed(data.x)
        has_edges = data.edge_index.size(1) > 0
        edge_attr = self.edge_embed(data.edge_attr) if has_edges else None

        for layer, norm in zip(self.layers, self.layer_norms):
            if has_edges:
                x_new = layer(x, data.edge_index, edge_attr)
            else:
                x_new = x
            x = norm(x + x_new)
            x = self.dropout(x)
        return x


# ============================================================================
# Bidirectional Interaction Map (CIGIN Sec 3.1)
# ============================================================================

class InteractionMap(nn.Module):
    """Bidirectional cross-molecule attention via outer product.

    For each (solute, solvent) pair in the batch:
      I = h1 @ h2^T / sqrt(d)                     [N_solute, N_solvent]
      h1_new = [h1 ; LayerNorm(I @ h2)]           [N_solute, 2*D]
      h2_new = [h2 ; LayerNorm(I^T @ h1)]         [N_solvent, 2*D]

    Uses padded dense batched BMM for GPU efficiency instead of per-graph loops.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h1_nodes: torch.Tensor,
        h2_nodes: torch.Tensor,
        batch1: torch.Tensor,
        batch2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch1.numel() == 0 and batch2.numel() == 0:
            return torch.zeros(0, self.hidden_dim * 2, device=h1_nodes.device), \
                   torch.zeros(0, self.hidden_dim * 2, device=h2_nodes.device)

        batch_size = max(
            batch1.max().item() + 1 if batch1.numel() > 0 else 0,
            batch2.max().item() + 1 if batch2.numel() > 0 else 0,
        )
        device = h1_nodes.device
        D = self.hidden_dim
        scale = D ** -0.5

        # Compute per-graph sizes
        counts1 = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts2 = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts1.scatter_add_(0, batch1, torch.ones_like(batch1))
        counts2.scatter_add_(0, batch2, torch.ones_like(batch2))
        max_n1 = int(counts1.max().item())
        max_n2 = int(counts2.max().item())

        # Compute within-graph indices for each node (vectorized)
        ones1 = torch.ones_like(batch1)
        ones2 = torch.ones_like(batch2)
        cumcounts1 = torch.zeros_like(batch1)
        cumcounts2 = torch.zeros_like(batch2)
        # For each node, its position within its graph = cumsum within that graph
        # Use a trick: sort by batch, compute per-batch cumsum
        _, perm1 = torch.sort(batch1, stable=True)
        _, perm2 = torch.sort(batch2, stable=True)
        # Since batch tensors from PyG are already sorted, perm = arange
        idx_in_graph1 = torch.zeros_like(batch1)
        idx_in_graph2 = torch.zeros_like(batch2)
        if batch1.numel() > 0:
            # Within each group, position = cumcount of seen items in that group
            offsets1 = torch.cat([torch.tensor([0], device=device), counts1.cumsum(0)[:-1]])
            global_pos1 = torch.arange(batch1.size(0), device=device)
            idx_in_graph1 = global_pos1 - offsets1[batch1]
        if batch2.numel() > 0:
            offsets2 = torch.cat([torch.tensor([0], device=device), counts2.cumsum(0)[:-1]])
            global_pos2 = torch.arange(batch2.size(0), device=device)
            idx_in_graph2 = global_pos2 - offsets2[batch2]

        # Pad into dense [B, max_N, D] tensors
        H1 = torch.zeros(batch_size, max_n1, D, device=device)
        H2 = torch.zeros(batch_size, max_n2, D, device=device)
        H1[batch1, idx_in_graph1] = h1_nodes
        H2[batch2, idx_in_graph2] = h2_nodes

        mask1 = torch.zeros(batch_size, max_n1, dtype=torch.bool, device=device)
        mask2 = torch.zeros(batch_size, max_n2, dtype=torch.bool, device=device)
        mask1[batch1, idx_in_graph1] = True
        mask2[batch2, idx_in_graph2] = True

        # Batched interaction: I = H1 @ H2^T / sqrt(D)  [B, max_n1, max_n2]
        I = torch.bmm(H1, H2.transpose(1, 2)) * scale
        I = torch.clamp(I, -10, 10)

        # Zero out padded positions
        pad_mask = mask1.unsqueeze(2) & mask2.unsqueeze(1)  # [B, max_n1, max_n2]
        I = I * pad_mask.float()

        # h1_interaction = I @ H2  [B, max_n1, D]
        H1_inter = torch.bmm(I, H2)
        # h2_interaction = I^T @ H1  [B, max_n2, D]
        H2_inter = torch.bmm(I.transpose(1, 2), H1)

        # Apply LayerNorm (only on valid positions)
        H1_inter_flat = H1_inter.reshape(-1, D)
        H2_inter_flat = H2_inter.reshape(-1, D)
        H1_inter_flat = self.layer_norm(H1_inter_flat)
        H2_inter_flat = self.layer_norm(H2_inter_flat)
        H1_inter = H1_inter_flat.reshape(batch_size, max_n1, D)
        H2_inter = H2_inter_flat.reshape(batch_size, max_n2, D)

        # Gather back to sparse format matching original node ordering
        h1_out = torch.cat([
            h1_nodes,
            H1_inter[batch1, idx_in_graph1]
        ], dim=-1)
        h2_out = torch.cat([
            h2_nodes,
            H2_inter[batch2, idx_in_graph2]
        ], dim=-1)

        return h1_out, h2_out


# ============================================================================
# MCAR: Multi-granularity Context-Aware Refinement
# ============================================================================

class MCAR(nn.Module):
    """MCAR module from RILOOD (Eq 7-10).

    Fuses the VAE latent z with the solvent graph embedding h_solvent:
      E = [z ; h_solvent]
      O_c = W_c(ReLU(E))          (global interaction)
      O_f = PReLU(W_l(E))         (local interaction)
      H_c = O_c * O_f             (Hadamard fusion)
      output = LayerNorm(proj(H_c) + z)    (residual)
    """

    def __init__(self, z_dim: int, h_dim: int):
        super().__init__()
        input_dim = z_dim + h_dim
        self.global_proj = nn.Sequential(
            nn.Linear(input_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim)
        )
        self.local_proj = nn.Sequential(nn.Linear(input_dim, z_dim), nn.PReLU())
        self.output_proj = nn.Linear(z_dim, z_dim)
        self.layer_norm = nn.LayerNorm(z_dim)

    def forward(self, z: torch.Tensor, h_solvent: torch.Tensor) -> torch.Tensor:
        E = torch.cat([z, h_solvent], dim=-1)
        Oc = self.global_proj(E)
        Of = self.local_proj(E)
        Hc = Oc * Of
        return self.layer_norm(self.output_proj(Hc) + z)


# ============================================================================
# MCVAE: Mixup-enhanced Conditional VAE
# ============================================================================

class MCVAE(nn.Module):
    """Conditional VAE with environment conditioning (Eq 5-6).

    Encoder: p(z | h, e) -> mu, log_var
    Decoder: p(h | z, e) -> h_hat (for reconstruction loss)
    Separate log_sigma head for reconstruction uncertainty.
    """

    def __init__(self, hidden_dim: int, env_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + env_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.log_sigma_rec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.env_dim = env_dim

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, h, e):
        encoded = self.encoder(torch.cat([h, e], dim=-1))
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        mu = torch.clamp(mu, -10, 10)
        log_var = torch.clamp(log_var, -10, 10)

        z = self.reparameterize(mu, log_var)
        h_hat = self.decoder(torch.cat([z, e], dim=-1))
        log_sigma_rec = torch.clamp(self.log_sigma_rec(h), -10, 10)

        return z, mu, log_var, h_hat, log_sigma_rec


# ============================================================================
# RILOOD Model (full architecture with temperature)
# ============================================================================

class RILOODModel(nn.Module):
    """RILOOD: dual MPNN + interaction map + MCVAE + MCAR + MI loss.

    Extended for SC3: temperature features are concatenated into the
    predictor alongside the MCAR-refined latent representation.
    Environment is modelled via a learned embedding (not one-hot) so
    we don't need to know the number of solvents at model init time.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 168,
        num_layers: int = 3,
        num_solvents: int = 200,
        mixup_alpha: float = 0.5,
        dropout: float = 0.3,
        num_temp_feats: int = NUM_TEMP_FEATS,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mixup_alpha = mixup_alpha

        self.solute_encoder = GNNEncoder(
            ATOM_FEAT_DIM, BOND_FEAT_DIM, hidden_dim, num_layers, dropout
        )
        self.solvent_encoder = GNNEncoder(
            ATOM_FEAT_DIM, BOND_FEAT_DIM, hidden_dim, num_layers, dropout
        )

        self.interaction_map = InteractionMap(hidden_dim)

        # Learned solvent embedding for MCVAE conditioning
        env_dim = 64
        self.solvent_embedding = nn.Embedding(num_solvents, env_dim)
        self.env_dim = env_dim

        self.mcvae = MCVAE(hidden_dim, env_dim, latent_dim)
        self.mcar = MCAR(z_dim=latent_dim, h_dim=hidden_dim)

        # Predictor: MCAR output (latent_dim) + temperature features (4)
        predictor_in = latent_dim + num_temp_feats
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, 1),
        )

        # Aleatoric regression uncertainty (Eq 4)
        self.log_sigma_reg = nn.Sequential(
            nn.Linear(predictor_in, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )

        # Projection for MI contrastive loss
        self.h1_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        solute_data: Batch,
        solvent_data: Batch,
        env_idx: torch.Tensor,
        temp_feats: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Node-level encoding
        h1_nodes = self.solute_encoder.forward_nodes(solute_data)
        h2_nodes = self.solvent_encoder.forward_nodes(solvent_data)

        # 2. Bidirectional interaction map
        h1_inter, h2_inter = self.interaction_map(
            h1_nodes, h2_nodes, solute_data.batch, solvent_data.batch
        )

        # 3. Set2Set readout on interaction-enriched node embeddings
        h1_graph = self.solute_encoder.readout(h1_inter, solute_data.batch)
        h1_graph = self.solute_encoder.final_proj(h1_graph)

        h2_graph = self.solvent_encoder.readout(h2_inter, solvent_data.batch)
        h2_graph = self.solvent_encoder.final_proj(h2_graph)

        batch_size = h1_graph.size(0)
        device = h1_graph.device

        # 4. Environment embedding
        e = self.solvent_embedding(env_idx.clamp(0, self.solvent_embedding.num_embeddings - 1))

        # 5. Mixup: blend solute with solvent representations (Eq 3)
        h1_mix = h1_graph
        if training and self.mixup_alpha > 0:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            h1_mix = lam * h1_graph + (1 - lam) * h2_graph

        # 6. MCVAE
        z, mu, log_var, h_hat, log_sigma_rec = self.mcvae(h1_mix, e)

        # 7. MCAR refinement
        hc = self.mcar(z, h2_graph)

        # 8. Concatenate temperature features and predict
        hc_temp = torch.cat([hc, temp_feats], dim=-1)
        pred = self.predictor(hc_temp).squeeze(-1)
        log_sigma_reg = torch.clamp(self.log_sigma_reg(hc_temp), -10, 10)

        # 9. Compute losses during training
        losses = {}
        if training and targets is not None:
            targets_loss = targets

            # Eq 4: heteroscedastic regression loss
            sigma_sq_reg = torch.exp(log_sigma_reg).squeeze() + 1e-6
            losses["reg"] = (
                (targets_loss - pred).pow(2) / sigma_sq_reg + log_sigma_reg.squeeze()
            ).mean()

            # Eq 6: MCVAE loss = KLD + reconstruction
            kld = -0.5 * torch.sum(
                1 + log_var - mu.pow(2) - log_var.exp(), dim=1
            ).mean()
            sigma_sq_rec = torch.exp(log_sigma_rec).squeeze() + 1e-6
            rec = (
                (h_hat - h1_mix).pow(2).sum(dim=1) / sigma_sq_rec
                + log_sigma_rec.squeeze()
            ).mean()
            losses["vae"] = kld + rec

            # Eq 11-12: MI contrastive loss
            hc_n = F.normalize(hc, dim=1)
            h1_proj = self.h1_proj(h1_graph)
            h1_n = F.normalize(h1_proj, dim=1)
            logits = torch.matmul(hc_n, h1_n.T) / 0.5
            losses["mi"] = F.cross_entropy(
                logits, torch.arange(batch_size, device=device)
            )

        return pred, losses


# ============================================================================
# Dataset and Collation for SC3 benchmark
# ============================================================================

class RILOODDataset(Dataset):
    """Dataset that returns PyG graph pairs + temperature features + target."""

    def __init__(self, df, graph_cache: Dict[str, Data], solvent_map: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.graph_cache = graph_cache
        self.solvent_map = solvent_map
        self.valid_indices = []

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            if row["Solute"] in self.graph_cache and row["Solvent"] in self.graph_cache:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]
        T = row["Temperature"]
        return {
            "solute": self.graph_cache[row["Solute"]],
            "solvent": self.graph_cache[row["Solvent"]],
            "env": self.solvent_map.get(row["Solvent"], 0),
            "temp_feats": torch.tensor(
                [T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)],
                dtype=torch.float,
            ),
            "target": row["LogS"],
        }


def rilood_collate_fn(batch):
    """Custom collation: batches PyG graphs and stacks tensors."""
    return {
        "solute": Batch.from_data_list([b["solute"] for b in batch]),
        "solvent": Batch.from_data_list([b["solvent"] for b in batch]),
        "env": torch.tensor([b["env"] for b in batch], dtype=torch.long),
        "temp_feats": torch.stack([b["temp_feats"] for b in batch]),
        "target": torch.tensor([b["target"] for b in batch], dtype=torch.float),
    }


# ============================================================================
# Graph cache builder
# ============================================================================

def build_graph_cache(all_smiles, verbose=True) -> Dict[str, Data]:
    """Pre-compute PyG graphs for all SMILES, returning a cache dict."""
    cache = {}
    failed = 0
    for smi in all_smiles:
        g = smiles_to_pyg(smi)
        if g is not None:
            cache[smi] = g
        else:
            failed += 1
    if verbose and failed:
        print(f"  Warning: {failed}/{len(all_smiles)} SMILES failed graph construction")
    return cache


def build_solvent_map(splits: dict) -> Dict[str, int]:
    """Assign integer IDs to unique solvent SMILES across all splits."""
    all_solvents = set()
    for df in splits.values():
        all_solvents.update(df["Solvent"].unique())
    return {s: i for i, s in enumerate(sorted(all_solvents))}
