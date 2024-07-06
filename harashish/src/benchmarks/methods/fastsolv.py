"""
FastSolv: MLP with Sobolev training for temperature-aware solubility prediction.

Adapts the approach from Attia et al. (2025) with corrected gradient propagation.
The original reference uses retain_graph=True without create_graph=True in
torch.autograd.grad, which prevents the Sobolev loss from backpropagating to
model parameters. This implementation uses create_graph=True for correct
second-order gradient flow.

The Sobolev loss enforces consistency between the model's analytical dlogS/dT
(computed via autograd through all temperature features) and the empirical
finite-difference gradient. This encodes the thermodynamic prior that
dissolution is typically endothermic (dlogS/dT > 0).

Architecture:
  - Same RDKit descriptor featurization as FastProp
  - MLP with BatchNorm + ReLU + Dropout
  - Single standardized temperature input, reconstructed into 4 derived
    features (T/300, 1000/T, (T/300)^2, ln(T/300)) inside the forward pass
    so autograd captures the total temperature derivative
  - Loss = MSE(logS) + 10.0 * nanmean((dy/dT_model - dy/dT_empirical)^2)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FastSolvNet(nn.Module):
    """Deep descriptor-based network with BatchNorm, matching FastProp architecture."""

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


class FastSolvModel(BaseMethod):
    """
    FastSolv: MLP with Sobolev (gradient-penalised) training.

    The Sobolev loss matches the model's analytical d(logS)/dT against
    empirical finite-difference gradients grouped by (solute, solvent) pairs,
    filtered by the endothermic dissolution prior (positive gradients only).
    """

    SOBOLEV_SCALE = 10.0
    HIDDEN_DIMS = (512, 256, 128)
    DROPOUT = 0.1
    MAX_EPOCHS = 300
    PATIENCE = 20
    BATCH_SIZE = 256
    LR = 1e-3
    GRAD_MAG_CAP = 1.0

    @classmethod
    def info(cls):
        return {
            "name": "FastSolv",
            "featurizer": "rdkit",
            "mode": "dual",
            "gpu_required": False,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch.optim as optim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Separate descriptors (all cols except last 4) and temperature ---
        desc_train = X_train[:, :-4]
        t_raw_train = X_train[:, -4].copy()  # T/300

        # --- Standardize descriptors ---
        self._desc_mean = desc_train.mean(axis=0)
        self._desc_std = desc_train.std(axis=0) + 1e-8
        desc_norm = ((desc_train - self._desc_mean) / self._desc_std).astype(np.float32)

        # --- Standardize temperature ---
        self._t_mean = float(t_raw_train.mean())
        self._t_std = float(t_raw_train.std()) + 1e-8
        t_std = ((t_raw_train - self._t_mean) / self._t_std).astype(np.float32)

        # --- Standardize targets ---
        self._y_mean = float(y_train.mean())
        self._y_std = float(y_train.std()) + 1e-8
        y_std = ((y_train - self._y_mean) / self._y_std).astype(np.float32)

        # --- Temperature feature stats for in-network reconstruction ---
        tf_raw = np.column_stack([
            t_raw_train,
            (10.0 / 3.0) / np.clip(t_raw_train, 1e-6, None),
            t_raw_train ** 2,
            np.log(np.clip(t_raw_train, 1e-6, None)),
        ])
        self._tf_mean = tf_raw.mean(axis=0).astype(np.float32)
        self._tf_std = (tf_raw.std(axis=0) + 1e-8).astype(np.float32)

        # --- Sobolev gradient targets ---
        grad_targets = self._compute_sobolev_targets(desc_train, t_std, y_std)
        n_valid = int(np.isfinite(grad_targets).sum())
        pct = 100.0 * n_valid / len(grad_targets)
        print(f"  Sobolev: {n_valid}/{len(grad_targets)} valid gradient targets ({pct:.1f}%)")

        # --- Tensors ---
        desc_t = torch.tensor(desc_norm, dtype=torch.float32, device=device)
        t_t = torch.tensor(t_std, dtype=torch.float32, device=device).unsqueeze(1)
        y_t = torch.tensor(y_std, dtype=torch.float32, device=device)
        g_t = torch.tensor(grad_targets, dtype=torch.float32, device=device)

        # Cache stat tensors on device for the forward helper
        tm = torch.tensor([self._t_mean], dtype=torch.float32, device=device)
        ts = torch.tensor([self._t_std], dtype=torch.float32, device=device)
        tfm = torch.tensor(self._tf_mean, dtype=torch.float32, device=device)
        tfs = torch.tensor(self._tf_std, dtype=torch.float32, device=device)

        def _make_features(desc, t_standardised):
            """Reconstruct 4 temp features from standardised temperature for autograd."""
            t_raw = t_standardised * ts + tm
            t_inv = (10.0 / 3.0) / t_raw.clamp(min=1e-6)
            t_sq = t_raw ** 2
            t_log = torch.log(t_raw.clamp(min=1e-6))
            tf = torch.cat([t_raw, t_inv, t_sq, t_log], dim=-1)
            tf_norm = (tf - tfm) / tfs
            return torch.cat([desc, tf_norm], dim=-1)

        # --- Build model ---
        in_dim = desc_norm.shape[1] + 4
        self.model = FastSolvNet(in_dim, self.HIDDEN_DIMS, self.DROPOUT).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.LR, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # --- Validation tensors ---
        if X_val is not None:
            dv = ((X_val[:, :-4] - self._desc_mean) / self._desc_std).astype(np.float32)
            tv = ((X_val[:, -4] - self._t_mean) / self._t_std).astype(np.float32)
            yv = ((y_val - self._y_mean) / self._y_std).astype(np.float32)
            desc_v = torch.tensor(dv, dtype=torch.float32, device=device)
            t_v = torch.tensor(tv, dtype=torch.float32, device=device).unsqueeze(1)
            y_v = torch.tensor(yv, dtype=torch.float32, device=device)

        # --- Training loop ---
        dataset = torch.utils.data.TensorDataset(desc_t, t_t, y_t, g_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True,
        )

        best_val = float("inf")
        best_state = None
        wait = 0
        import sys

        for epoch in range(self.MAX_EPOCHS):
            self.model.train()
            epoch_mse = 0.0
            epoch_sob = 0.0
            n_batches = 0
            for db, tb, yb, gb in loader:
                optimizer.zero_grad()

                tb = tb.detach().requires_grad_(True)
                x = _make_features(db, tb)
                y_hat = self.model(x)

                mse = F.mse_loss(y_hat, yb)

                # Sobolev: d(y_hat)/d(t_std) via autograd with create_graph
                dy_dt = torch.autograd.grad(
                    y_hat, tb,
                    grad_outputs=torch.ones_like(y_hat),
                    create_graph=True,
                    retain_graph=True,
                )[0].squeeze(-1)

                valid_mask = ~torch.isnan(gb)
                if valid_mask.any():
                    diff_sq = (dy_dt[valid_mask] - gb[valid_mask]) ** 2
                    sobolev = self.SOBOLEV_SCALE * diff_sq.mean()
                else:
                    sobolev = torch.zeros(1, device=yb.device, requires_grad=False)[0]

                (mse + sobolev).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_mse += mse.item()
                epoch_sob += sobolev.item()
                n_batches += 1

            avg_mse = epoch_mse / max(n_batches, 1)
            avg_sob = epoch_sob / max(n_batches, 1)

            # --- Validation & early stopping ---
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(_make_features(desc_v, t_v))
                    val_loss = F.mse_loss(val_pred, y_v).item()
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    wait = 0
                else:
                    wait += 1

                if epoch % 10 == 0 or wait >= self.PATIENCE:
                    lr_now = optimizer.param_groups[0]["lr"]
                    print(
                        f"    ep {epoch:3d}  mse={avg_mse:.4f}  sob={avg_sob:.4f}  "
                        f"val={val_loss:.4f}  best={best_val:.4f}  "
                        f"wait={wait}  lr={lr_now:.1e}",
                        flush=True,
                    )

                if wait >= self.PATIENCE:
                    break
            else:
                if epoch % 10 == 0:
                    print(
                        f"    ep {epoch:3d}  mse={avg_mse:.4f}  sob={avg_sob:.4f}",
                        flush=True,
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._device = device

    def predict(self, X) -> np.ndarray:
        desc = X[:, :-4]
        t_raw = X[:, -4].copy()

        desc_norm = ((desc - self._desc_mean) / self._desc_std).astype(np.float32)
        t_std_arr = ((t_raw - self._t_mean) / self._t_std).astype(np.float32)

        d_t = torch.tensor(desc_norm, dtype=torch.float32, device=self._device)
        t_t = torch.tensor(t_std_arr, dtype=torch.float32, device=self._device).unsqueeze(1)

        tm = torch.tensor([self._t_mean], dtype=torch.float32, device=self._device)
        ts = torch.tensor([self._t_std], dtype=torch.float32, device=self._device)
        tfm = torch.tensor(self._tf_mean, dtype=torch.float32, device=self._device)
        tfs = torch.tensor(self._tf_std, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            t_raw_t = t_t * ts + tm
            t_inv = (10.0 / 3.0) / t_raw_t.clamp(min=1e-6)
            t_sq = t_raw_t ** 2
            t_log = torch.log(t_raw_t.clamp(min=1e-6))
            tf = torch.cat([t_raw_t, t_inv, t_sq, t_log], dim=-1)
            tf_norm = (tf - tfm) / tfs
            x = torch.cat([d_t, tf_norm], dim=-1)
            y_std_pred = self.model(x).cpu().numpy()

        return y_std_pred * self._y_std + self._y_mean

    # ------------------------------------------------------------------
    # Sobolev gradient target computation
    # ------------------------------------------------------------------

    def _compute_sobolev_targets(self, desc_raw, t_std, y_std):
        """
        Compute empirical d(y_std)/d(t_std) via finite differences.

        Groups training data by identical descriptor vectors (same solute+solvent
        pair). Within each group, sorts by standardized temperature and computes
        np.gradient finite differences.

        Physics filter (endothermic dissolution prior):
          - Only keeps positive gradients (solubility increases with T)
          - Caps at GRAD_MAG_CAP in standardized space
          - NaN for all others; Sobolev loss uses masked mean to ignore them
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
                if np.isfinite(g) and 0.0 < g < self.GRAD_MAG_CAP:
                    grads[orig_idx] = g

        return grads
