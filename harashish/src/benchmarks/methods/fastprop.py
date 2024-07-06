"""
FastProp-inspired deep learning on molecular descriptors.

Based on: Burns & Green (2025) "FastProp: Fast molecular property prediction
with descriptors-based deep learning"

Uses RDKit descriptors + temperature features -> deep MLP with PyTorch.
This is the GPU-accelerated analog of the sklearn MLP baseline.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseMethod


class FastPropNet(nn.Module):
    """Deep descriptor-based network for solubility prediction."""

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


class FastPropModel(BaseMethod):
    """
    FastProp-style deep learning on descriptors.

    Uses PyTorch MLP with batch normalization, trained on RDKit features.
    Fits within the standard pipeline (receives feature matrices).
    """

    @classmethod
    def info(cls):
        return {"name": "FastProp", "featurizer": "rdkit", "mode": "dual", "gpu_required": False}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch.optim as optim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize inputs
        self._mean = X_train.mean(axis=0)
        self._std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - self._mean) / self._std

        X_t = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

        in_dim = X_train.shape[1]
        self.model = FastPropNet(in_dim, hidden_dims=(512, 256, 128), dropout=0.1).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        if X_val is not None:
            X_val_norm = (X_val - self._mean) / self._std
            X_v = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
            y_v = torch.tensor(y_val, dtype=torch.float32).to(device)

        best_val_loss = float("inf")
        best_state = None
        wait = 0
        patience = 20
        batch_size = 256

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(300):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = nn.functional.mse_loss(self.model(X_v), y_v).item()
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self._device = device

    def predict(self, X) -> np.ndarray:
        X_norm = (X - self._mean) / self._std
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            return self.model(X_t).cpu().numpy()
