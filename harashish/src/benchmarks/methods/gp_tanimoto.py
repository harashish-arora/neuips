"""
Gaussian Process with Tanimoto kernel on Morgan fingerprints.

Uses gpytorch with the Tanimoto kernel from GAUCHE (Klarner et al., NeurIPS 2023).
Provides built-in uncertainty quantification via the GP posterior variance.

Since exact GP inference is O(n³) and infeasible for ~61K training rows, we use
a subset-of-data approximation: stratified sample of ~3000 points from training.

The Tanimoto (Jaccard) kernel on binary fingerprints:
    k(x, y) = <x,y> / (||x||² + ||y||² - <x,y>)

References:
    Klarner et al. "GAUCHE: A Library for Gaussian Processes in Chemistry", NeurIPS 2023.
    Ralaivola et al. "Graph kernels for chemical informatics", Neural Networks, 2005.
"""

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, ProductKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from .base import BaseMethod

_SUBSET_SIZE = 3000


def batch_tanimoto_sim(x1, x2, eps=1e-6):
    """Tanimoto similarity between batched tensors (from GAUCHE)."""
    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("Tensors must have a batch dimension")
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_norm = torch.sum(x1 ** 2, dim=-1, keepdims=True)
    x2_norm = torch.sum(x2 ** 2, dim=-1, keepdims=True)
    tan_similarity = (dot_prod + eps) / (
        eps + x1_norm + torch.transpose(x2_norm, -1, -2) - dot_prod
    )
    return tan_similarity.clamp_min_(0)


class TanimotoKernel(Kernel):
    """Tanimoto kernel for bit/count vectors (from GAUCHE, Klarner et al. 2023)."""
    is_stationary = False
    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        return batch_tanimoto_sim(x1, x2)


class _ExactGPModel(ExactGP):
    """Exact GP with Tanimoto kernel on full [fingerprint || temp] input."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPTanimotoModel(BaseMethod):
    """GP with Tanimoto kernel on Morgan fingerprints (gpytorch-based).

    Uses subset-of-data approximation for scalability.
    """

    @classmethod
    def info(cls):
        return {
            "name": "GP_Tanimoto",
            "featurizer": "morgan",
            "mode": "dual",
            "gpu_required": False,
        }

    def _stratified_subset(self, X, y, n_subset, rng):
        n = len(y)
        if n <= n_subset:
            return X, y
        n_bins = min(50, n_subset // 20)
        bins = np.linspace(y.min() - 0.01, y.max() + 0.01, n_bins + 1)
        bin_idx = np.clip(np.digitize(y, bins) - 1, 0, n_bins - 1)
        per_bin = max(1, n_subset // n_bins)
        selected = []
        for b in range(n_bins):
            mask = np.where(bin_idx == b)[0]
            if len(mask) == 0:
                continue
            take = min(len(mask), per_bin)
            selected.extend(rng.choice(mask, size=take, replace=False).tolist())
        remaining = n_subset - len(selected)
        if remaining > 0:
            pool = list(set(range(n)) - set(selected))
            if pool:
                extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                selected.extend(extra.tolist())
        return X[np.array(selected[:n_subset])], y[np.array(selected[:n_subset])]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        rng = np.random.RandomState(self.seed)
        torch.manual_seed(self.seed)

        self._n_fp = X_train.shape[1] - 4

        X_sub, y_sub = self._stratified_subset(X_train, y_train, _SUBSET_SIZE, rng)

        self._y_mean = float(np.mean(y_sub))
        self._y_std = max(float(np.std(y_sub)), 1e-6)
        y_norm = (y_sub - self._y_mean) / self._y_std

        train_x = torch.tensor(X_sub, dtype=torch.float64)
        train_y = torch.tensor(y_norm, dtype=torch.float64)

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._likelihood.double()
        self._model = _ExactGPModel(train_x, train_y, self._likelihood)
        self._model.double()

        self._model.train()
        self._likelihood.train()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)

        for i in range(50):
            optimizer.zero_grad()
            output = self._model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        self._model.eval()
        self._likelihood.eval()

    def predict(self, X):
        X_t = torch.tensor(X, dtype=torch.float64)
        chunk_size = 2000
        preds = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X_t), chunk_size):
                chunk = X_t[i:i + chunk_size]
                pred = self._likelihood(self._model(chunk))
                preds.append(pred.mean.numpy())
        y_pred_norm = np.concatenate(preds)
        return y_pred_norm * self._y_std + self._y_mean

    def predict_with_uncertainty(self, X):
        """Return (mean, std) predictions."""
        X_t = torch.tensor(X, dtype=torch.float64)
        chunk_size = 2000
        means, stds = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X_t), chunk_size):
                chunk = X_t[i:i + chunk_size]
                pred = self._likelihood(self._model(chunk))
                means.append(pred.mean.numpy())
                stds.append(pred.stddev.numpy())
        y_mean = np.concatenate(means) * self._y_std + self._y_mean
        y_std = np.concatenate(stds) * self._y_std
        return y_mean, y_std
