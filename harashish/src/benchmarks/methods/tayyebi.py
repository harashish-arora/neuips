"""
Tayyebi et al. baseline: Mordred descriptors + Random Forest.

Reference: Tayyebi et al., adapted from aqueous-only to multi-solvent.
Uses Mordred 2D descriptors with variance/correlation filtering + RF regressor.

The variance threshold (0.1) and correlation cutoff (0.8) are applied at
train time.  The same column mask is stored and reused at inference.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold

from .base import BaseMethod


class TayyebiMordredModel(BaseMethod):
    """Mordred descriptors + RandomForest (Tayyebi-style)."""

    @classmethod
    def info(cls):
        return {
            "name": "Tayyebi_Mordred",
            "featurizer": "mordred",
            "mode": "dual",
            "gpu_required": False,
        }

    def _learn_filter(self, X):
        """Learn variance + correlation filter from training features."""
        self._vt = VarianceThreshold(threshold=0.1)
        self._vt.fit(X)
        X_var = self._vt.transform(X)

        corr = np.corrcoef(X_var, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        upper = np.triu(np.abs(corr), k=1)
        to_drop = set()
        for i in range(upper.shape[1]):
            if i in to_drop:
                continue
            for j in range(i + 1, upper.shape[1]):
                if upper[i, j] > 0.8:
                    to_drop.add(j)
        self._keep_idx = sorted(set(range(X_var.shape[1])) - to_drop)

    def _apply_filter(self, X):
        X_var = self._vt.transform(X)
        return X_var[:, self._keep_idx]

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._learn_filter(X_train)
        X_filtered = self._apply_filter(X_train)

        self.model = RandomForestRegressor(
            n_estimators=100,
            n_jobs=8,
            random_state=self.seed,
        )
        self.model.fit(X_filtered, y_train)

    def predict(self, X):
        X_filtered = self._apply_filter(X)
        return self.model.predict(X_filtered)
