"""Base class for all benchmark methods."""

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseMethod(ABC):
    """Abstract base class for a benchmark method."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    @classmethod
    def info(cls) -> dict:
        """Return method metadata."""
        return {
            "name": cls.__name__,
            "featurizer": "rdkit",
            "mode": "dual",       # "dual" = solute+solvent, "single" = solute only
            "gpu_required": False,
        }

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict LogS values."""

    def save(self, path: str):
        """Save the trained model to disk (pickle by default)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str):
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj
