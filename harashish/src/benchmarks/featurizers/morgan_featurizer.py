"""Morgan (ECFP) fingerprint featurizer."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


class MorganFeaturizer:
    """Compute Morgan (ECFP) fingerprints."""

    def __init__(self, radius: int = 2, n_bits: int = 1024):
        self.radius = radius
        self.n_bits = n_bits
        self.feature_names = [f"Morgan_{i}" for i in range(n_bits)]
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

    def transform_single(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        fp = self._gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.float32)

    def transform(self, smiles_list) -> pd.DataFrame:
        cache = {}
        rows = []
        for s in smiles_list:
            if s not in cache:
                cache[s] = self.transform_single(s)
            rows.append(cache[s])
        return pd.DataFrame(np.vstack(rows), columns=self.feature_names)

    def __repr__(self):
        return f"MorganFeaturizer(radius={self.radius}, n_bits={self.n_bits})"
