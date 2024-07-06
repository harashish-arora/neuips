"""Mordred 2D descriptor featurizer."""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from mordred import Calculator, descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)


class MordredFeaturizer:
    """Compute Mordred 2D molecular descriptors.

    Computes all 2D Mordred descriptors, dropping FilterItLogS and SLogP
    (target-leaking descriptors per Tayyebi et al.).
    Variance and correlation filtering is handled downstream by the model.
    """

    def __init__(self):
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.feature_names = None

    def transform_single(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles("C")  # fallback
        result = self.calc(mol)
        feats = {}
        for desc, val in zip(self.calc.descriptors, result):
            name = str(desc)
            if name in ("FilterItLogS", "SLogP"):
                continue
            try:
                feats[name] = float(val)
            except (ValueError, TypeError):
                feats[name] = 0.0
        return feats

    def transform(self, smiles_list) -> pd.DataFrame:
        """Compute features for a list of SMILES, with caching."""
        cache = {}
        rows = []
        for s in smiles_list:
            if s not in cache:
                cache[s] = self.transform_single(s)
            rows.append(cache[s])
        df = pd.DataFrame(rows)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df = df.fillna(0)
        if self.feature_names is None:
            self.feature_names = list(df.columns)
        return df

    def __repr__(self):
        n = len(self.feature_names) if self.feature_names else "?"
        return f"MordredFeaturizer(n_features={n})"
