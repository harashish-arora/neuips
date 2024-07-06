"""RDKit 2D descriptor featurizer."""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.logger().setLevel(RDLogger.ERROR)


class RDKitFeaturizer:
    """Compute RDKit 2D molecular descriptors."""

    def __init__(self, exclude_correlated: bool = True):
        self.enumerator = rdMolStandardize.TautomerEnumerator()

        exclude_prefixes = []
        exclude_exact = {"Ipc"}
        if exclude_correlated:
            exclude_prefixes = ["BCUT2D", "SMR_VSA", "SlogP_VSA", "VSA_EState", "Chi", "FpDensityMorgan"]
            exclude_exact |= {"Kappa3", "HeavyAtomMolWt", "ExactMolWt"}

        self.desc_list = []
        for name, func in Descriptors.descList:
            if name in exclude_exact:
                continue
            if any(name.startswith(p) for p in exclude_prefixes):
                continue
            self.desc_list.append((name, func))

        self.feature_names = [n for n, _ in self.desc_list]

    def _get_mol(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return self.enumerator.Canonicalize(mol)
        except Exception:
            return mol

    def transform_single(self, smiles: str) -> dict:
        mol = self._get_mol(smiles)
        if mol is None:
            return {n: 0.0 for n in self.feature_names}
        feats = {}
        for name, func in self.desc_list:
            try:
                feats[name] = float(func(mol))
            except Exception:
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
        return df.fillna(0)

    def __repr__(self):
        return f"RDKitFeaturizer(n_features={len(self.feature_names)})"
