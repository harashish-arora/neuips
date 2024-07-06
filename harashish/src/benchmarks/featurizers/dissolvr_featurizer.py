"""
Dissolvr featurizer: RDKit descriptors + MOSE topology + Joback thermo proxies + Abraham.

Adapted from Dissolvr/baselines/regime-ii/featurizer.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


class DissolvrFeaturizer:
    """Full Dissolvr feature set: RDKit + MOSE + thermo proxies."""

    def __init__(self):
        self.enumerator = rdMolStandardize.TautomerEnumerator()

        exclude_prefixes = ["BCUT2D", "SMR_VSA", "SlogP_VSA", "VSA_EState", "Chi", "FpDensityMorgan"]
        exclude_exact = {"Ipc", "Kappa3", "MolMR", "HeavyAtomMolWt", "ExactMolWt"}

        self.desc_map = {}
        for name, func in Descriptors.descList:
            if name in exclude_exact:
                continue
            if any(name.startswith(p) for p in exclude_prefixes):
                continue
            self.desc_map[name] = func

    def _get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return self.enumerator.Canonicalize(mol)
        except Exception:
            return mol

    def _mose_features(self, mol):
        A = Chem.GetAdjacencyMatrix(mol).astype(float)
        degrees = np.sum(A, axis=0)
        A2 = np.linalg.matrix_power(A, 2)
        A3 = A2 @ A
        A4 = A3 @ A
        A5 = A4 @ A
        A6 = A5 @ A
        A7 = A6 @ A
        A8 = A7 @ A
        return {
            "mose_cyc3": float(np.trace(A3)),
            "mose_cyc4": float(np.trace(A4)),
            "mose_cyc5": float(np.trace(A5)),
            "mose_cyc6": float(np.trace(A6)),
            "mose_cyc7": float(np.trace(A7)),
            "mose_cyc8": float(np.trace(A8)),
            "mose_path3": float(np.sum(A2)),
            "mose_path4": float(np.sum(A3)),
            "mose_path5": float(np.sum(A4)),
            "mose_branched_4": float(np.sum(np.power(degrees, 3))),
            "mose_star_5": float(np.sum(np.power(degrees, 4))),
            "mose_benzene": len(mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1"))),
            "mose_fused": len(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]"))),
        }

    def _thermo_proxies(self, mol):
        joback = {
            "ch3": ("[CH3;X4;!R]", -5.10), "ch2_c": ("[CH2;X4;!R]", 11.27),
            "ch_c": ("[CH1;X4;!R]", 12.64), "c_c": ("[CH0;X4;!R]", 46.43),
            "ch2_r": ("[CH2;X4;R]", 8.25), "ch_r": ("[CH1;X4;R]", 20.15),
            "c_r": ("[CH0;X4;R]", 37.40), "c=c_c": ("[CX3;!R]=[CX3;!R]", 4.18),
            "c=c_r": ("[c,C;R]=[c,C;R]", 13.02), "F": ("[F]", 9.88),
            "Cl": ("[Cl]", 17.51), "Br": ("[Br]", 26.15), "I": ("[I]", 37.0),
            "oh_a": ("[OH;!#6a]", 20.0), "oh_p": ("[OH;a]", 44.45),
            "ether_c": ("[OD2;!R]([#6])[#6]", 22.42), "ether_r": ("[OD2;R]([#6])[#6]", 31.22),
            "co": ("[CX3]=[OX1]", 26.15), "ester": ("[CX3](=[OX1])[OX2H0]", 30.0),
            "nh2": ("[NH2]", 25.72), "nh_c": ("[NH1;!R]", 27.15), "nh_r": ("[NH1;R]", 30.12),
            "nitro": ("[NX3](=[OX1])=[OX1]", 45.0), "nitrile": ("[NX1]#[CX2]", 33.15),
        }
        tm_sum = 122.5 + sum(
            len(mol.GetSubstructMatches(Chem.MolFromSmarts(p))) * w
            for p, w in joback.values()
        )
        pred_Tm = max(tm_sum, 150.0)

        rd_hbd = Descriptors.NumHDonors(mol)
        rd_hba = Descriptors.NumHAcceptors(mol)
        acid_ref = sum(
            len(mol.GetSubstructMatches(Chem.MolFromSmarts(s)))
            for s in ["[OH]c", "C(=O)[OH]"]
        )
        base_ref = sum(
            len(mol.GetSubstructMatches(Chem.MolFromSmarts(s)))
            for s in ["[NH2,NH1,NH0]", "n1ccccc1", "[CX3]=[OX1]"]
        )
        proxy_E = Descriptors.MolMR(mol) / 10.0
        hetero_count = len(mol.GetSubstructMatches(
            Chem.MolFromSmarts("[O,N,S,F,Cl,Br,I,P,Se,B,Na,K,Ca,Li]")
        ))
        proxy_S = (hetero_count * 0.2) + (
            len(mol.GetSubstructMatches(Chem.MolFromSmarts("a1aaaaa1"))) * 0.3
        )
        atom_v_map = {
            "C": 16.35, "N": 14.39, "O": 12.43, "F": 10.48, "Cl": 20.95,
            "Br": 26.21, "I": 34.53, "S": 22.91, "P": 24.87, "H": 8.71,
            "Se": 25.10, "B": 18.32, "Na": 18.00, "K": 24.00, "Ca": 21.00,
            "Li": 14.00,
        }
        v_total = sum(atom_v_map.get(a.GetSymbol(), 15.0) for a in mol.GetAtoms())
        v_total += sum(a.GetTotalNumHs() for a in mol.GetAtoms()) * 8.71
        v_total -= 6.56 * mol.GetNumBonds()

        return {
            "pred_Tm": pred_Tm,
            "abraham_A": rd_hbd * 0.1 + acid_ref * 0.4,
            "abraham_B": rd_hba * 0.1 + base_ref * 0.3,
            "abraham_S": proxy_S,
            "abraham_E": proxy_E,
            "abraham_V": v_total / 100.0,
        }

    def _calc_feats(self, smiles):
        mol = self._get_mol(smiles)
        if mol is None:
            return {}
        feats = {}
        feats.update(self._mose_features(mol))
        feats.update(self._thermo_proxies(mol))
        for name, func in self.desc_map.items():
            try:
                feats[name] = float(func(mol))
            except Exception:
                feats[name] = 0.0
        return feats

    def transform(self, smiles_list) -> pd.DataFrame:
        cache = {}
        rows = []
        for s in smiles_list:
            if s not in cache:
                cache[s] = self._calc_feats(s)
            rows.append(cache[s])
        df = pd.DataFrame(rows)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        return df.fillna(0)

    def __repr__(self):
        return "DissolvrFeaturizer()"
