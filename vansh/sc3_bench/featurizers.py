"""
Featurizers for SC3 benchmark.

Supported featurizers:
  rdkit        – 158 RDKit 2D descriptors per molecule
  morgan       – 1024-bit Morgan ECFP4 fingerprints
  dissolvr     – RDKit + MOSE topology + Joback thermo + Abraham proxies
  mordred      – ~1600 Mordred 2D descriptors (FilterItLogS/SLogP excluded)
  maccs        – 166-bit MACCS substructure keys (from RDKit)
  atompair     – 1024-bit Atom-Pair fingerprints (RDKit, distance encoding)
  abraham_only – just the 5 Abraham proxy descriptors (A, B, S, E, V) for
                 a clean "domain-features-only" representation baseline

build_features() concatenates solute + solvent features with 4 temperature
features to produce the final X matrix.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.logger().setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
# RDKit 2D descriptors
# ---------------------------------------------------------------------------

class RDKitFeaturizer:
    def __init__(self, exclude_correlated: bool = True):
        self.enumerator = rdMolStandardize.TautomerEnumerator()
        exclude_prefixes = []
        exclude_exact = {"Ipc"}
        if exclude_correlated:
            exclude_prefixes = [
                "BCUT2D", "SMR_VSA", "SlogP_VSA", "VSA_EState",
                "Chi", "FpDensityMorgan",
            ]
            exclude_exact |= {"Kappa3", "HeavyAtomMolWt", "ExactMolWt"}

        self.desc_list = [
            (name, func) for name, func in Descriptors.descList
            if name not in exclude_exact
            and not any(name.startswith(p) for p in exclude_prefixes)
        ]
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


# ---------------------------------------------------------------------------
# Morgan (ECFP) fingerprints
# ---------------------------------------------------------------------------

class MorganFeaturizer:
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        self.radius = radius
        self.n_bits = n_bits
        self.feature_names = [f"Morgan_{i}" for i in range(n_bits)]
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits,
        )

    def transform_single(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        return self._gen.GetFingerprintAsNumPy(mol).astype(np.float32)

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


# ---------------------------------------------------------------------------
# Dissolvr featurizer: RDKit + MOSE topology + Joback thermo + Abraham proxies
# ---------------------------------------------------------------------------

class DissolvrFeaturizer:
    """Full Dissolvr feature set (domain-expert descriptors)."""

    def __init__(self):
        self.enumerator = rdMolStandardize.TautomerEnumerator()
        exclude_prefixes = [
            "BCUT2D", "SMR_VSA", "SlogP_VSA", "VSA_EState",
            "Chi", "FpDensityMorgan",
        ]
        exclude_exact = {"Ipc", "Kappa3", "MolMR", "HeavyAtomMolWt", "ExactMolWt"}
        self.desc_map = {
            name: func for name, func in Descriptors.descList
            if name not in exclude_exact
            and not any(name.startswith(p) for p in exclude_prefixes)
        }
        self.feature_names = None

    def _get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            return self.enumerator.Canonicalize(mol)
        except Exception:
            return mol

    @staticmethod
    def _mose_features(mol):
        A = Chem.GetAdjacencyMatrix(mol).astype(float)
        degrees = np.sum(A, axis=0)
        powers = [A]
        for _ in range(7):
            powers.append(powers[-1] @ A)
        feats = {}
        for k in range(3, 9):
            feats[f"mose_cyc{k}"] = float(np.trace(powers[k - 1]))
        for k in range(3, 6):
            feats[f"mose_path{k}"] = float(np.sum(powers[k - 2]))
        feats["mose_branched_4"] = float(np.sum(np.power(degrees, 3)))
        feats["mose_star_5"]     = float(np.sum(np.power(degrees, 4)))
        feats["mose_benzene"] = len(mol.GetSubstructMatches(Chem.MolFromSmarts("c1ccccc1")))
        feats["mose_fused"]   = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]")))
        return feats

    @staticmethod
    def _thermo_proxies(mol):
        joback = {
            "ch3": ("[CH3;X4;!R]", -5.10), "ch2_c": ("[CH2;X4;!R]", 11.27),
            "ch_c": ("[CH1;X4;!R]", 12.64), "c_c": ("[CH0;X4;!R]", 46.43),
            "ch2_r": ("[CH2;X4;R]", 8.25),  "ch_r": ("[CH1;X4;R]", 20.15),
            "c_r": ("[CH0;X4;R]", 37.40),   "c=c_c": ("[CX3;!R]=[CX3;!R]", 4.18),
            "c=c_r": ("[c,C;R]=[c,C;R]", 13.02),
            "F": ("[F]", 9.88), "Cl": ("[Cl]", 17.51),
            "Br": ("[Br]", 26.15), "I": ("[I]", 37.0),
            "oh_a": ("[OH;!#6a]", 20.0), "oh_p": ("[OH;a]", 44.45),
            "ether_c": ("[OD2;!R]([#6])[#6]", 22.42),
            "ether_r": ("[OD2;R]([#6])[#6]", 31.22),
            "co": ("[CX3]=[OX1]", 26.15),
            "ester": ("[CX3](=[OX1])[OX2H0]", 30.0),
            "nh2": ("[NH2]", 25.72), "nh_c": ("[NH1;!R]", 27.15),
            "nh_r": ("[NH1;R]", 30.12),
            "nitro": ("[NX3](=[OX1])=[OX1]", 45.0),
            "nitrile": ("[NX1]#[CX2]", 33.15),
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
        proxy_S = hetero_count * 0.2 + (
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
            "pred_Tm":    pred_Tm,
            "abraham_A":  rd_hbd * 0.1 + acid_ref * 0.4,
            "abraham_B":  rd_hba * 0.1 + base_ref * 0.3,
            "abraham_S":  proxy_S,
            "abraham_E":  proxy_E,
            "abraham_V":  v_total / 100.0,
        }

    def transform_single(self, smiles: str) -> dict:
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
        return f"DissolvrFeaturizer(n_features={n})"


# ---------------------------------------------------------------------------
# Mordred 2D descriptors
# ---------------------------------------------------------------------------

class MordredFeaturizer:
    """Mordred 2D descriptors (FilterItLogS & SLogP excluded to avoid leakage)."""

    def __init__(self):
        # Mordred uses deprecated numpy.product; patch before import
        if not hasattr(np, "product"):
            np.product = np.prod
        from mordred import Calculator, descriptors as mordred_descriptors
        self.calc = Calculator(mordred_descriptors, ignore_3D=True)
        self.feature_names = None

    def transform_single(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles("C")
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


# ---------------------------------------------------------------------------
# MACCS keys (166-bit substructure fingerprint)
# ---------------------------------------------------------------------------

class MACCSFeaturizer:
    """166-bit MACCS substructure keys (the canonical SMARTS-based fingerprint).

    MACCSkeys.GenMACCSKeys returns a 167-bit vector where bit 0 is unused;
    we keep all 167 bits so columns line up with the RDKit definition.
    """

    def __init__(self):
        self.n_bits = 167
        self.feature_names = [f"MACCS_{i}" for i in range(self.n_bits)]

    def transform_single(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        bv = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(self.n_bits, dtype=np.float32)
        for i in bv.GetOnBits():
            arr[i] = 1.0
        return arr

    def transform(self, smiles_list) -> pd.DataFrame:
        cache = {}
        rows = []
        for s in smiles_list:
            if s not in cache:
                cache[s] = self.transform_single(s)
            rows.append(cache[s])
        return pd.DataFrame(np.vstack(rows), columns=self.feature_names)

    def __repr__(self):
        return f"MACCSFeaturizer(n_bits={self.n_bits})"


# ---------------------------------------------------------------------------
# Atom-Pair fingerprints (topological pair encoding)
# ---------------------------------------------------------------------------

class AtomPairFeaturizer:
    """1024-bit folded atom-pair fingerprints.

    Encodes pairs of atoms (typed by element/aromaticity/charge/etc.) along
    with the topological distance between them, giving a different view of
    the molecule than Morgan's circular subgraphs.
    """

    def __init__(self, n_bits: int = 1024):
        self.n_bits = n_bits
        self.feature_names = [f"AP_{i}" for i in range(n_bits)]
        self._gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)

    def transform_single(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits, dtype=np.float32)
        return self._gen.GetFingerprintAsNumPy(mol).astype(np.float32)

    def transform(self, smiles_list) -> pd.DataFrame:
        cache = {}
        rows = []
        for s in smiles_list:
            if s not in cache:
                cache[s] = self.transform_single(s)
            rows.append(cache[s])
        return pd.DataFrame(np.vstack(rows), columns=self.feature_names)

    def __repr__(self):
        return f"AtomPairFeaturizer(n_bits={self.n_bits})"


# ---------------------------------------------------------------------------
# Abraham-only featurizer: the five proxy LSER descriptors A/B/S/E/V
# ---------------------------------------------------------------------------
#
# This is a deliberately *minimal* representation – only the 5 Abraham proxies
# (extracted from DissolvrFeaturizer._thermo_proxies) plus a Tm proxy.  It
# isolates the contribution of pure domain (LSER) features from any of the
# RDKit / topology / fingerprint signal.
#
# Six features per molecule:
#   pred_Tm, abraham_A, abraham_B, abraham_S, abraham_E, abraham_V
# These are exactly the same definitions DissolvrFeaturizer uses, just kept
# in their own class so we can study representation purely.

class AbrahamOnlyFeaturizer:
    """Six Abraham/Joback proxy descriptors per molecule (no RDKit topology)."""

    def __init__(self):
        self.feature_names = [
            "pred_Tm", "abraham_A", "abraham_B",
            "abraham_S", "abraham_E", "abraham_V",
        ]

    def transform_single(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {n: 0.0 for n in self.feature_names}
        return DissolvrFeaturizer._thermo_proxies(mol)

    def transform(self, smiles_list) -> pd.DataFrame:
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
        return f"AbrahamOnlyFeaturizer(n_features={len(self.feature_names)})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FEATURIZER_REGISTRY = {
    "rdkit":        RDKitFeaturizer,
    "morgan":       MorganFeaturizer,
    "dissolvr":     DissolvrFeaturizer,
    "mordred":      MordredFeaturizer,
    "maccs":        MACCSFeaturizer,
    "atompair":     AtomPairFeaturizer,
    "abraham_only": AbrahamOnlyFeaturizer,
}


def get_featurizer(name: str, **kwargs):
    if name not in FEATURIZER_REGISTRY:
        raise ValueError(f"Unknown featurizer: {name}. Available: {list(FEATURIZER_REGISTRY)}")
    return FEATURIZER_REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Feature builder (solute + solvent + temperature → X matrix)
# ---------------------------------------------------------------------------

_feat_caches: dict[str, dict] = {}


def build_features(df: pd.DataFrame, featurizer, cache: dict | None = None) -> np.ndarray:
    """Build [solute_feats | solvent_feats | 4 temp_feats] feature matrix."""
    if cache is None:
        cache = _feat_caches.setdefault(repr(featurizer), {})

    solute_smiles = df["Solute"].tolist()
    solvent_smiles = df["Solvent"].tolist()

    for s in set(solute_smiles + solvent_smiles):
        if s not in cache:
            cache[s] = featurizer.transform_single(s)

    solute_rows  = [cache[s] for s in solute_smiles]
    solvent_rows = [cache[s] for s in solvent_smiles]

    if isinstance(solute_rows[0], dict):
        sol_df  = pd.DataFrame(solute_rows)
        solv_df = pd.DataFrame(solvent_rows)
        solv_df.columns = [f"solv_{c}" for c in solv_df.columns]
    else:
        sol_df  = pd.DataFrame(np.vstack(solute_rows), columns=featurizer.feature_names)
        solv_df = pd.DataFrame(
            np.vstack(solvent_rows),
            columns=[f"solv_{c}" for c in featurizer.feature_names],
        )

    T = df["Temperature"].values.astype(np.float32)
    temp_df = pd.DataFrame({
        "T_norm": T / 300.0,
        "T_inv":  1000.0 / T,
        "T_sq":   (T / 300.0) ** 2,
        "T_log":  np.log(T / 300.0),
    })

    X = pd.concat([
        sol_df.reset_index(drop=True),
        solv_df.reset_index(drop=True),
        temp_df.reset_index(drop=True),
    ], axis=1)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    return X.values.astype(np.float32)
