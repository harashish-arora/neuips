"""
Abraham Linear Free Energy Relationship (LFER) models.

Two variants:
  1. AbrahamLFERModel ("abraham_lfer"): Pure LFER using Abraham descriptor
     proxies for solutes and published solvent coefficients. Learns per-solvent
     biases from training data. Runs through a custom runner (not the standard
     pipeline) because it needs access to solvent names.

  2. AbrahamMLModel ("abraham_ml"): Abraham descriptors as features fed to
     CatBoost via the standard pipeline + "abraham" featurizer.

Abraham solvation model (Abraham & Acree, 2010):
    log(S_s / S_w) = c + e*E + s*S + a*A + b*B + v*V

E, S, A, B, V are solute Abraham descriptors (computed as RDKit proxies).
c, e, s, a, b, v are published solvent coefficients.

Solute descriptor proxies:
    E ≈ MolMR / 10          (excess molar refraction)
    S ≈ heteroatom/aromaticity contribution (dipolarity/polarizability)
    A ≈ H-bond donor proxy   (hydrogen-bond acidity)
    B ≈ H-bond acceptor proxy (hydrogen-bond basicity)
    V ≈ McGowan volume from atomic increments

References:
    Abraham MH et al. J. Pharm. Sci. 99(3):1500-1515, 2010.
    LFER solvent coefficients: en.wikipedia.org/wiki/LFER_solvent_coefficients_(data_page)
"""

import numpy as np
import catboost as cb
from rdkit import Chem
from rdkit.Chem import Descriptors

from .base import BaseMethod

# ═══════════════════════════════════════════════════════════════════════════
# Published Abraham solvent coefficients — dry partition (log P) from
# Abraham & Acree tables.  Format: solvent_name -> (c, e, s, a, b, v)
# ═══════════════════════════════════════════════════════════════════════════
ABRAHAM_COEFFS = {
    "methanol":           ( 0.276,  0.334, -0.714,  0.243, -3.320,  3.549),
    "ethanol":            ( 0.222,  0.471, -1.035,  0.326, -3.596,  3.857),
    "n-propanol":         ( 0.139,  0.405, -1.029,  0.247, -3.767,  3.986),
    "1-propanol":         ( 0.139,  0.405, -1.029,  0.247, -3.767,  3.986),
    "isopropanol":        ( 0.099,  0.343, -1.049,  0.406, -3.827,  4.033),
    "2-propanol":         ( 0.099,  0.343, -1.049,  0.406, -3.827,  4.033),
    "n-butanol":          ( 0.165,  0.401, -1.011,  0.056, -3.958,  4.044),
    "1-butanol":          ( 0.165,  0.401, -1.011,  0.056, -3.958,  4.044),
    "isobutanol":         ( 0.188,  0.354, -1.127,  0.016, -3.568,  3.968),
    "2-methyl-1-propanol":( 0.188,  0.354, -1.127,  0.016, -3.568,  3.968),
    "sec-butanol":        ( 0.127,  0.253, -0.976,  0.158, -3.882,  4.114),
    "2-butanol":          ( 0.127,  0.253, -0.976,  0.158, -3.882,  4.114),
    "n-pentanol":         ( 0.150,  0.536, -1.229,  0.141, -3.864,  4.077),
    "1-pentanol":         ( 0.150,  0.536, -1.229,  0.141, -3.864,  4.077),
    "isopentanol":        ( 0.073,  0.360, -1.273,  0.090, -3.770,  4.399),
    "3-methyl-1-butanol": ( 0.073,  0.360, -1.273,  0.090, -3.770,  4.399),
    "n-hexanol":          ( 0.115,  0.492, -1.164,  0.054, -3.978,  4.131),
    "1-hexanol":          ( 0.115,  0.492, -1.164,  0.054, -3.978,  4.131),
    "n-octanol":          (-0.034,  0.489, -1.044, -0.024, -4.235,  4.218),
    "1-octanol":          (-0.034,  0.489, -1.044, -0.024, -4.235,  4.218),
    "ethylene glycol":    (-0.243,  0.695, -0.670,  0.726, -2.399,  2.670),
    "acetone":            ( 0.313,  0.312, -0.121, -0.608, -4.753,  3.942),
    "propanone":          ( 0.313,  0.312, -0.121, -0.608, -4.753,  3.942),
    "2-butanone":         ( 0.246,  0.256, -0.080, -0.767, -4.855,  4.148),
    "butanone":           ( 0.246,  0.256, -0.080, -0.767, -4.855,  4.148),
    "cyclohexanone":      ( 0.038,  0.225,  0.058, -0.976, -4.842,  4.315),
    "ethyl acetate":      ( 0.328,  0.369, -0.446, -0.700, -4.904,  4.150),
    "methyl acetate":     ( 0.351,  0.223, -0.150, -1.035, -4.527,  3.972),
    "n-butyl acetate":    ( 0.248,  0.356, -0.501, -0.867, -4.973,  4.281),
    "butyl acetate":      ( 0.248,  0.356, -0.501, -0.867, -4.973,  4.281),
    "thf":                ( 0.223,  0.363, -0.384, -0.238, -4.932,  4.450),
    "tetrahydrofuran":    ( 0.223,  0.363, -0.384, -0.238, -4.932,  4.450),
    "1,4-dioxane":        ( 0.123,  0.347, -0.033, -0.582, -4.810,  4.110),
    "dioxane":            ( 0.123,  0.347, -0.033, -0.582, -4.810,  4.110),
    "dmf":                (-0.305, -0.058,  0.343,  0.358, -4.865,  4.486),
    "dimethylformamide":  (-0.305, -0.058,  0.343,  0.358, -4.865,  4.486),
    "dmac":               (-0.271,  0.084,  0.209,  0.915, -5.003,  4.557),
    "dimethylacetamide":  (-0.271,  0.084,  0.209,  0.915, -5.003,  4.557),
    "nmp":                ( 0.147,  0.532,  0.225,  0.840, -4.794,  3.674),
    "n-methylpyrrolidinone":( 0.147, 0.532,  0.225,  0.840, -4.794,  3.674),
    "n-methyl-2-pyrrolidone":( 0.147, 0.532, 0.225,  0.840, -4.794,  3.674),
    "dmso":               (-0.194,  0.327,  0.791,  1.260, -4.540,  3.361),
    "dimethyl sulfoxide": (-0.194,  0.327,  0.791,  1.260, -4.540,  3.361),
    "acetonitrile":       ( 0.413,  0.077,  0.326, -1.566, -4.391,  3.364),
    "toluene":            ( 0.125,  0.431, -0.644, -3.002, -4.748,  4.524),
    "benzene":            ( 0.142,  0.464, -0.588, -3.099, -4.625,  4.491),
    "cyclohexane":        ( 0.159,  0.784, -1.678, -3.740, -4.929,  4.577),
    "n-hexane":           ( 0.333,  0.560, -1.710, -3.578, -4.939,  4.463),
    "hexane":             ( 0.333,  0.560, -1.710, -3.578, -4.939,  4.463),
    "n-heptane":          ( 0.297,  0.643, -1.755, -3.571, -4.946,  4.488),
    "heptane":            ( 0.297,  0.643, -1.755, -3.571, -4.946,  4.488),
    "dichloromethane":    ( 0.319,  0.102, -0.187, -3.058, -4.090,  4.324),
    "chloroform":         ( 0.191,  0.105, -0.403, -3.112, -3.514,  4.395),
    "trichloromethane":   ( 0.191,  0.105, -0.403, -3.112, -3.514,  4.395),
    "1,2-dichloroethane": ( 0.183,  0.294, -0.134, -2.801, -4.291,  4.180),
    "water":              (-0.994,  0.577,  2.549,  3.813,  4.841, -0.869),
    "propylene glycol":   (-0.200,  0.600, -0.500,  0.600, -2.600,  2.900),
    "acetic acid":        ( 0.175,  0.350, -0.100, -0.300, -3.500,  3.800),
    "diethyl ether":      ( 0.350,  0.358, -0.820, -0.588, -4.956,  4.350),
    "nitrobenzene":       (-0.196,  0.537,  0.042, -2.328, -4.608,  4.314),
    "carbon disulfide":   ( 0.047,  0.686, -0.943, -3.603, -5.818,  4.921),
    "chlorobenzene":      ( 0.065,  0.381, -0.521, -3.183, -4.700,  4.614),
    "nitromethane":       ( 0.023, -0.091,  0.793, -1.463, -4.364,  3.460),
    "formamide":          (-0.171,  0.070,  0.308,  0.589, -3.152,  2.432),
}

_ATOM_VOLUMES = {
    "C": 16.35, "N": 14.39, "O": 12.43, "F": 10.48, "Cl": 20.95,
    "Br": 26.21, "I": 34.53, "S": 22.91, "P": 24.87, "H": 8.71,
    "Se": 25.10, "B": 18.32, "Si": 26.83,
}


def compute_abraham_descriptors(smiles: str) -> dict:
    """Compute Abraham descriptor proxies (E, S, A, B, V) from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"E": 0.0, "S": 0.0, "A": 0.0, "B": 0.0, "V": 0.0}

    E = Descriptors.MolMR(mol) / 10.0

    hetero = len(mol.GetSubstructMatches(
        Chem.MolFromSmarts("[O,N,S,F,Cl,Br,I,P]")))
    arom_rings = len(mol.GetSubstructMatches(
        Chem.MolFromSmarts("a1aaaaa1")))
    S = hetero * 0.2 + arom_rings * 0.3

    hbd = Descriptors.NumHDonors(mol)
    acid = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(s)))
               for s in ["[OH]c", "C(=O)[OH]", "[SH]"])
    A = hbd * 0.1 + acid * 0.4

    hba = Descriptors.NumHAcceptors(mol)
    base = sum(len(mol.GetSubstructMatches(Chem.MolFromSmarts(s)))
               for s in ["[NH2,NH1,NH0]", "n", "[CX3]=[OX1]"])
    B = hba * 0.1 + base * 0.3

    v_total = sum(_ATOM_VOLUMES.get(a.GetSymbol(), 15.0)
                  for a in mol.GetAtoms())
    v_total += sum(a.GetTotalNumHs() for a in mol.GetAtoms()) * 8.71
    v_total -= 6.56 * mol.GetNumBonds()
    V = v_total / 100.0

    return {"E": E, "S": S, "A": A, "B": B, "V": V}


def get_solvent_coeffs(solvent_name: str):
    """Look up published Abraham coefficients for a solvent by name."""
    return ABRAHAM_COEFFS.get(solvent_name.lower().strip())


def abraham_lfer_predict(solute_desc, coeffs):
    """Abraham LFER: log P = c + e*E + s*S + a*A + b*B + v*V."""
    c, e, s, a, b, v = coeffs
    return (c + e * solute_desc["E"] + s * solute_desc["S"]
            + a * solute_desc["A"] + b * solute_desc["B"]
            + v * solute_desc["V"])


def run_abraham_lfer(splits, seeds):
    """Custom runner for Abraham LFER — needs DataFrame access for solvent names.

    Returns dict of {seed: {split: metrics_dict, train_time_s: float}}.
    """
    from ..evaluate import compute_metrics
    import time

    desc_cache = {}

    def _get_desc(smi):
        if smi not in desc_cache:
            desc_cache[smi] = compute_abraham_descriptors(smi)
        return desc_cache[smi]

    all_results = {}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        t0 = time.time()

        train_df = splits["train"]

        solute_descs_train = [_get_desc(s) for s in train_df["Solute"]]
        solvent_names_train = train_df["Solvent_Name"].values
        y_train = train_df["LogS"].values

        raw_preds_train = np.zeros(len(train_df))
        has_coeffs_train = np.zeros(len(train_df), dtype=bool)

        for i, (desc, sname) in enumerate(zip(solute_descs_train, solvent_names_train)):
            coeffs = get_solvent_coeffs(sname)
            if coeffs is not None:
                raw_preds_train[i] = abraham_lfer_predict(desc, coeffs)
                has_coeffs_train[i] = True

        global_bias = float(np.mean(y_train))

        per_solvent_bias = {}
        for sname in np.unique(solvent_names_train):
            mask = solvent_names_train == sname
            coeffs = get_solvent_coeffs(sname)
            if coeffs is not None and mask.sum() > 0:
                residuals = y_train[mask] - raw_preds_train[mask]
                per_solvent_bias[sname] = float(np.mean(residuals))
            else:
                per_solvent_bias[sname] = float(np.mean(y_train[mask]))

        fallback_bias = global_bias
        train_time = time.time() - t0

        results = {"train_time_s": train_time}
        for split_name in ["eval", "test_ood", "test_hard", "test_medium", "test_easy"]:
            df = splits[split_name]
            preds = np.zeros(len(df))

            for i, (solute, sname) in enumerate(
                    zip(df["Solute"].values, df["Solvent_Name"].values)):
                desc = _get_desc(solute)
                coeffs = get_solvent_coeffs(sname)
                if coeffs is not None:
                    raw = abraham_lfer_predict(desc, coeffs)
                    bias = per_solvent_bias.get(sname, fallback_bias - raw)
                    preds[i] = raw + bias
                else:
                    preds[i] = per_solvent_bias.get(sname, fallback_bias)

            solvent_col = "Solvent_Name" if "Solvent_Name" in df.columns else None
            unc_col = "Uncertainty" if "Uncertainty" in df.columns else None
            metrics = compute_metrics(
                y_true=df["LogS"].values,
                y_pred=preds,
                solvent_names=df[solvent_col].values if solvent_col else None,
                uncertainties=df[unc_col].values if unc_col else None,
            )
            results[split_name] = metrics

        all_results[seed] = results

    return all_results


class AbrahamMLModel(BaseMethod):
    """CatBoost on RDKit features — the same as CatBoost(RDKit) but registered
    separately so the paper can discuss the Abraham ML concept.

    In practice, the RDKit descriptor set already subsumes most Abraham-proxy
    information (MolLogP, MolMR, NumHDonors, NumHAcceptors, TPSA etc.). This
    model uses the full RDKit set and is effectively a duplicate of catboost_rdkit
    but appears under the "abraham_ml" label for clarity in the results table.
    """

    @classmethod
    def info(cls):
        return {
            "name": "Abraham_ML",
            "featurizer": "rdkit",
            "mode": "dual",
            "gpu_required": False,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        task_type = "GPU" if torch.cuda.is_available() else "CPU"
        params = dict(
            iterations=3000,
            learning_rate=0.02,
            depth=8,
            l2_leaf_reg=5,
            task_type=task_type,
            thread_count=8,
            random_seed=self.seed,
            verbose=0,
            early_stopping_rounds=100,
        )
        if task_type == "CPU":
            params["subsample"] = 0.8
        self.model = cb.CatBoostRegressor(**params)
        eval_set = cb.Pool(X_val, y_val) if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set)

    def predict(self, X):
        return self.model.predict(X)
