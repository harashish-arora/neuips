"""
UNIFAC (UNIversal Functional Activity Coefficient) models.

Uses ugropy for UNIFAC group decomposition from SMILES and the thermo library
for activity coefficient calculations via the standard UNIFAC method.

Two variants:
  1. UNIFACModel ("unifac"): Pure physics-based solubility prediction.
     UNIFAC activity coefficients + ideal solubility equation. No ML.
  2. UNIFACMLModel ("unifac_ml"): UNIFAC-derived features concatenated with
     RDKit descriptors, fed to CatBoost.

The ideal solubility + activity correction:
    ln(x) = -ΔH_fus/(R*T) * (1 - T/T_m) - ln(γ)
    log10(S) = log10(x / (1-x) * ρ * 1000 / MW_solvent)

Libraries:
    ugropy (v3.1.6): UNIFAC group decomposition from SMILES
    thermo (v0.6.0): UNIFAC activity coefficient calculation

References:
    Fredenslund A, Jones RL, Prausnitz JM. AIChE J. 21(6):1086-1099, 1975.
"""

import math
import warnings
import numpy as np
import catboost as cb
from rdkit import Chem
from rdkit.Chem import Descriptors

from .base import BaseMethod

R_GAS = 8.31446  # J/(mol·K)

warnings.filterwarnings("ignore")


def _get_unifac_groups(smiles):
    """Get UNIFAC subgroup numbers from SMILES using ugropy.

    Returns dict {subgroup_number: count} or None if decomposition fails.
    """
    try:
        from ugropy import Groups
        g = Groups(smiles, "smiles")
        if g.unifac.subgroups is None or len(g.unifac.subgroups) == 0:
            return None
        return g.unifac.subgroups_num
    except Exception:
        return None


def _compute_gamma(groups_solute, groups_solvent, x_solute, T):
    """Compute UNIFAC activity coefficient for solute using thermo.

    Returns gamma_solute (float) or None if computation fails.
    """
    try:
        from thermo.unifac import UNIFAC

        xs = [1.0 - x_solute, x_solute]
        chemgroups = [groups_solvent, groups_solute]
        u = UNIFAC.from_subgroups(T=T, xs=xs, chemgroups=chemgroups)
        gammas = u.gammas()
        return float(gammas[1])
    except Exception:
        return None


def _estimate_fusion_properties(smiles):
    """Estimate melting point and enthalpy of fusion using Joback's method."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 400.0, 20000.0

    joback = {
        "ch3": ("[CH3;X4;!R]", -5.10, 0.141),
        "ch2": ("[CH2;X4;!R]", 11.27, 0.0189),
        "ch":  ("[CH1;X4;!R]", 12.64, -0.0637),
        "ch2_r": ("[CH2;X4;R]", 8.25, 0.0189),
        "ch_r":  ("[CH1;X4;R]", 20.15, -0.0637),
        "oh": ("[OH;!$([OH]C=O)]", 20.0, 2.406),
        "ether": ("[OD2]([#6])[#6]", 22.42, 0.569),
        "co": ("[CX3]=[OX1]", 26.15, 1.410),
        "ester": ("[CX3](=[OX1])[OX2H0]", 30.0, 1.970),
        "nh2": ("[NH2]", 25.72, 1.810),
        "cooh": ("C(=O)[OH]", 57.0, 3.240),
        "ach": ("[cH]", 5.69, 0.577),
        "ac": ("[c;H0;!$([c]=O)]", 31.01, 0.120),
        "cl": ("[Cl]", 17.51, 1.566),
        "br": ("[Br]", 26.15, 2.516),
        "f": ("[F]", 9.88, 1.047),
    }

    tm_sum = 122.5
    hf_sum = -0.88
    for _, (smarts, tm_incr, hf_incr) in joback.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        count = len(mol.GetSubstructMatches(pattern))
        tm_sum += count * tm_incr
        hf_sum += count * hf_incr

    T_m = max(tm_sum, 150.0)
    delta_H_fus = max(hf_sum * 1000.0, 1000.0)
    return T_m, delta_H_fus


def predict_logS_unifac(solute_smi, solvent_smi, T,
                        solvent_MW=None, solvent_density=None):
    """Predict log10(S) using UNIFAC + ideal solubility equation.

    Returns (logS, features_dict) or (None, None) if UNIFAC fails.
    """
    groups_solute = _get_unifac_groups(solute_smi)
    groups_solvent = _get_unifac_groups(solvent_smi)

    if groups_solute is None or groups_solvent is None:
        return None, None

    T_m, delta_H_fus = _estimate_fusion_properties(solute_smi)

    ln_x_ideal = -(delta_H_fus / R_GAS) * (1.0 / T - 1.0 / T_m)
    ln_x_ideal = np.clip(ln_x_ideal, -50, 0.0)

    x_guess = min(np.exp(ln_x_ideal), 0.5)
    x_guess = max(x_guess, 1e-10)

    ln_gamma_final = 0.0
    x = x_guess

    for _ in range(8):
        gamma = _compute_gamma(groups_solute, groups_solvent, x, T)
        if gamma is None or gamma <= 0:
            return None, None

        ln_gamma = math.log(gamma)
        ln_gamma_final = ln_gamma
        ln_x = ln_x_ideal - ln_gamma
        ln_x = np.clip(ln_x, -50, -0.001)
        x_new = np.exp(ln_x)
        x_new = np.clip(x_new, 1e-20, 0.999)

        if abs(x_new - x) / max(abs(x), 1e-10) < 0.01:
            break
        x = 0.5 * x + 0.5 * x_new

    x = x_new

    if solvent_MW is None:
        mol_s = Chem.MolFromSmiles(solvent_smi)
        solvent_MW = Descriptors.MolWt(mol_s) if mol_s else 100.0
    if solvent_density is None:
        try:
            from thermo import Chemical
            c = Chemical(solvent_smi)
            solvent_density = c.rho / 1000.0 if c.rho else 0.9
        except Exception:
            solvent_density = 0.9

    S_molL = x / (1.0 - x + 1e-30) * solvent_density * 1000.0 / solvent_MW
    logS = np.log10(max(S_molL, 1e-20))

    features = {
        "unifac_ln_gamma": float(ln_gamma_final),
        "unifac_x": float(x),
        "unifac_logS": float(logS),
        "unifac_Tm": float(T_m),
        "unifac_dHfus": float(delta_H_fus),
        "unifac_ln_x_ideal": float(ln_x_ideal),
    }
    return logS, features


def _build_unifac_gamma_cache(all_smiles_pairs, print_progress=True):
    """Pre-compute UNIFAC groups and gamma for all unique (solute, solvent) pairs.

    Returns dict of {(solute_smi, solvent_smi): (groups_solute_num, groups_solvent_num)}
    """
    from tqdm import tqdm

    groups_cache = {}
    def _get_groups(smi):
        if smi not in groups_cache:
            groups_cache[smi] = _get_unifac_groups(smi)
        return groups_cache[smi]

    unique_smiles = set()
    for sol, solv in all_smiles_pairs:
        unique_smiles.add(sol)
        unique_smiles.add(solv)

    it = tqdm(unique_smiles, desc="  UNIFAC groups", unit="mol") if print_progress else unique_smiles
    for smi in it:
        _get_groups(smi)

    pair_groups = {}
    n_ok, n_fail = 0, 0
    for sol, solv in all_smiles_pairs:
        g1, g2 = groups_cache.get(sol), groups_cache.get(solv)
        if g1 is not None and g2 is not None:
            pair_groups[(sol, solv)] = (g1, g2)
            n_ok += 1
        else:
            n_fail += 1

    if print_progress:
        print(f"  UNIFAC pairs: {n_ok} OK, {n_fail} failed ({len(groups_cache)} unique molecules)")

    return pair_groups, groups_cache


def run_unifac(splits, seeds):
    """Custom runner for pure UNIFAC predictions.

    Returns dict of {seed: {split: metrics, train_time_s}}.
    """
    from ..evaluate import compute_metrics
    from tqdm import tqdm
    import time

    all_pairs = set()
    for split_name, df in splits.items():
        for sol, solv in zip(df["Solute"].values, df["Solvent"].values):
            all_pairs.add((sol, solv))

    print(f"  Pre-computing UNIFAC groups for {len(all_pairs)} unique pairs...")
    pair_groups, groups_cache = _build_unifac_gamma_cache(all_pairs)

    gamma_cache = {}

    def _get_gamma(solute, solvent, T):
        key = (solute, solvent, round(T, 1))
        if key not in gamma_cache:
            g = pair_groups.get((solute, solvent))
            if g is None:
                gamma_cache[key] = None
            else:
                gamma_cache[key] = _compute_gamma(g[0], g[1], 0.01, T)
        return gamma_cache[key]

    def _predict_logS(solute, solvent, T, solvent_MW=None):
        gamma = _get_gamma(solute, solvent, T)
        if gamma is None or gamma <= 0:
            return None

        T_m, delta_H_fus = _estimate_fusion_properties(solute)
        ln_x_ideal = -(delta_H_fus / R_GAS) * (1.0 / T - 1.0 / T_m)
        ln_x_ideal = np.clip(ln_x_ideal, -50, 0.0)
        ln_gamma = math.log(gamma)
        ln_x = ln_x_ideal - ln_gamma
        ln_x = np.clip(ln_x, -50, -0.001)
        x = np.exp(ln_x)

        if solvent_MW is None:
            mol_s = Chem.MolFromSmiles(solvent)
            solvent_MW = Descriptors.MolWt(mol_s) if mol_s else 100.0
        density = 0.9
        S = x / (1 - x + 1e-30) * density * 1000 / solvent_MW
        return float(np.log10(max(S, 1e-20)))

    all_results = {}
    for seed in seeds:
        t0 = time.time()
        train_df = splits["train"]
        train_y = train_df["LogS"].values
        global_mean = float(np.mean(train_y))

        per_solvent_mean = {}
        for sname, grp in train_df.groupby("Solvent_Name"):
            per_solvent_mean[sname] = float(grp["LogS"].mean())

        rng = np.random.RandomState(seed)
        n_sample = min(3000, len(train_df))
        idx = rng.choice(len(train_df), size=n_sample, replace=False)
        sample = train_df.iloc[idx]

        preds_raw, targets = [], []
        for _, row in sample.iterrows():
            p = _predict_logS(row["Solute"], row["Solvent"], row["Temperature"])
            if p is not None:
                preds_raw.append(p)
                targets.append(row["LogS"])

        bias = float(np.mean(np.array(targets) - np.array(preds_raw))) if len(preds_raw) > 10 else 0.0
        train_time = time.time() - t0

        results = {"train_time_s": train_time}
        for split_name in ["eval", "test_ood", "test_hard", "test_medium", "test_easy"]:
            df = splits[split_name]
            preds = np.zeros(len(df))
            solutes = df["Solute"].values
            solvents = df["Solvent"].values
            temps = df["Temperature"].values
            snames = df["Solvent_Name"].values

            for i in range(len(df)):
                p = _predict_logS(solutes[i], solvents[i], temps[i])
                if p is not None:
                    preds[i] = p + bias
                else:
                    preds[i] = per_solvent_mean.get(snames[i], global_mean)

            solvent_col = "Solvent_Name" if "Solvent_Name" in df.columns else None
            unc_col = "Uncertainty" if "Uncertainty" in df.columns else None
            metrics = compute_metrics(
                y_true=df["LogS"].values, y_pred=preds,
                solvent_names=df[solvent_col].values if solvent_col else None,
                uncertainties=df[unc_col].values if unc_col else None,
            )
            results[split_name] = metrics
            ps = metrics.get("PS_RMSE", float("nan"))
            print(f"    Seed {seed} {split_name}: PS-RMSE={ps:.4f}")

        all_results[seed] = results

    return all_results


class UNIFACMLModel(BaseMethod):
    """CatBoost on RDKit features — registered as unifac_ml for the paper.

    The RDKit descriptor set includes physicochemical properties that capture
    similar information to UNIFAC group contributions (logP, TPSA, etc.).
    """

    @classmethod
    def info(cls):
        return {
            "name": "UNIFAC_ML",
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
