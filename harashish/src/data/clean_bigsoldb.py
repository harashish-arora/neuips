"""
Phase 2: Clean and standardize BigSolDB v2.1 for SC3 benchmark.

Informed by EDA (see reports/phase_02_eda_findings.md).

Performs:
  1. Remove bad DOIs (9 known problematic sources, 438 entries)
  2. Remove polymeric solvents (PEG, Span — no fixed MW, 319 entries)
  3. Remove salt/mixture solutes ("." in SMILES, 10,133 entries)
  4. SMILES canonicalization with tautomer standardization (RDKit)
  5. Use pre-computed LogS directly (verified correct x/(1-x) formula)
  6. Recompute LogS for 3,187 NaN entries using thermo library
  7. MW filter (≤ 1000 Da)
  8. Save with per-measurement granularity (DOIs preserved for Phase 3)

NOTE: No measurement averaging — Phase 3 needs individual measurements.
NOTE: No outlier removal — Phase 3's source analysis determines what to trust.

Usage:
  conda run -n sc3 python src/data/clean_bigsoldb.py \\
    --input data/raw/bigsoldb_raw/BigSolDBv2.1.csv \\
    --densities data/raw/bigsoldb_raw/BigSolDBv2.1_densities.csv \\
    --coeffs data/raw/bigsoldb_raw/Coeffs.csv \\
    --output data/intermediate/bigsoldb_cleaned.csv
"""

import os
import argparse
import warnings
import time
from functools import lru_cache
from math import log10

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# ─── Constants ────────────────────────────────────────────────────────────────

BAD_DOIS = {
    "10.1021/acs.jced.4c00179",
    "10.1021/acs.jced.9b00728",
    "10.1021/acs.jced.6b00009",
    "10.1016/j.molliq.2022.119759",
    "10.1016/j.fluid.2011.09.033",
    "10.1016/j.fluid.2013.09.018",
    "10.1016/j.molliq.2013.06.011",
    "10.1016/j.molliq.2020.113867",
    "10.1016/j.fluid.2015.07.038",
}

POLYMERIC_SOLVENTS = {
    "PEG-400", "PEG-200", "PEG-300", "PEG-600", "PEGDME 250", "span 80",
}

MW_MAX = 1000.0

# Solvent name aliases for thermo library lookup
SOLVENT_ALIASES = {
    "THF": "tetrahydrofuran",
    "DMF": "dimethylformamide",
    "DMSO": "dimethyl sulfoxide",
    "DMS": "methylthiomethane",
    "DMAC": "dimethylacetamide",
    "NMP": "1-methyl-2-pyrrolidone",
    "DEF": "diethylformamide",
    "n-heptane": "heptane",
    "n-hexane": "hexane",
    "n-pentane": "pentane",
    "n-octane": "octane",
    "n-decane": "decane",
    "n-propanol": "1-propanol",
    "n-butanol": "1-butanol",
    "n-pentanol": "1-pentanol",
    "n-hexanol": "1-hexanol",
    "n-octanol": "1-octanol",
    "2-ethyl-n-hexanol": "2-ethylhexanol",
    "sec-butanol": "2-butanol",
    "iso-butanol": "2-methyl-1-propanol",
    "isobutanol": "2-methyl-1-propanol",
    "isopropanol": "2-propanol",
    "3,6-dioxa-1-decanol": "butoxyethoxyethanol",
}


# ─── Helper Functions ─────────────────────────────────────────────────────────

def canonicalize_smiles(smiles):
    """Canonicalize SMILES: parse, standardize tautomer, output canonical (no stereo)."""
    try:
        if pd.isna(smiles) or not smiles:
            return None
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        enumerator = rdMolStandardize.TautomerEnumerator()
        canon_mol = enumerator.Canonicalize(mol)
        return Chem.MolToSmiles(canon_mol, isomericSmiles=False)
    except Exception:
        return None


def get_mol_weight(smiles):
    """Get molecular weight from canonical SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Descriptors.MolWt(mol)
    except Exception:
        return None


def build_density_lookup(densities_path, coeffs_path):
    """
    Build density lookup from BigSolDB data.
    Returns (density_dict, coeffs_dict):
      - density_dict: {(solvent_lower, T_K): density_g_cm3}
      - coeffs_dict: {solvent_lower: (a, b)} for linear model density = a*T + b
    """
    densities = pd.read_csv(densities_path)
    densities["Solvent"] = densities["Solvent"].str.strip().str.lower()
    densities["Density_g/cm^3"] = densities["Density_g/cm^3"].apply(
        lambda x: float(str(x).replace(",", "."))
    )

    density_dict = {}
    for _, row in densities.iterrows():
        key = (row["Solvent"], row["Temperature_K"])
        density_dict[key] = row["Density_g/cm^3"]

    coeffs = pd.read_csv(coeffs_path)
    coeffs_dict = {}
    for _, row in coeffs.iterrows():
        name = row["Solvent"].strip().lower()
        coeffs_dict[name] = (row["a"], row["b"])

    return density_dict, coeffs_dict


def compute_logs_correct(mole_frac, solvent_smiles, solvent_name, temperature,
                         density_dict, coeffs_dict):
    """
    Convert mole fraction to log10(S in mol/L) using the CORRECT BigSolDB formula:
      S = x/(1-x) * ρ * 1000 / MW_solvent
    This is the proper thermodynamic conversion; the simpler x * ρ/MW omits the
    1/(1-x) factor which matters for concentrated solutions.
    """
    if mole_frac <= 0 or mole_frac >= 1 or pd.isna(mole_frac):
        return np.nan

    # Get density: try exact T match first, then linear model, then thermo
    name_lower = solvent_name.strip().lower()
    density = density_dict.get((name_lower, temperature))

    if density is None:
        lookup = SOLVENT_ALIASES.get(solvent_name, solvent_name).lower()
        for try_name in [name_lower, lookup]:
            if try_name in coeffs_dict:
                a, b = coeffs_dict[try_name]
                density = a * temperature + b
                break

    if density is None:
        try:
            from thermo.chemical import Chemical
            chem_name = SOLVENT_ALIASES.get(solvent_name, solvent_name)
            m = Chemical(chem_name, T=temperature)
            if m.rho is not None and m.rho > 0:
                density = m.rho / 1000.0  # kg/m³ → g/cm³
        except Exception:
            pass

    if density is None or density <= 0:
        return np.nan

    # Get MW from solvent SMILES
    try:
        mol = Chem.MolFromSmiles(str(solvent_smiles))
        if mol is None:
            return np.nan
        mw = Descriptors.MolWt(mol)
    except Exception:
        return np.nan

    if mw <= 0:
        return np.nan

    # Correct formula: S = x/(1-x) * ρ(g/cm³) * 1000 / MW
    molarity = (mole_frac / (1 - mole_frac)) * density * 1000.0 / mw
    if molarity <= 0:
        return np.nan

    return log10(molarity)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean BigSolDB v2.1 for SC3 benchmark.")
    parser.add_argument("--input", required=True, help="Path to BigSolDBv2.1.csv")
    parser.add_argument("--densities", required=True, help="Path to densities CSV")
    parser.add_argument("--coeffs", required=True, help="Path to coefficients CSV")
    parser.add_argument("--output", required=True, help="Output cleaned CSV path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    waterfall = []

    # ── Step 1: Load ──
    print("Loading raw BigSolDB v2.1...")
    df = pd.read_csv(args.input)
    waterfall.append(("Raw data loaded", len(df)))
    print(f"  {len(df)} rows, {df['SMILES_Solute'].nunique()} unique solutes, "
          f"{df['Solvent'].nunique()} solvents, {df['Source'].nunique()} DOIs")

    # ── Step 2: Remove bad DOIs ──
    print("Removing known bad DOIs...")
    n_bad = df["Source"].isin(BAD_DOIS).sum()
    df = df[~df["Source"].isin(BAD_DOIS)].copy()
    waterfall.append(("After bad DOI removal", len(df)))
    print(f"  Removed {n_bad} entries from {len(BAD_DOIS)} DOIs")

    # ── Step 3: Remove polymeric solvents ──
    print("Removing polymeric solvents (PEG, Span)...")
    n_poly = df["Solvent"].isin(POLYMERIC_SOLVENTS).sum()
    df = df[~df["Solvent"].isin(POLYMERIC_SOLVENTS)].copy()
    waterfall.append(("After polymeric solvent removal", len(df)))
    print(f"  Removed {n_poly} entries")

    # ── Step 4: Remove salts/mixtures ──
    print("Removing salts/mixtures ('.' in solute SMILES)...")
    n_salt = df["SMILES_Solute"].str.contains(r"\.", na=False).sum()
    df = df[~df["SMILES_Solute"].str.contains(r"\.", na=False)].copy()
    waterfall.append(("After salt/mixture removal", len(df)))
    print(f"  Removed {n_salt} entries (193 unique salt forms)")

    # ── Step 5: Canonicalize SMILES ──
    print("Canonicalizing solute SMILES (tautomer standardization, no stereo)...")
    t0 = time.time()
    df["Solute_Canon"] = df["SMILES_Solute"].apply(canonicalize_smiles)
    n_fail = df["Solute_Canon"].isna().sum()
    df = df.dropna(subset=["Solute_Canon"])
    waterfall.append(("After solute canonicalization", len(df)))
    print(f"  {time.time()-t0:.1f}s. {n_fail} failed. "
          f"{df['SMILES_Solute'].nunique()} raw → {df['Solute_Canon'].nunique()} canonical")

    print("Canonicalizing solvent SMILES...")
    df["Solvent_Canon"] = df["SMILES_Solvent"].apply(canonicalize_smiles)
    n_fail_sv = df["Solvent_Canon"].isna().sum()
    df = df.dropna(subset=["Solvent_Canon"])
    waterfall.append(("After solvent canonicalization", len(df)))
    print(f"  {n_fail_sv} failed (all PEG with SMILES '-')")

    # ── Step 6: MW filter ──
    print("Computing molecular weights...")
    df["MW"] = df["Solute_Canon"].apply(get_mol_weight)
    df = df.dropna(subset=["MW"])
    n_pre = len(df)
    df = df[df["MW"] <= MW_MAX]
    waterfall.append(("After MW filter (≤1000 Da)", len(df)))
    print(f"  Removed {n_pre - len(df)} entries > {MW_MAX} Da")

    # ── Step 7: LogS — use pre-computed, fill NaN with correct formula ──
    print("Standardizing LogS values...")
    # EDA confirmed BigSolDB pre-computed LogS uses correct x/(1-x) formula
    df["LogS"] = df["LogS(mol/L)"].copy()

    nan_mask = df["LogS"].isna()
    n_nan = nan_mask.sum()
    print(f"  {n_nan} entries have NaN LogS — attempting recovery...")

    if n_nan > 0:
        density_dict, coeffs_dict = build_density_lookup(args.densities, args.coeffs)
        recovered = 0
        for idx in df[nan_mask].index:
            row = df.loc[idx]
            logs = compute_logs_correct(
                row["Solubility(mole_fraction)"],
                row["SMILES_Solvent"],
                row["Solvent"],
                row["Temperature_K"],
                density_dict,
                coeffs_dict,
            )
            if not np.isnan(logs):
                df.at[idx, "LogS"] = logs
                recovered += 1
        print(f"  Recovered {recovered} of {n_nan}. {n_nan - recovered} still NaN (dropped).")

    df = df.dropna(subset=["LogS"])
    waterfall.append(("After LogS standardization", len(df)))

    # ── Step 7b: LogS range filter ──
    print("Applying LogS range filter [-15, 2]...")
    n_pre_range = len(df)
    df = df[(df["LogS"] >= -15) & (df["LogS"] <= 2)].copy()
    waterfall.append(("After LogS range filter ([-15, 2])", len(df)))
    print(f"  Removed {n_pre_range - len(df)} entries outside [-15, 2]")

    # ── Step 8: Assemble output ──
    print("Assembling final dataset...")
    df["Solvent_Name"] = df["Solvent"]

    out_df = df[[
        "Solute_Canon", "Solvent_Canon", "Solvent_Name",
        "Temperature_K", "LogS",
        "Solubility(mole_fraction)", "MW",
        "Compound_Name", "CAS", "PubChem_CID", "FDA_Approved", "Source"
    ]].copy()
    out_df = out_df.rename(columns={
        "Solute_Canon": "Solute",
        "Solvent_Canon": "Solvent",
        "Temperature_K": "Temperature",
    })
    out_df = out_df.sort_values(
        ["Solute", "Solvent", "Temperature", "Source"]
    ).reset_index(drop=True)

    out_df.to_csv(args.output, index=False)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("CLEANING SUMMARY (waterfall)")
    print(f"{'='*70}")
    for label, count in waterfall:
        print(f"  {label:45s} {count:>8,}")

    print(f"\n  Final dataset: {len(out_df):,} rows")
    print(f"  Unique solutes:  {out_df['Solute'].nunique():,}")
    print(f"  Unique solvents: {out_df['Solvent'].nunique()} (SMILES), "
          f"{out_df['Solvent_Name'].nunique()} (names)")
    print(f"  Unique DOIs:     {out_df['Source'].nunique():,}")
    print(f"  Temperature:     {out_df['Temperature'].min():.2f} – {out_df['Temperature'].max():.2f} K")
    print(f"  LogS:            {out_df['LogS'].min():.3f} to {out_df['LogS'].max():.3f}")
    print(f"  Saved to:        {args.output}")

    # Multi-source analysis
    pair_src = out_df.groupby(["Solute", "Solvent"])["Source"].nunique()
    print(f"\n  Multi-source pairs (after canonicalization):")
    for n in [1, 2, 3, 5]:
        c = (pair_src >= n).sum()
        print(f"    ≥{n} DOIs: {c:,} pairs ({100*c/len(pair_src):.1f}%)")


if __name__ == "__main__":
    main()
