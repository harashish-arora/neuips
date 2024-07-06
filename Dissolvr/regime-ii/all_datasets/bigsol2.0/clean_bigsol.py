import os
import argparse
import warnings
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from pandarallel import pandarallel
from math import log10
from functools import lru_cache
from thermo.chemical import Chemical

# configuration
DATA_DIR = "."
OUTPUT_PATH = os.path.join(DATA_DIR, "bigsol_clean.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

STD_THRESHOLD = 0.7
TOP_N_SOLVENTS = 21

# silence logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

# solvent alias map
SOLVENT_ALIASES = {
    "THF": "tetrahydrofuran",
    "n-heptane": "heptane",
    "DMS": "methylthiomethane",
    "2-ethyl-n-hexanol": "2-Ethyl hexanol",
    "3,6-dioxa-1-decanol": "butoxyethoxyethanol",
    "DEF": "diethylformamide",
    "ethanol": "ethanol",
    "methanol": "methanol"
}

# These solvents are either polymers (no fixed MW) or fail thermo lookup
EXCLUDE_SOLVENTS = {
    "ε-caprolactone", "2-methyl-cyclohexyl acetate", "diisobutyl methanol",
    "propanediol butyl ether", "span 80", "PEG-400", "PEG-200", 
    "PEGDME 250", "PEG-600", "PEG-300"
}

@lru_cache(maxsize=2048)
def get_molar_volume_factor(name, temp):
    # Cached lookup for (MW / rho) to avoid redundant thermo DB hits.
    # Returns (MW/rho) which is L/mol.
    if name in EXCLUDE_SOLVENTS:
        return None
        
    lookup_name = SOLVENT_ALIASES.get(name, name)
    try:
        m = Chemical(lookup_name, T=temp)
        if m.MW is None or m.rho is None or m.rho <= 0:
            return None
        return m.MW / m.rho
    except:
        return None

def canonicalize_smiles(smiles):
    # Solute canonicalization using TautomerEnumerator.
    try:
        if pd.isna(smiles): return None
        enumerator = rdMolStandardize.TautomerEnumerator() 
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        
        # Canonicalize the tautomer state
        canon_mol = enumerator.Canonicalize(mol)
        return Chem.MolToSmiles(canon_mol, isomericSmiles=False)
    except Exception:
        return None

def resolve_column(df, candidates):
    for col in candidates:
        if col in df.columns: return col
    return None

def convert_row_to_logs(row):
    # Priority: 1. Ground truth LogS column. 2. Back-calculation.
    # 1. If LogS is already present and valid, use it
    precalc = row.get('Precalc_LogS')
    if pd.notna(precalc):
        return float(precalc)

    # 2. Back-calculate from mole fraction
    name = str(row['Solvent_Name'])
    temp = row['T_K']
    x = row['Solubility_X']
    
    if x <= 0: return np.nan
    
    factor = get_molar_volume_factor(name, temp)
    if factor is None:
        return np.nan
    
    # Formula: log10(X / (MW/rho))
    return log10(x / factor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to BigSolDB CSV.")
    args = parser.parse_args()

    # Initialize parallel processing
    pandarallel.initialize(progress_bar=True, verbose=0)
    
    # Ensure output directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Cleaning BigSolDB...")
    df = pd.read_csv(args.input)
    
    # 1. Map Headers
    col_map = {
        'solute': resolve_column(df, ['SMILES', 'SMILES_Solute']),
        'solvent_smiles': resolve_column(df, ['SMILES_Solvent']),
        'solvent_name': resolve_column(df, ['Solvent']),
        'temp': resolve_column(df, ['T,K', 'Temperature_K']),
        'solubility': resolve_column(df, ['Solubility', 'Solubility(mole_fraction)']),
        'logs_ground_truth': resolve_column(df, ['LogS', 'LogS(mol/L)'])
    }

    # 2. Basic Cleaning & Polymer Filtering
    df = df.dropna(subset=[col_map['solute'], col_map['solvent_smiles'], col_map['solubility'], col_map['temp']])
    df = df[~df[col_map['solvent_name']].isin(EXCLUDE_SOLVENTS)]
    
    # 3. Canonicalize SMILES (Parallel)
    print("Canonicalizing Solutes (Tautomer Standard)...")
    df['Solute_Clean'] = df[col_map['solute']].parallel_apply(canonicalize_smiles)
    print("Canonicalizing Solvents...")
    df['Solvent_Clean'] = df[col_map['solvent_smiles']].parallel_apply(canonicalize_smiles)
    
    # 4. Filter for Single Molecules (Krasnov's rule)
    # We keep salts IF they are one piece, but if there's a "." it's multi-component and dropped.
    df = df[~df['Solute_Clean'].str.contains('\.', na=False)]
    
    # 5. Convert Units (Parallel + Cached)
    print("Standardizing Target (LogS in mol/L)...")
    calc_df = pd.DataFrame({
        'Solvent_Name': df[col_map['solvent_name']],
        'T_K': df[col_map['temp']],
        'Solubility_X': df[col_map['solubility']],
        'Precalc_LogS': df[col_map['logs_ground_truth']] if col_map['logs_ground_truth'] else [np.nan]*len(df)
    })
    
    df['LogS'] = calc_df.parallel_apply(convert_row_to_logs, axis=1)
    
    # Drop failures
    df = df.dropna(subset=['Solute_Clean', 'Solvent_Clean', 'LogS'])

    # 6. Inter-lab disagreement (Grouping)
    df['T_Int'] = df[col_map['temp']].round(0).astype(int)
    group_cols = ['Solute_Clean', 'Solvent_Clean', 'T_Int']
    
    print("Handling Lab Disagreements...")
    stats = df.groupby(group_cols)['LogS'].agg(['mean', 'std']).reset_index()
    stats['std'] = stats['std'].fillna(0)
    
    clean_df = stats[stats['std'] < STD_THRESHOLD].copy()
    clean_df.rename(columns={
        'Solute_Clean': 'Solute',
        'Solvent_Clean': 'Solvent',
        'T_Int': 'Temperature',
        'mean': 'LogS'
    }, inplace=True)
    
    final_cols = ['Solute', 'Solvent', 'Temperature', 'LogS']
    clean_df = clean_df[final_cols]

    # 7. Final Save
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFinal Clean Dataset: {len(clean_df)} points saved to {OUTPUT_PATH}")

    # 8. Perform Train/Test Split (Top 21 Solvents Logic)
    print("\nPerforming Train/Test Split...")
    
    # Identify Top N Solvents 
    solvent_counts = clean_df['Solvent'].value_counts()
    train_solvents_list = solvent_counts.head(TOP_N_SOLVENTS).index.tolist()
    
    print(f"Top {TOP_N_SOLVENTS} Solvents identified for Training Set.")
    
    # Create Split Masks
    train_mask = clean_df['Solvent'].isin(train_solvents_list)
    train_df = clean_df[train_mask].copy()
    test_df = clean_df[~train_mask].copy()
    
    # Save Split Files
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    
    print(f"Train Set Saved: {len(train_df)} rows ({TRAIN_PATH})")
    print(f"Test Set Saved:  {len(test_df)} rows ({TEST_PATH})")

if __name__ == "__main__":
    main()
