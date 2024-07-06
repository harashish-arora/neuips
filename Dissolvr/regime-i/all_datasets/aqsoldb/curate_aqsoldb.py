#silence logging
import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

#imports
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

#simple curation constants
mw_cutoff = 900.0
logS_max = 0.5
logS_min = -13.5

input_csv = "aqsoldb.csv"
input_sc2 = "sc2.csv"
output_csv = "AqSolDB_curated.csv"

#optional splitting
split = True #False to skip splitting
test_size = 0.1
seed = 42
train_output = "train.csv"
test_output = "test.csv"

#helper functions
def process_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        mw = Descriptors.MolWt(mol)

        canon_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canon_smiles, mw
    except Exception:
        return None, None


if __name__ == "__main__":
    df = pd.read_csv(input_csv)

    #standardise column name
    if "Solubility" in df.columns:
        df = df.rename(columns={"Solubility": "LogS"})
    elif "S" in df.columns:
        df = df.rename(columns={"S": "LogS"})
    
    print(f"Initial entries: {len(df)}")


    df_sc2 = pd.read_csv(input_sc2)
    sc2_processed = df_sc2["SMILES"].apply(process_molecule)

    sc2_smiles_set = set(item[0] for item in sc2_processed if item[0] is not None)
    print(f"Identified {len(sc2_smiles_set)} unique molecules in SC2 to exclude.")

    #process aqsoldb
    print("Processing training molecules...")
    processed = df["SMILES"].apply(process_molecule)

    df["SMILES"], df["MW"] = zip(*processed)
    df = df.dropna(subset=["SMILES", "MW"])

    #remove salts/mixtures
    df = df[~df["SMILES"].str.contains(r"\.")]

    #apply cutoffs
    df["LogS"] = pd.to_numeric(df["LogS"], errors="coerce")
    df = df[df["MW"] <= mw_cutoff]
    df = df[(df["LogS"] >= logS_min) & (df["LogS"] <= logS_max)]

    #remove sc2 overlaps
    initial_count = len(df)
    df = df[~df["SMILES"].isin(sc2_smiles_set)]
    removed_count = initial_count - len(df)
    
    print(f"Removed {removed_count} molecules that overlapped with SC2.")
    print(f"After filtering (MW, LogS, Salts & SC2 overlap): {len(df)}")

    #aggregate dupes
    df_out = df.groupby("SMILES", as_index=False).agg({"LogS": "mean"})

    #save curated dataset
    df_out.to_csv(output_csv, index=False)
    print(f"Saved dataset to: {output_csv}")

    #optional train/test split
    if split:
        print(f"Performing Train/Test Split (Test Size: {test_size})...")
        train_df, test_df = train_test_split(
            df_out, 
            test_size=test_size, 
            random_state=seed
        )

        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        print(f"Training Set: {train_output} ({len(train_df)} molecules)")
        print(f"Test Set: {test_output} ({len(test_df)} molecules)")
