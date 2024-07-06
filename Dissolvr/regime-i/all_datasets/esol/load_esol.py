# load_esol.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# config
SEED = 101
ESOL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"

def main():
    # 1. Download Dataset
    print(f"Downloading ESOL from {ESOL_URL}...")
    try:
        df = pd.read_csv(ESOL_URL)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # 2. Rename Columns
    
    print("Renaming columns...")
    rename_map = {
        'smiles': 'SMILES',
        'measured log solubility in mols per litre': 'LogS'
    }
    
    if not set(rename_map.keys()).issubset(df.columns):
        print("Error: Expected columns not found in source CSV.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    df = df.rename(columns=rename_map)
    
    # Keep only the columns we need
    df = df[['SMILES', 'LogS']]
    print(f"Total samples: {len(df)}")

    # 3. Train/Test Split
    print(f"Splitting data (Seed={SEED}, Test Size=0.2)...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

    # 4. Save to Disk
    print(f"Saving {TRAIN_FILENAME} ({len(train_df)} rows)...")
    train_df.to_csv(TRAIN_FILENAME, index=False)
    
    print(f"Saving {TEST_FILENAME} ({len(test_df)} rows)...")
    test_df.to_csv(TEST_FILENAME, index=False)

    print()
    print("Done.")

if __name__ == "__main__":
    main()
