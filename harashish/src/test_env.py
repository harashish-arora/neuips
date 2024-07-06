"""Quick test to verify the conda environment works."""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd
import numpy as np
import scipy

mol = Chem.MolFromSmiles('c1ccccc1O')
print('RDKit works:', Chem.MolToSmiles(mol))
enum = rdMolStandardize.TautomerEnumerator()
canon = enum.Canonicalize(mol)
print('Tautomer canon:', Chem.MolToSmiles(canon, isomericSmiles=False))
print(f'pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}')
print('All good!')
