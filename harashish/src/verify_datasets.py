"""Verification script for SC3 benchmark datasets."""
import pandas as pd
import numpy as np

# Load all datasets
easy = pd.read_csv('data/sc3/sc3_easy.csv')
med = pd.read_csv('data/sc3/sc3_medium.csv')
hard = pd.read_csv('data/sc3/sc3_hard.csv')
train = pd.read_csv('data/clean/train.csv')
val = pd.read_csv('data/clean/val.csv')
cleaned = pd.read_csv('data/intermediate/bigsoldb_cleaned.csv')

print('=== Dataset sizes ===')
for name, d in [('Cleaned', cleaned), ('Train', train), ('Val', val),
                ('SC3-Easy', easy), ('SC3-Medium', med), ('SC3-Hard', hard)]:
    print(f'  {name:12s}: {len(d):>8,} rows, {d["Solute"].nunique():>5d} solutes, '
          f'{d["Solvent"].nunique():>4d} solvents')

# Tier nesting check (Easy ⊂ Medium ⊂ Hard)
print('\n=== Tier nesting ===')
easy_pairs = set(zip(easy['Solute'], easy['Solvent']))
med_pairs = set(zip(med['Solute'], med['Solvent']))
hard_pairs = set(zip(hard['Solute'], hard['Solvent']))
print(f'  Easy pairs in Medium: {len(easy_pairs & med_pairs)} / {len(easy_pairs)}')
print(f'  Medium pairs in Hard: {len(med_pairs & hard_pairs)} / {len(med_pairs)}')
print(f'  Easy ⊂ Medium: {easy_pairs.issubset(med_pairs)}')
print(f'  Medium ⊂ Hard: {med_pairs.issubset(hard_pairs)}')

# Anti-leakage
print('\n=== Anti-leakage (molecule-level) ===')
train_sol = set(train['Solute'].unique())
val_sol = set(val['Solute'].unique())
for name, d in [('Easy', easy), ('Medium', med), ('Hard', hard)]:
    sc3_sol = set(d['Solute'].unique())
    print(f'  {name} solutes in train: {len(sc3_sol & train_sol)}')
    print(f'  {name} solutes in val:   {len(sc3_sol & val_sol)}')

# Val stratification
print('\n=== Val stratification ===')
train_solvents = set(train['Solvent'].unique())
val_solvents = set(val['Solvent'].unique())
val_only = val_solvents - train_solvents
print(f'  Solvents in train: {len(train_solvents)}')
print(f'  Solvents in val:   {len(val_solvents)}')
print(f'  Val solvents not in train: {len(val_only)}')

# LogS range
print('\n=== LogS range ===')
for name, d in [('Cleaned', cleaned), ('Train', train), ('Val', val),
                ('Easy', easy), ('Medium', med), ('Hard', hard)]:
    print(f'  {name:8s}: [{d["LogS"].min():.4f}, {d["LogS"].max():.4f}]')

# SC3 multi-source quality
print('\n=== SC3 ground-truth quality ===')
for name, d in [('Easy', easy), ('Medium', med), ('Hard', hard)]:
    multi = (d['N_Sources'] >= 2).sum()
    total = len(d)
    unc = d['Uncertainty'].dropna()
    il = d['Interlab_MAE'] if 'Interlab_MAE' in d.columns else None
    print(f'  {name}:')
    print(f'    Multi-source (N>=2): {multi}/{total} ({100*multi/total:.1f}%)')
    if len(unc) > 0:
        print(f'    Uncertainty: median={unc.median():.4f}, mean={unc.mean():.4f}')
    if il is not None:
        print(f'    Interlab MAE: median={il.median():.4f}, mean={il.mean():.4f}, max={il.max():.4f}')
    print(f'    N_Sources distribution: {d["N_Sources"].value_counts().sort_index().to_dict()}')
