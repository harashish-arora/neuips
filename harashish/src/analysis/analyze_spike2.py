"""Deeper analysis of the near-zero spike.
Are these truly independent measurements or hidden copycats?"""
import pandas as pd
import numpy as np

pw = pd.read_csv('reports/phase_03_artifacts/pairwise_maes.csv')
direct = pd.read_csv('reports/phase_04_aleatoric/direct_comparisons.csv')

print('Direct comparisons columns:', direct.columns.tolist())
print('Pairwise MAE columns:', pw.columns.tolist())
print()

# Look at the source pairs with very low MAE (< 0.01)
low_pairs = pw[pw['mae'] < 0.01]
print(f'=== Source pairs with MAE < 0.01: {len(low_pairs)} ===')
print(f'  MAE < 0.001: {(pw["mae"] < 0.001).sum()}')
print(f'  MAE < 0.005: {(pw["mae"] < 0.005).sum()}')
print(f'  MAE < 0.010: {(pw["mae"] < 0.010).sum()}')

# How many unique (solute, solvent) pairs do these low-MAE source pairs cover?
if 'solute' in pw.columns and 'solvent' in pw.columns:
    low_pairs_unique = low_pairs.groupby(['solute', 'solvent']).size()
    print(f'  Covering {len(low_pairs_unique)} unique (solute, solvent) pairs')

# Now: look at the distribution AFTER removing ALL pairs with MAE < 0.02
# This is probably the right cutoff: <0.02 is "suspiciously close"
# vs pairs in the 0.02-0.05 range which are "plausibly independent but very good agreement"
print()
print('=== Mixture model perspective ===')
# The idea: the distribution is a mixture of:
#   (a) copycats: very low MAE (spike near zero)
#   (b) truly independent: broader distribution
# We need to find where (a) ends and (b) begins

# Look at the density/histogram shape
from collections import Counter
bins_fine = np.arange(0, 0.1, 0.002)
counts, edges = np.histogram(pw['mae'].values, bins=bins_fine)
for i in range(len(counts)):
    bar = '#' * (counts[i])
    print(f'  [{edges[i]:.3f}, {edges[i+1]:.3f}): {counts[i]:3d} {bar}')

# The natural break in the histogram tells us where copycats end
print()
print('=== Recommended approach ===')
# Look at where the histogram "flattens" after the initial spike
# The first few bins are much denser than the middle range
# A natural cutoff appears around 0.02-0.03

# What's the aleatoric limit if we use pair-level cutoff of 0.02?
good_pairs = pw[pw['mae'] >= 0.02]
print(f'After excluding pairs with MAE < 0.02: {len(good_pairs)} / {len(pw)} pairs remain')

# Compute the overall inter-lab variability from these remaining pairs
interlab = pd.read_csv('reports/phase_03_artifacts/interlab_variability.csv')
print(f'\nInterlab variability file: {len(interlab)} pairs')
print(f'Columns: {interlab.columns.tolist()}')
print()

# Distribution of inter-lab MAE
il_maes = interlab['mae'].values
print(f'Full inter-lab MAE: median={np.median(il_maes):.4f}, mean={np.mean(il_maes):.4f}')

bins_il = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 5.0]
for i in range(len(bins_il)-1):
    n = ((il_maes >= bins_il[i]) & (il_maes < bins_il[i+1])).sum()
    print(f'  [{bins_il[i]:.2f}, {bins_il[i+1]:.2f}): {n:4d}  ({100*n/len(il_maes):.1f}%)')

# What are good threshold candidates for Easy/Medium/Hard?
print()
print('=== Threshold candidates for Easy/Medium/Hard ===')
for thresh in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    n_below = (il_maes <= thresh).sum()
    print(f'  MAE <= {thresh:.2f}: {n_below:4d} pairs ({100*n_below/len(il_maes):.1f}%)')
