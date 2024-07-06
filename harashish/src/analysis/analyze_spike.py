"""Analyze the near-zero spike in inter-lab deviations to determine proper copycat cutoff."""
import pandas as pd
import numpy as np

# --- Pairwise source-level MAEs ---
pw = pd.read_csv('reports/phase_03_artifacts/pairwise_maes.csv')
maes = pw['mae'].values
print('=== Pairwise MAE distribution (source-pair level) ===')
print(f'Total comparisons: {len(pw)}')
bins = [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 5.0]
for i in range(len(bins)-1):
    n = ((maes >= bins[i]) & (maes < bins[i+1])).sum()
    print(f'  [{bins[i]:.3f}, {bins[i+1]:.3f}): {n:4d}  ({100*n/len(maes):.1f}%)')
print()
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f'  P{p:2d}: {np.percentile(maes, p):.4f}')

# --- Direct inter-lab deviations (per-temperature) ---
direct = pd.read_csv('reports/phase_04_aleatoric/direct_comparisons.csv')
devs = direct['abs_deviation'].values
print(f'\n=== Direct inter-lab deviations (per-temperature-point level) ===')
print(f'Total: {len(devs)}')
bins2 = [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 5.0]
for i in range(len(bins2)-1):
    n = ((devs >= bins2[i]) & (devs < bins2[i+1])).sum()
    print(f'  [{bins2[i]:.3f}, {bins2[i+1]:.3f}): {n:4d}  ({100*n/len(devs):.1f}%)')
print()
for p in [5, 10, 25, 50, 75, 90, 95]:
    print(f'  P{p:2d}: {np.percentile(devs, p):.4f}')

# --- What happens to the aleatoric limit at different cutoffs? ---
print('\n=== Aleatoric limit vs copycat cutoff (direct deviations) ===')
print(f'{"Cutoff":>8s}  {"N_remain":>8s}  {"Median":>8s}  {"Mean":>8s}  {"P90":>8s}  {"P95":>8s}')
for cutoff in [0.0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05]:
    filtered = devs[devs >= cutoff]
    if len(filtered) == 0:
        continue
    print(f'{cutoff:8.3f}  {len(filtered):8d}  {np.median(filtered):8.4f}  {np.mean(filtered):8.4f}  {np.percentile(filtered, 90):8.4f}  {np.percentile(filtered, 95):8.4f}')

# --- Same analysis but at the SOURCE-PAIR level ---
# First exclude source pairs with MAE < cutoff, then look at individual deviations
print('\n=== Aleatoric limit: exclude source PAIRS below cutoff, then compute from remaining individual deviations ===')
# We need to know which source pairs each deviation came from
if 'source1' in direct.columns and 'source2' in direct.columns:
    for cutoff in [0.0, 0.01, 0.02, 0.03, 0.05]:
        # Get source pairs above cutoff
        good_pairs = pw[pw['mae'] >= cutoff]
        good_pair_set = set()
        for _, r in good_pairs.iterrows():
            good_pair_set.add((r['source1'], r['source2']))
            good_pair_set.add((r['source2'], r['source1']))
        # Filter direct deviations to only those from good pairs
        mask = direct.apply(lambda r: (r['source1'], r['source2']) in good_pair_set, axis=1)
        filtered = devs[mask.values]
        if len(filtered) == 0:
            continue
        print(f'  Pair cutoff {cutoff:.3f}: {len(filtered):5d} deviations remain, '
              f'median={np.median(filtered):.4f}, mean={np.mean(filtered):.4f}, '
              f'P90={np.percentile(filtered, 90):.4f}, P95={np.percentile(filtered, 95):.4f}')
else:
    print('  (source columns not available in direct_comparisons.csv)')
    print(f'  Columns: {direct.columns.tolist()}')
