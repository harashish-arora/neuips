"""Compute corrected aleatoric limit and design Easy/Medium/Hard tiers."""
import pandas as pd
import numpy as np

pw = pd.read_csv('reports/phase_03_artifacts/pairwise_maes.csv')
interlab = pd.read_csv('reports/phase_03_artifacts/interlab_variability.csv')
direct = pd.read_csv('reports/phase_04_aleatoric/direct_comparisons.csv')

# ─── 1. Corrected aleatoric limit ────────────────────────────────────────────
# Remove source pairs with MAE < 0.02 (likely copycats)
COPYCAT_THRESH = 0.02
copycat_pairs = set()
for _, r in pw[pw['mae'] < COPYCAT_THRESH].iterrows():
    copycat_pairs.add((r['source1'], r['source2']))
    copycat_pairs.add((r['source2'], r['source1']))

# Filter direct comparisons to exclude copycat pairs
mask = direct.apply(lambda r: (r['source1'], r['source2']) not in copycat_pairs, axis=1)
clean_devs = direct[mask]['abs_deviation'].values

print(f'=== Corrected aleatoric limit (copycat threshold = {COPYCAT_THRESH}) ===')
print(f'Original direct comparisons: {len(direct)}')
print(f'After removing copycats: {len(clean_devs)}')
print(f'Removed: {len(direct) - len(clean_devs)} ({100*(len(direct)-len(clean_devs))/len(direct):.1f}%)')
print()
print(f'Corrected eps_direct:')
print(f'  Median: {np.median(clean_devs):.4f}')
print(f'  Mean:   {np.mean(clean_devs):.4f}')
print(f'  P25:    {np.percentile(clean_devs, 25):.4f}')
print(f'  P75:    {np.percentile(clean_devs, 75):.4f}')
print(f'  P90:    {np.percentile(clean_devs, 90):.4f}')
print(f'  P95:    {np.percentile(clean_devs, 95):.4f}')

# Bootstrap CI on corrected median
rng = np.random.default_rng(42)
medians = []
for _ in range(5000):
    sample = rng.choice(clean_devs, size=len(clean_devs), replace=True)
    medians.append(np.median(sample))
ci_lo, ci_hi = np.percentile(medians, [2.5, 97.5])
print(f'  Bootstrap 95% CI on median: [{ci_lo:.4f}, {ci_hi:.4f}]')

# Composite
delta_interp = 0.006  # from Apelblat fit CIs (unchanged)
eps_direct_med = np.median(clean_devs)
eps_aleatoric = np.sqrt(eps_direct_med**2 + 2 * delta_interp**2)
print(f'\nComposite eps_aleatoric = sqrt({eps_direct_med:.4f}^2 + 2*{delta_interp}^2) = {eps_aleatoric:.4f}')

# ─── 2. Inter-lab MAE after cleaning ────────────────────────────────────────
# Also compute per-pair inter-lab MAE excluding copycat source pairs
# This is the interlab variability we use for Easy/Medium/Hard thresholds
print(f'\n=== Inter-lab variability (pair-level) ===')
il_maes = interlab['mae'].values
# But we should also recompute excluding copycat pairs within each (solute, solvent)
# For now, use the existing interlab file which already merges exact+near dups (MAE<0.01)
# We need to also exclude pairs in [0.01, 0.02) from the interlab calculation
# But interlab.csv reports per (solute, solvent) MAE, not per source pair
# So the per-pair MAE already averages across source pairs — some may be copycats

# Let's look at what the tiers would be
print(f'Total pairs with inter-lab data: {len(interlab)}')
print()

# For Easy/Medium/Hard:
# We want different aleatoric-bound tiers
# Easy: pairs where inter-lab MAE is very low (tight ground truth)
# Medium: pairs where inter-lab MAE is moderate
# Hard: pairs where inter-lab MAE is higher (noisier ground truth)

# Candidate thresholds
print('=== Tier design (aleatoric bound thresholds) ===')
for label, lo, hi in [('Easy', 0.0, 0.1), ('Medium', 0.0, 0.2), ('Hard', 0.0, 0.5)]:
    mask = (il_maes >= lo) & (il_maes <= hi)
    n = mask.sum()
    sub = il_maes[mask]
    print(f'{label:8s} (MAE <= {hi}): {n:4d} pairs, median MAE = {np.median(sub):.4f}')

# But we should exclude the <0.02 likely-copycat pairs too
print()
print('=== Tier design (with copycat exclusion >= 0.02) ===')
clean_il = interlab[interlab['mae'] >= COPYCAT_THRESH]
clean_il_maes = clean_il['mae'].values
print(f'After excluding likely-copycat pairs (MAE < {COPYCAT_THRESH}): {len(clean_il)} pairs')
for label, hi in [('Easy', 0.1), ('Medium', 0.2), ('Hard', 0.5)]:
    mask = clean_il_maes <= hi
    n = mask.sum()
    if n > 0:
        sub = clean_il_maes[mask]
        print(f'{label:8s} (MAE <= {hi}): {n:4d} pairs, median MAE = {np.median(sub):.4f}, mean = {np.mean(sub):.4f}')

# Alternative: keep all pairs but just use different thresholds
# Easy: tightest ground truth (MAE ≤ 0.1 after cleaning)
# Medium: moderate (MAE ≤ 0.2)
# Hard: loose (MAE ≤ 0.5)
# Each tier INCLUDES the pairs from easier tiers (nested)
print()
print('=== Alternative: nested tiers (each includes easier pairs) ===')
for label, hi in [('Easy', 0.1), ('Medium', 0.2), ('Hard', 0.5)]:
    # Exclude copycat pairs (MAE < 0.02) from ALL tiers
    mask = (il_maes >= COPYCAT_THRESH) & (il_maes <= hi)
    n = mask.sum()
    sub = il_maes[mask]
    if n > 0:
        print(f'{label:8s} (0.02 <= MAE <= {hi}): {n:4d} pairs, median = {np.median(sub):.4f}')

print()
print('=== Or: disjoint tiers ===')
for label, lo, hi in [('Easy', 0.02, 0.1), ('Medium', 0.1, 0.2), ('Hard', 0.2, 0.5)]:
    mask = (il_maes >= lo) & (il_maes < hi)
    n = mask.sum()
    if n > 0:
        sub = il_maes[mask]
        print(f'{label:8s} ({lo:.2f} <= MAE < {hi:.2f}): {n:4d} pairs, median = {np.median(sub):.4f}')

# How many data points would each tier have?
print()
print('=== Estimated data points per tier ===')
df = pd.read_csv('data/intermediate/bigsoldb_cleaned.csv')
for label, hi in [('Easy (<=0.1)', 0.1), ('Medium (<=0.2)', 0.2), ('Hard (<=0.5)', 0.5)]:
    tier_pairs = interlab[(interlab['mae'] >= COPYCAT_THRESH) & (interlab['mae'] <= hi)]
    total_pts = 0
    for _, r in tier_pairs.iterrows():
        n = len(df[(df['Solute'] == r['solute']) & (df['Solvent'] == r['solvent'])])
        total_pts += n
    print(f'{label:20s}: {len(tier_pairs):4d} pairs, ~{total_pts:6d} raw data points')
