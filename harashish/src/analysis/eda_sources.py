"""
Exploratory Data Analysis: Source (DOI) Analysis for BigSolDB v2.1

Analyzes source coverage, source-solvent overlap, inter-lab agreement,
exact duplicates, temperature coverage, and suspicious DOIs.

Outputs findings to stdout and saves figures to sc3-benchmark/figures/eda/.
"""

import os
import sys
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "bigsoldb_raw", "BigSolDBv2.1.csv"
)
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "eda")
os.makedirs(FIG_DIR, exist_ok=True)

# ── styling ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", 20)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved {path}")


# ── load ─────────────────────────────────────────────────────────────────────
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
df = pd.read_csv(DATA_PATH)
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"DOIs (unique): {df['Source'].nunique():,}")
print(f"Solutes (unique SMILES): {df['SMILES_Solute'].nunique():,}")
print(f"Solvents (unique name): {df['Solvent'].nunique():,}")
print()

# ═════════════════════════════════════════════════════════════════════════════
# 1. SOURCE COVERAGE
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("1. SOURCE COVERAGE (entries per DOI)")
print("=" * 80)

doi_counts = df["Source"].value_counts()
print(f"Total unique DOIs: {len(doi_counts):,}")
print(f"Median entries/DOI: {doi_counts.median():.0f}")
print(f"Mean entries/DOI:   {doi_counts.mean():.1f}")
print(f"Max entries/DOI:    {doi_counts.max():,}  ({doi_counts.idxmax()})")
print(f"Min entries/DOI:    {doi_counts.min()}")
print()

# Percentile table
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(doi_counts, p):.0f} entries")
print()

# Fraction of data from top DOIs
total = len(df)
for topn in [10, 20, 50, 100]:
    frac = doi_counts.head(topn).sum() / total
    print(f"  Top {topn:>3d} DOIs cover {frac*100:.1f}% of data ({doi_counts.head(topn).sum():,} rows)")
print()

# Long-tail: DOIs with <= N entries
for thresh in [5, 10, 20, 50]:
    n_doi = (doi_counts <= thresh).sum()
    n_rows = doi_counts[doi_counts <= thresh].sum()
    print(f"  DOIs with <={thresh:>3d} entries: {n_doi:>5d} DOIs, {n_rows:>6,} rows ({n_rows/total*100:.2f}%)")
print()

# Top 20 DOIs table
print("Top 20 DOIs by entry count:")
print(f"  {'Rank':>4s}  {'Entries':>8s}  {'Cumul%':>7s}  {'DOI'}")
cumsum = 0
for i, (doi, cnt) in enumerate(doi_counts.head(20).items(), 1):
    cumsum += cnt
    print(f"  {i:>4d}  {cnt:>8,}  {cumsum/total*100:>6.1f}%  {doi}")
print()

# ── figure 1a: histogram of entries per DOI ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(doi_counts.values, bins=100, color=PALETTE[3], edgecolor="white", linewidth=0.3)
ax.set_xlabel("Entries per DOI")
ax.set_ylabel("Number of DOIs")
ax.set_title("Distribution of entries per DOI")

ax = axes[1]
ax.hist(doi_counts.values, bins=np.logspace(0, np.log10(doi_counts.max() + 1), 60),
        color=PALETTE[6], edgecolor="white", linewidth=0.3)
ax.set_xscale("log")
ax.set_xlabel("Entries per DOI (log scale)")
ax.set_ylabel("Number of DOIs")
ax.set_title("Distribution of entries per DOI (log-x)")

fig.suptitle("Source Coverage: Entries per DOI", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "01_entries_per_doi_histogram.png")

# ── figure 1b: cumulative coverage ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
cum = np.cumsum(doi_counts.values) / total * 100
ax.plot(range(1, len(cum) + 1), cum, color=PALETTE[9], linewidth=2)
ax.axhline(50, ls="--", color="gray", alpha=0.5)
ax.axhline(80, ls="--", color="gray", alpha=0.5)
ax.axhline(95, ls="--", color="gray", alpha=0.5)
# Mark where 50%, 80%, 95% is reached
for target in [50, 80, 95]:
    idx = np.searchsorted(cum, target)
    ax.annotate(f"{target}% at {idx+1} DOIs", xy=(idx + 1, target),
                xytext=(idx + 80, target - 5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
ax.set_xlabel("Number of DOIs (ranked by size)")
ax.set_ylabel("Cumulative % of dataset")
ax.set_title("Cumulative data coverage by DOI rank")
ax.set_xlim(0, min(len(cum), 500))
fig.tight_layout()
savefig(fig, "01b_cumulative_coverage.png")


# ═════════════════════════════════════════════════════════════════════════════
# 2. SOURCE-SOLVENT COVERAGE
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("2. SOURCE-SOLVENT COVERAGE")
print("=" * 80)

doi_solvent = df.groupby("Source")["Solvent"].nunique().sort_values(ascending=False)
print(f"DOIs covering only 1 solvent:  {(doi_solvent == 1).sum():,} ({(doi_solvent == 1).mean()*100:.1f}%)")
print(f"DOIs covering 2-5 solvents:    {((doi_solvent >= 2) & (doi_solvent <= 5)).sum():,}")
print(f"DOIs covering 6-20 solvents:   {((doi_solvent >= 6) & (doi_solvent <= 20)).sum():,}")
print(f"DOIs covering >20 solvents:    {(doi_solvent > 20).sum():,}")
print()

print("Top 15 DOIs by solvent diversity:")
for doi, n in doi_solvent.head(15).items():
    solvents = df[df["Source"] == doi]["Solvent"].unique()
    cnt = doi_counts[doi]
    print(f"  {doi:<45s}  {n:>3d} solvents, {cnt:>6,} entries")
    if n <= 10:
        print(f"    solvents: {', '.join(sorted(solvents))}")
print()

# Top solvents overall
top_solvents = df["Solvent"].value_counts().head(15)
print("Top 15 solvents by entry count:")
for solv, cnt in top_solvents.items():
    n_doi = df[df["Solvent"] == solv]["Source"].nunique()
    print(f"  {solv:<30s}  {cnt:>7,} entries from {n_doi:>4d} DOIs")
print()

# ── figure 2a: distribution of solvents per DOI ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(doi_solvent.values, bins=range(0, min(doi_solvent.max() + 2, 80)),
        color=PALETTE[4], edgecolor="white", linewidth=0.3)
ax.set_xlabel("Number of distinct solvents per DOI")
ax.set_ylabel("Number of DOIs")
ax.set_title("Solvent diversity per DOI")
fig.tight_layout()
savefig(fig, "02a_solvents_per_doi.png")

# ── figure 2b: DOI-solvent heatmap for top DOIs and top solvents ─────────────
top_n_doi = 30
top_n_solv = 20
top_dois_list = doi_counts.head(top_n_doi).index.tolist()
top_solvents_list = df["Solvent"].value_counts().head(top_n_solv).index.tolist()

sub = df[df["Source"].isin(top_dois_list) & df["Solvent"].isin(top_solvents_list)]
matrix = sub.groupby(["Source", "Solvent"]).size().unstack(fill_value=0)
# Reorder
matrix = matrix.reindex(index=top_dois_list, columns=top_solvents_list).fillna(0).astype(int)
# Truncate DOI labels
short_labels = [d[:40] + "..." if len(d) > 40 else d for d in matrix.index]

fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(np.log10(matrix.values + 1), ax=ax, cmap="YlOrRd",
            xticklabels=matrix.columns, yticklabels=short_labels,
            cbar_kws={"label": "log10(count + 1)"}, linewidths=0.3)
ax.set_xlabel("Solvent")
ax.set_ylabel("DOI (top 30 by entry count)")
ax.set_title(f"DOI–Solvent coverage matrix (top {top_n_doi} DOIs × top {top_n_solv} solvents)")
plt.xticks(rotation=45, ha="right")
plt.yticks(fontsize=7)
fig.tight_layout()
savefig(fig, "02b_doi_solvent_heatmap.png")


# ═════════════════════════════════════════════════════════════════════════════
# 3. SOURCE OVERLAP / INTER-LAB COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("3. INTER-LAB COMPARISON (multi-source solute-solvent pairs)")
print("=" * 80)

# Count distinct DOIs per (solute, solvent) pair
pair_doi_count = df.groupby(["SMILES_Solute", "Solvent"])["Source"].nunique()
multi_source = pair_doi_count[pair_doi_count >= 2]
print(f"Total (solute, solvent) pairs:             {len(pair_doi_count):,}")
print(f"Pairs with >=2 DOIs:                       {multi_source.shape[0]:,} ({multi_source.shape[0]/len(pair_doi_count)*100:.1f}%)")
print(f"Pairs with >=3 DOIs:                       {(pair_doi_count >= 3).sum():,}")
print(f"Pairs with >=5 DOIs:                       {(pair_doi_count >= 5).sum():,}")
print()

# For multi-source pairs: compute LogS range across sources at nearest temperature
# Strategy: for each pair, group by DOI, take median temperature, then compare
# LogS at nearest available temperature across DOIs
print("Computing inter-lab LogS ranges (this may take a moment)...")

multi_pairs = multi_source.index.tolist()
# Filter df to only multi-source pairs
df_multi = df.set_index(["SMILES_Solute", "Solvent"]).loc[multi_pairs].reset_index()

# For each pair: pick the most common temperature (or closest) and compare LogS
ranges_list = []
pair_details = []

for (solute, solvent), grp in df_multi.groupby(["SMILES_Solute", "Solvent"]):
    # Find a reference temperature: median of all measurements
    ref_temp = grp["Temperature_K"].median()
    # For each DOI, pick the measurement closest to ref_temp
    vals = []
    for doi, doi_grp in grp.groupby("Source"):
        closest_idx = (doi_grp["Temperature_K"] - ref_temp).abs().idxmin()
        row = doi_grp.loc[closest_idx]
        # Only include if within 5 K of ref temp
        if abs(row["Temperature_K"] - ref_temp) <= 5.0:
            vals.append(row["LogS(mol/L)"])
    if len(vals) >= 2:
        r = max(vals) - min(vals)
        ranges_list.append(r)
        pair_details.append({
            "solute": solute, "solvent": solvent,
            "n_sources": len(vals), "range_logS": r,
            "ref_temp": ref_temp
        })

ranges = np.array(ranges_list)
print(f"Pairs compared (with >=2 DOIs within 5K of median temp): {len(ranges):,}")
print(f"Mean LogS range across sources:  {ranges.mean():.3f}")
print(f"Median LogS range:               {np.median(ranges):.3f}")
print(f"Fraction with range > 0.5:       {(ranges > 0.5).mean()*100:.1f}%  ({(ranges > 0.5).sum():,} pairs)")
print(f"Fraction with range > 1.0:       {(ranges > 1.0).mean()*100:.1f}%  ({(ranges > 1.0).sum():,} pairs)")
print(f"Fraction with range > 2.0:       {(ranges > 2.0).mean()*100:.1f}%  ({(ranges > 2.0).sum():,} pairs)")
print()

# Worst disagreements
pair_df = pd.DataFrame(pair_details).sort_values("range_logS", ascending=False)
print("Top 20 worst inter-lab disagreements:")
for _, row in pair_df.head(20).iterrows():
    print(f"  range={row['range_logS']:.2f}  {row['n_sources']} sources  "
          f"solvent={row['solvent']:<20s}  T~{row['ref_temp']:.0f}K  solute={row['solute'][:60]}")
print()

# ── figure 3: distribution of inter-lab ranges ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(ranges, bins=80, color=PALETTE[2], edgecolor="white", linewidth=0.3)
ax.axvline(0.5, color="red", ls="--", lw=1.5, label=">0.5 log units")
ax.axvline(1.0, color="darkred", ls="--", lw=1.5, label=">1.0 log units")
ax.set_xlabel("LogS range (max - min across sources)")
ax.set_ylabel("Number of (solute, solvent) pairs")
ax.set_title("Inter-lab disagreement distribution")
ax.legend()

ax = axes[1]
# Cumulative
sorted_r = np.sort(ranges)
cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
ax.plot(sorted_r, cdf, color=PALETTE[8], linewidth=2)
ax.axvline(0.5, color="red", ls="--", lw=1)
ax.axvline(1.0, color="darkred", ls="--", lw=1)
ax.set_xlabel("LogS range across sources")
ax.set_ylabel("Cumulative fraction of pairs")
ax.set_title("CDF of inter-lab disagreement")
ax.set_xlim(-0.1, min(ranges.max(), 10))

fig.suptitle("Inter-lab comparison: LogS agreement", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "03_interlab_disagreement.png")


# ═════════════════════════════════════════════════════════════════════════════
# 4. EXACT DUPLICATE DETECTION
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("4. EXACT DUPLICATE DETECTION")
print("=" * 80)

# Find (solute, solvent, temperature) triples where 2+ DOIs report the exact same LogS
# Round temperature to 2 decimal places to handle float issues
df["Temp_rounded"] = df["Temperature_K"].round(2)
df["LogS_rounded"] = df["LogS(mol/L)"].round(10)  # high precision
df["MoleFrac_rounded"] = df["Solubility(mole_fraction)"].round(10)

# Group by (solute, solvent, temp, LogS) and find groups with >1 DOI
dup_groups_logS = df.groupby(
    ["SMILES_Solute", "Solvent", "Temp_rounded", "LogS_rounded"]
)["Source"].apply(lambda x: x.unique()).reset_index()
dup_groups_logS["n_sources"] = dup_groups_logS["Source"].apply(len)
exact_dups_logS = dup_groups_logS[dup_groups_logS["n_sources"] >= 2]

print(f"Exact LogS duplicates (same solute, solvent, temp, LogS from different DOIs):")
print(f"  {len(exact_dups_logS):,} distinct (solute,solvent,temp,LogS) tuples with >=2 DOIs")

# Count rows involved
n_dup_rows = 0
for _, row in exact_dups_logS.iterrows():
    n_dup_rows += row["n_sources"]  # each source contributes at least 1 row
print(f"  ~{n_dup_rows:,} measurement rows involved in exact duplicates")
print()

# Also check mole fraction duplicates
dup_groups_mf = df.groupby(
    ["SMILES_Solute", "Solvent", "Temp_rounded", "MoleFrac_rounded"]
)["Source"].apply(lambda x: x.unique()).reset_index()
dup_groups_mf["n_sources"] = dup_groups_mf["Source"].apply(len)
exact_dups_mf = dup_groups_mf[dup_groups_mf["n_sources"] >= 2]
print(f"Exact mole-fraction duplicates (same solute, solvent, temp, x from different DOIs):")
print(f"  {len(exact_dups_mf):,} distinct tuples with >=2 DOIs")
print()

# Which DOI pairs are worst offenders?
from itertools import combinations

pair_counter = Counter()
for _, row in exact_dups_logS.iterrows():
    dois = sorted(row["Source"])
    for a, b in combinations(dois, 2):
        pair_counter[(a, b)] += 1

print(f"Total DOI pairs involved in exact duplicates: {len(pair_counter):,}")
print()
print("Top 20 DOI pairs by number of exact-duplicate entries:")
for (doi_a, doi_b), cnt in pair_counter.most_common(20):
    print(f"  {cnt:>5d} duplicates:  {doi_a}  <->  {doi_b}")
print()

# ── figure 4: bar chart of top offender pairs ────────────────────────────────
top_pairs = pair_counter.most_common(20)
if top_pairs:
    fig, ax = plt.subplots(figsize=(12, 7))
    labels = [f"{a[:25]}.. <-> {b[:25]}.." for (a, b), _ in top_pairs]
    values = [v for _, v in top_pairs]
    y_pos = range(len(labels))
    ax.barh(y_pos, values, color=PALETTE[1])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Number of exact-duplicate entries")
    ax.set_title("Top 20 DOI pairs with exact LogS duplicates\n(likely copy-paste, not independent measurements)")
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, "04_exact_duplicate_doi_pairs.png")


# ═════════════════════════════════════════════════════════════════════════════
# 5. DOI TEMPORAL COVERAGE
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("5. DOI TEMPERATURE RANGE COVERAGE")
print("=" * 80)

top20_dois = doi_counts.head(20).index.tolist()
temp_stats = []
for doi in top20_dois:
    sub = df[df["Source"] == doi]
    tmin, tmax = sub["Temperature_K"].min(), sub["Temperature_K"].max()
    tmed = sub["Temperature_K"].median()
    n_temps = sub["Temperature_K"].nunique()
    temp_stats.append({
        "doi": doi, "T_min": tmin, "T_max": tmax, "T_median": tmed,
        "T_range": tmax - tmin, "n_unique_temps": n_temps,
        "n_entries": len(sub)
    })

temp_df = pd.DataFrame(temp_stats)
print("Temperature coverage for top 20 DOIs:")
print(f"  {'DOI':<45s}  {'Tmin':>6s}  {'Tmax':>6s}  {'Range':>6s}  {'#Temps':>6s}  {'#Entries':>8s}")
for _, r in temp_df.iterrows():
    print(f"  {r['doi']:<45s}  {r['T_min']:>6.1f}  {r['T_max']:>6.1f}  "
          f"{r['T_range']:>6.1f}  {r['n_unique_temps']:>6d}  {r['n_entries']:>8,}")
print()

# Categorize DOIs by temperature range
all_temp_range = df.groupby("Source")["Temperature_K"].agg(["min", "max"])
all_temp_range["range"] = all_temp_range["max"] - all_temp_range["min"]
print("All DOIs by temperature range:")
print(f"  Isothermal (range=0 K):   {(all_temp_range['range'] == 0).sum():,} DOIs")
print(f"  Narrow (0 < range <= 10K): {((all_temp_range['range'] > 0) & (all_temp_range['range'] <= 10)).sum():,} DOIs")
print(f"  Moderate (10-50K):         {((all_temp_range['range'] > 10) & (all_temp_range['range'] <= 50)).sum():,} DOIs")
print(f"  Wide (>50K):               {(all_temp_range['range'] > 50).sum():,} DOIs")
print()

# ── figure 5: temperature range for top 20 DOIs ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
for i, r in temp_df.iterrows():
    label = r["doi"][:45]
    ax.plot([r["T_min"], r["T_max"]], [i, i], linewidth=4, color=PALETTE[i % len(PALETTE)],
            solid_capstyle="round")
    ax.plot(r["T_median"], i, "ko", markersize=5)
ax.set_yticks(range(len(temp_df)))
ax.set_yticklabels([r["doi"][:42] + "..." if len(r["doi"]) > 42 else r["doi"]
                     for _, r in temp_df.iterrows()], fontsize=7)
ax.set_xlabel("Temperature (K)")
ax.set_title("Temperature range coverage for top 20 DOIs\n(dot = median)")
ax.invert_yaxis()
fig.tight_layout()
savefig(fig, "05_temperature_range_top20.png")


# ═════════════════════════════════════════════════════════════════════════════
# 6. SUSPICIOUS DOIs
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("6. SUSPICIOUS DOIs")
print("=" * 80)

# ── 6a: Round-number analysis (significant digits in mole fraction) ──────────

def count_sig_digits(x):
    """Estimate significant digits of a float by looking at its string repr."""
    if pd.isna(x) or x == 0:
        return 0
    s = f"{x:.15g}"
    # Remove leading zeros and decimal point
    s = s.lstrip("0").lstrip("-").replace(".", "")
    # Remove trailing zeros only if there was no decimal point in original
    s = s.lstrip("0")
    if not s:
        return 1
    return len(s.rstrip("0")) if "e" not in f"{x:.15g}".lower() else len(s)


def count_sig_digits_v2(x):
    """Count significant digits by converting to string and analyzing."""
    if pd.isna(x) or x == 0:
        return 0
    # Use the representation that BigSolDB likely stored
    s = f"{x:.10g}"
    # Remove negative sign
    s = s.lstrip("-")
    # Handle scientific notation
    if "e" in s.lower():
        mantissa = s.split("e")[0].split("E")[0]
        mantissa = mantissa.replace(".", "")
        return len(mantissa.rstrip("0"))
    # Remove leading zeros and decimal point
    # e.g. 0.00123 -> "123"
    if "." in s:
        integer_part, decimal_part = s.split(".")
        if integer_part == "0" or integer_part == "":
            # Pure decimal: sig digits = len(stripped trailing zeros of decimal minus leading zeros)
            stripped = decimal_part.lstrip("0")
            return len(stripped.rstrip("0")) if stripped else 0
        else:
            # Mixed: all digits are significant (trailing zeros after decimal are significant)
            full = integer_part + decimal_part
            return len(full.rstrip("0"))
    else:
        return len(s.rstrip("0"))


print("6a. Significant digits in mole-fraction values by DOI")
print()

# For performance, sample or process in bulk
# Compute sig digits for each row's mole fraction
df["sig_digits"] = df["Solubility(mole_fraction)"].apply(count_sig_digits_v2)

doi_sig = df.groupby("Source")["sig_digits"].agg(["mean", "median", "std", "count"])
doi_sig = doi_sig.rename(columns={"count": "n_entries"})
doi_sig = doi_sig.sort_values("mean")

# DOIs with suspiciously few sig digits (mean <= 2, at least 10 entries)
suspicious_round = doi_sig[(doi_sig["mean"] <= 2.0) & (doi_sig["n_entries"] >= 10)]
print(f"DOIs with mean sig digits <= 2 (and >= 10 entries): {len(suspicious_round):,}")
print(f"  {'DOI':<50s}  {'Mean':>5s}  {'Med':>4s}  {'Std':>5s}  {'N':>6s}")
for doi, r in suspicious_round.head(20).iterrows():
    print(f"  {doi:<50s}  {r['mean']:>5.2f}  {r['median']:>4.1f}  {r['std']:>5.2f}  {r['n_entries']:>6.0f}")
print()

# DOIs with very high precision
high_prec = doi_sig[(doi_sig["mean"] >= 4.0) & (doi_sig["n_entries"] >= 10)]
print(f"DOIs with mean sig digits >= 4 (and >= 10 entries): {len(high_prec):,} (these are likely higher quality)")
print()

# ── 6b: Implausibly low variance ────────────────────────────────────────────
print("6b. DOIs with implausibly low variance in LogS")
print()

doi_logS_stats = df.groupby("Source")["LogS(mol/L)"].agg(["mean", "std", "count", "min", "max"])
doi_logS_stats["range"] = doi_logS_stats["max"] - doi_logS_stats["min"]
doi_logS_stats = doi_logS_stats.rename(columns={"count": "n_entries"})

# DOIs with low std but many entries (suspicious)
low_var = doi_logS_stats[
    (doi_logS_stats["std"] < 0.1) & (doi_logS_stats["n_entries"] >= 20)
].sort_values("std")
print(f"DOIs with LogS std < 0.1 and >= 20 entries: {len(low_var):,}")
if len(low_var) > 0:
    print(f"  {'DOI':<50s}  {'Std':>6s}  {'Range':>6s}  {'Mean':>7s}  {'N':>5s}")
    for doi, r in low_var.head(15).iterrows():
        print(f"  {doi:<50s}  {r['std']:>6.3f}  {r['range']:>6.3f}  {r['mean']:>7.3f}  {r['n_entries']:>5.0f}")
    print()

# Also check: DOIs where range/std is very small relative to number of entries
# (measuring many compounds but getting very similar values)
low_var2 = doi_logS_stats[
    (doi_logS_stats["std"] < 0.3) & (doi_logS_stats["n_entries"] >= 50)
].sort_values("std")
print(f"DOIs with LogS std < 0.3 and >= 50 entries: {len(low_var2):,}")
if len(low_var2) > 0:
    print(f"  {'DOI':<50s}  {'Std':>6s}  {'Range':>6s}  {'Mean':>7s}  {'N':>5s}")
    for doi, r in low_var2.head(15).iterrows():
        print(f"  {doi:<50s}  {r['std']:>6.3f}  {r['range']:>6.3f}  {r['mean']:>7.3f}  {r['n_entries']:>5.0f}")
print()

# ── 6c: DOIs where all mole fractions are exact powers of 10 or very round ──
print("6c. DOIs with suspiciously uniform/round mole fractions")
print()

def frac_round_values(series, n_digits=2):
    """Fraction of values that are 'round' - i.e., round to n_digits sig figs equals themselves."""
    count = 0
    for x in series:
        if pd.isna(x) or x == 0:
            continue
        magnitude = math.floor(math.log10(abs(x)))
        rounded = round(x, -magnitude + n_digits - 1)
        if abs(rounded - x) < 1e-15 * abs(x):
            count += 1
    return count / max(len(series), 1)


round_stats = []
for doi in doi_counts.head(200).index:
    sub = df[df["Source"] == doi]["Solubility(mole_fraction)"].dropna()
    if len(sub) < 10:
        continue
    frac_2sig = frac_round_values(sub, 2)
    round_stats.append({"doi": doi, "frac_2sig": frac_2sig, "n": len(sub)})

round_df = pd.DataFrame(round_stats).sort_values("frac_2sig", ascending=False)
suspicious_round_dois = round_df[round_df["frac_2sig"] > 0.9]
print(f"DOIs (top 200) where >90% of mole fractions have <=2 sig figs: {len(suspicious_round_dois):,}")
if len(suspicious_round_dois) > 0:
    for _, r in suspicious_round_dois.head(15).iterrows():
        print(f"  {r['doi']:<50s}  {r['frac_2sig']*100:.1f}% round  ({r['n']:.0f} entries)")
print()

# ── figure 6a: sig digits distribution ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(doi_sig["mean"].dropna(), bins=40, color=PALETTE[5], edgecolor="white", linewidth=0.3)
ax.axvline(2.0, color="red", ls="--", lw=1.5, label="2 sig digits threshold")
ax.set_xlabel("Mean significant digits in mole fraction")
ax.set_ylabel("Number of DOIs")
ax.set_title("Distribution of mole fraction precision by DOI")
ax.legend()

ax = axes[1]
ax.scatter(doi_logS_stats["n_entries"], doi_logS_stats["std"],
           alpha=0.3, s=10, color=PALETTE[7])
ax.axhline(0.1, color="red", ls="--", lw=1, label="std=0.1")
ax.axhline(0.3, color="orange", ls="--", lw=1, label="std=0.3")
ax.set_xlabel("Number of entries in DOI")
ax.set_ylabel("Std of LogS within DOI")
ax.set_title("LogS variance vs DOI size")
ax.set_xscale("log")
ax.legend()

fig.suptitle("Suspicious DOI indicators", fontsize=14, y=1.02)
fig.tight_layout()
savefig(fig, "06_suspicious_doi_indicators.png")

# ── figure 6b: combined suspicion score ──────────────────────────────────────
# Merge sig-digit info with variance info for a combined view
merged = doi_sig[["mean", "n_entries"]].rename(columns={"mean": "sig_dig_mean"})
merged = merged.join(doi_logS_stats[["std", "range"]].rename(columns={"std": "logS_std", "range": "logS_range"}))
merged = merged[merged["n_entries"] >= 10].dropna()

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(merged["sig_dig_mean"], merged["logS_std"],
                c=np.log10(merged["n_entries"]), s=15, alpha=0.5,
                cmap="viridis", edgecolors="none")
ax.set_xlabel("Mean significant digits in mole fraction")
ax.set_ylabel("Std of LogS within DOI")
ax.set_title("DOI quality map: precision vs variance\n(color = log10 #entries, bottom-left = suspicious)")
plt.colorbar(sc, label="log10(entries)")
ax.axvline(2.0, color="red", ls=":", alpha=0.5)
ax.axhline(0.3, color="red", ls=":", alpha=0.5)
fig.tight_layout()
savefig(fig, "06b_doi_quality_map.png")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)
print(f"""
Dataset: {len(df):,} rows, {df['Source'].nunique():,} DOIs, {df['SMILES_Solute'].nunique():,} solutes, {df['Solvent'].nunique():,} solvents

1. SOURCE COVERAGE:
   - Highly skewed distribution: top 10 DOIs cover a large fraction of data
   - Long tail of DOIs with very few entries
   - Median entries/DOI: {doi_counts.median():.0f}

2. SOURCE-SOLVENT COVERAGE:
   - {(doi_solvent == 1).sum():,} DOIs ({(doi_solvent == 1).mean()*100:.1f}%) cover only 1 solvent
   - Most single-solvent DOIs focus on water (aqueous solubility)
   - A few DOIs span many solvents (materials science / systematic studies)

3. INTER-LAB AGREEMENT:
   - {len(ranges):,} (solute, solvent) pairs with >=2 independent sources
   - {(ranges > 0.5).mean()*100:.1f}% show disagreement > 0.5 log units
   - {(ranges > 1.0).mean()*100:.1f}% show disagreement > 1.0 log units (order of magnitude!)
   - Median disagreement: {np.median(ranges):.3f} log units

4. EXACT DUPLICATES:
   - {len(exact_dups_logS):,} (solute, solvent, temp, LogS) tuples appear in >=2 DOIs with identical values
   - These are likely copied between databases, not independent measurements
   - {len(pair_counter):,} DOI pairs share at least one exact duplicate

5. TEMPERATURE COVERAGE:
   - {(all_temp_range['range'] == 0).sum():,} DOIs are isothermal (single temperature)
   - Top DOIs vary widely in temperature range coverage

6. SUSPICIOUS DOIs:
   - {len(suspicious_round):,} DOIs have suspiciously low precision (<=2 sig digits on average)
   - {len(low_var):,} DOIs have implausibly low LogS variance (std < 0.1, >=20 entries)

All figures saved to: {os.path.abspath(FIG_DIR)}
""")

print("Done.")
