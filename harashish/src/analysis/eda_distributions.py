"""
Exploratory Data Analysis: BigSolDB v2.1 Distribution Analysis
==============================================================
Thorough investigation of solubility distributions, temperature patterns,
mole-fraction sanity, LogS consistency, outliers, and solvent data quality.

Run from the repo root with the project venv activated.
"""

import os
import sys
import warnings
import textwrap

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(BASE, "..", ".."))
DATA_DIR = os.path.join(REPO, "data", "raw", "bigsoldb_raw")
FIG_DIR = os.path.join(REPO, "figures", "eda")
os.makedirs(FIG_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "BigSolDBv2.1.csv")
COEFFS_PATH = os.path.join(DATA_DIR, "Coeffs.csv")

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── helpers ──────────────────────────────────────────────────────────────────

def section(title: str):
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def subsection(title: str):
    print(f"\n--- {title} ---")


def savefig(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ── load data ────────────────────────────────────────────────────────────────
section("Loading data")
df = pd.read_csv(CSV_PATH)
print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
print(f"\nNull counts:\n{df.isnull().sum().to_string()}")

coeffs = pd.read_csv(COEFFS_PATH)
print(f"\nDensity coefficients loaded for {len(coeffs)} solvents.")

# rename for convenience
df.rename(columns={
    "Solubility(mole_fraction)": "x",
    "Solubility(mol/L)": "molL",
    "LogS(mol/L)": "LogS",
    "Source": "Source_DOI",
}, inplace=True, errors="ignore")

# Also try the actual column name if "Source" didn't exist
if "Source_DOI" not in df.columns:
    for c in df.columns:
        if "source" in c.lower() or "doi" in c.lower():
            df.rename(columns={c: "Source_DOI"}, inplace=True)
            break

# ═══════════════════════════════════════════════════════════════════════════
# 1. LogS Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════
section("1. LogS Distribution Analysis")

logs = df["LogS"].dropna()
print(f"LogS non-null: {len(logs):,} / {len(df):,} ({100*len(logs)/len(df):.1f}%)")
print(f"  Mean:   {logs.mean():.3f}")
print(f"  Median: {logs.median():.3f}")
print(f"  Std:    {logs.std():.3f}")
print(f"  Min:    {logs.min():.3f}")
print(f"  Max:    {logs.max():.3f}")
print(f"  Skew:   {logs.skew():.3f}")
print(f"  Kurt:   {logs.kurtosis():.3f}")

# Percentiles
pcts = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
for p in pcts:
    print(f"  P{p:>5}: {np.percentile(logs, p):.3f}")

# Multimodality: dip test approximation via KDE peak counting
subsection("Multimodality check (KDE peak detection)")
from scipy.signal import find_peaks

kde_x = np.linspace(logs.min(), logs.max(), 2000)
kde = stats.gaussian_kde(logs, bw_method=0.05)
kde_y = kde(kde_x)
peaks, properties = find_peaks(kde_y, prominence=0.001)
print(f"  Number of KDE peaks (bw=0.05): {len(peaks)}")
for i, p in enumerate(peaks):
    print(f"    Peak {i+1}: LogS = {kde_x[p]:.2f}, density = {kde_y[p]:.4f}")

# Histogram + KDE
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(logs, bins=200, density=True, alpha=0.6, color="steelblue", edgecolor="none")
ax.plot(kde_x, kde_y, "r-", lw=1.5, label="KDE (bw=0.05)")
for p in peaks:
    ax.axvline(kde_x[p], color="orange", ls="--", lw=0.8)
ax.set_xlabel("LogS (mol/L)")
ax.set_ylabel("Density")
ax.set_title("LogS distribution (all data)")
ax.legend()

# Tail analysis
ax = axes[1]
ax.hist(logs, bins=200, density=True, alpha=0.6, color="steelblue", edgecolor="none",
        cumulative=True)
ax.set_xlabel("LogS (mol/L)")
ax.set_ylabel("Cumulative density")
ax.set_title("LogS CDF")
savefig(fig, "01_logs_distribution.png")

# Extreme tails
subsection("Extreme tail entries")
extreme_low = df[df["LogS"] < logs.quantile(0.001)][["SMILES_Solute", "Solvent", "LogS", "Temperature_K"]].head(10)
extreme_high = df[df["LogS"] > logs.quantile(0.999)][["SMILES_Solute", "Solvent", "LogS", "Temperature_K"]].head(10)
print("Bottom 0.1% (extremely insoluble):")
print(extreme_low.to_string(index=False))
print("\nTop 0.1% (extremely soluble):")
print(extreme_high.to_string(index=False))

# Per-solvent LogS distribution (top 10)
subsection("Per-solvent LogS distributions (top 10 solvents)")
top_solvents = df["Solvent"].value_counts().head(10)
print("Top 10 solvents by frequency:")
for s, n in top_solvents.items():
    sub = df[df["Solvent"] == s]["LogS"].dropna()
    print(f"  {s:25s}: n={n:>6,}, mean={sub.mean():>7.2f}, std={sub.std():>5.2f}, "
          f"median={sub.median():>7.2f}, range=[{sub.min():.1f}, {sub.max():.1f}]")

fig, axes = plt.subplots(2, 5, figsize=(24, 8), sharex=False)
for idx, (solvent, _) in enumerate(top_solvents.items()):
    ax = axes[idx // 5, idx % 5]
    sub = df[df["Solvent"] == solvent]["LogS"].dropna()
    ax.hist(sub, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.set_title(f"{solvent}\n(n={len(sub):,})", fontsize=10)
    ax.set_xlabel("LogS")
fig.suptitle("LogS distributions by solvent (top 10)", fontsize=14)
fig.tight_layout()
savefig(fig, "02_logs_per_solvent.png")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Temperature Distribution
# ═══════════════════════════════════════════════════════════════════════════
section("2. Temperature Distribution")

temp = df["Temperature_K"].dropna()
print(f"Temperature non-null: {len(temp):,} / {len(df):,}")
print(f"  Mean:   {temp.mean():.2f} K")
print(f"  Median: {temp.median():.2f} K")
print(f"  Std:    {temp.std():.2f} K")
print(f"  Min:    {temp.min():.2f} K")
print(f"  Max:    {temp.max():.2f} K")

# Exact 298.15 K
n_29815 = (df["Temperature_K"] == 298.15).sum()
n_near_29815 = ((df["Temperature_K"] >= 297.5) & (df["Temperature_K"] <= 298.5)).sum()
print(f"\n  Exactly 298.15 K:  {n_29815:,} ({100*n_29815/len(df):.1f}%)")
print(f"  Within 0.5 K of 298.15: {n_near_29815:,} ({100*n_near_29815/len(df):.1f}%)")

# Round-number clustering
subsection("Round-number temperature clustering")
round_temps = [273.15, 293.15, 298.15, 303.15, 308.15, 313.15, 318.15, 323.15, 333.15, 343.15, 353.15, 373.15]
print(f"  {'Temperature':>12s}  {'Count':>8s}  {'Pct':>6s}")
for t in round_temps:
    n = (df["Temperature_K"] == t).sum()
    if n > 0:
        print(f"  {t:>12.2f}  {n:>8,}  {100*n/len(df):>5.1f}%")

# Also check common round values
subsection("Most common temperature values (top 20)")
temp_counts = df["Temperature_K"].value_counts().head(20)
for t, n in temp_counts.items():
    print(f"  {t:>10.2f} K : {n:>6,} entries ({100*n/len(df):.1f}%)")

# Suspicious ranges
subsection("Suspicious temperature ranges")
below_250 = df[df["Temperature_K"] < 250]
above_400 = df[df["Temperature_K"] > 400]
above_500 = df[df["Temperature_K"] > 500]
print(f"  T < 250 K : {len(below_250):,} entries")
print(f"  T > 400 K : {len(above_400):,} entries")
print(f"  T > 500 K : {len(above_500):,} entries")

if len(below_250) > 0:
    print("\n  Samples with T < 250 K:")
    print(below_250[["SMILES_Solute", "Solvent", "Temperature_K", "LogS"]].head(10).to_string(index=False))
if len(above_400) > 0:
    print("\n  Samples with T > 400 K:")
    print(above_400[["SMILES_Solute", "Solvent", "Temperature_K", "LogS"]].head(10).to_string(index=False))

# Temperature histogram
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(temp, bins=300, color="darkorange", edgecolor="none", alpha=0.7)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Count")
ax.set_title("Temperature distribution (full range)")
ax.axvline(298.15, color="red", ls="--", lw=1, label="298.15 K")
ax.legend()

ax = axes[1]
ax.hist(temp[(temp > 270) & (temp < 370)], bins=200, color="darkorange", edgecolor="none", alpha=0.7)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Count")
ax.set_title("Temperature distribution (270-370 K)")
ax.axvline(298.15, color="red", ls="--", lw=1)

# Fractional-K analysis: how many are .15 vs round?
ax = axes[2]
temp_frac = temp % 1
ax.hist(temp_frac, bins=100, color="darkorange", edgecolor="none", alpha=0.7)
ax.set_xlabel("Fractional part of T (K)")
ax.set_ylabel("Count")
ax.set_title("Fractional K distribution\n(spike at 0.15 = Celsius origin)")

savefig(fig, "03_temperature_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Mole Fraction Sanity
# ═══════════════════════════════════════════════════════════════════════════
section("3. Mole Fraction Sanity Checks")

xf = df["x"].dropna()
print(f"Mole fraction non-null: {len(xf):,} / {len(df):,}")
print(f"  Min:    {xf.min():.2e}")
print(f"  Max:    {xf.max():.6f}")
print(f"  Mean:   {xf.mean():.6f}")
print(f"  Median: {xf.median():.6f}")

n_gt1 = (xf > 1).sum()
n_eq1 = (xf == 1).sum()
n_close1 = (xf > 0.99).sum()
n_zero = (xf == 0).sum()
n_negative = (xf < 0).sum()
n_tiny = (xf < 1e-10).sum()
n_small = (xf < 1e-6).sum()

print(f"\n  x > 1:      {n_gt1:,}")
print(f"  x == 1:     {n_eq1:,}")
print(f"  x > 0.99:   {n_close1:,}")
print(f"  x == 0:     {n_zero:,}")
print(f"  x < 0:      {n_negative:,}")
print(f"  x < 1e-10:  {n_tiny:,}")
print(f"  x < 1e-6:   {n_small:,}")

if n_gt1 > 0:
    print("\n  Entries with x > 1:")
    print(df[df["x"] > 1][["SMILES_Solute", "Solvent", "x", "LogS", "Temperature_K"]].head(10).to_string(index=False))

if n_close1 > 0:
    print(f"\n  Entries with x > 0.99 (n={n_close1}):")
    print(df[df["x"] > 0.99][["SMILES_Solute", "Solvent", "x", "LogS"]].head(10).to_string(index=False))

# log10(mole_fraction) distribution
subsection("log10(mole_fraction) distribution")
xf_pos = xf[xf > 0]
log_xf = np.log10(xf_pos)
print(f"  log10(x) stats (x>0 only, n={len(xf_pos):,}):")
print(f"    Mean:   {log_xf.mean():.3f}")
print(f"    Median: {log_xf.median():.3f}")
print(f"    Std:    {log_xf.std():.3f}")
print(f"    Min:    {log_xf.min():.3f}")
print(f"    Max:    {log_xf.max():.3f}")
print(f"    Skew:   {log_xf.skew():.3f}")
print(f"    Kurt:   {log_xf.kurtosis():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(xf[xf < 0.1], bins=200, color="teal", edgecolor="none", alpha=0.7)
ax.set_xlabel("Mole fraction")
ax.set_ylabel("Count")
ax.set_title("Mole fraction distribution (x < 0.1)")

ax = axes[1]
ax.hist(log_xf, bins=200, color="teal", edgecolor="none", alpha=0.7)
ax.set_xlabel("log10(mole fraction)")
ax.set_ylabel("Count")
ax.set_title("log10(mole fraction) distribution")

ax = axes[2]
ax.hist(xf, bins=200, color="teal", edgecolor="none", alpha=0.7)
ax.set_xlabel("Mole fraction")
ax.set_ylabel("Count")
ax.set_title("Mole fraction distribution (full range)")
ax.set_yscale("log")

savefig(fig, "04_mole_fraction_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. LogS vs Back-computed LogS Check
# ═══════════════════════════════════════════════════════════════════════════
section("4. LogS Consistency Check (pre-computed vs back-computed)")

# Build density lookup from coefficients
density_params = {}
for _, row in coeffs.iterrows():
    density_params[row["Solvent"].strip().lower()] = (row["a"], row["b"])

# For each row, we need MW of the solvent to back-compute.
# LogS = log10(x * density_solvent(T) * 1000 / MW_solvent)
# We'll try to get MW from rdkit if available, otherwise use known values.

# Known MW for common solvents
KNOWN_MW = {
    "water": 18.015, "ethanol": 46.069, "methanol": 32.042,
    "toluene": 92.141, "benzene": 78.114, "n-heptane": 100.205,
    "n-hexane": 86.178, "cyclohexane": 84.162, "n-pentane": 72.151,
    "n-octane": 114.232, "n-dodecane": 170.340, "isopropanol": 60.096,
    "n-butanol": 74.123, "n-pentanol": 88.150, "n-propanol": 60.096,
    "n-heptanol": 116.204, "n-octanol": 130.231, "n-nonanol": 144.258,
    "n-decanol": 158.284, "anisole": 108.140, "ethyl acetate": 88.106,
    "n-butyl acetate": 116.160, "n-propyl acetate": 102.133,
    "acetic acid": 60.052, "acetone": 58.080, "acetonitrile": 41.053,
    "1,4-dioxane": 88.106, "thf": 72.107, "dmf": 73.095,
    "dmso": 78.133, "chloroform": 119.378, "dichloromethane": 84.933,
    "1,2-dichloroethane": 98.959, "cyclohexanone": 98.143,
    "tetrachloromethane": 153.823, "formic acid": 46.025,
    "chlorobenzene": 112.558, "diethyl ether": 74.123,
    "dimethyl carbonate": 90.078, "gamma-butyrolactone": 86.089,
    "methyl acetate": 74.079, "2-butanone": 72.107,
}

# Try to compute MW from SMILES_Solvent using rdkit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
    print("  RDKit available -- will compute MW from SMILES_Solvent.")
except ImportError:
    HAS_RDKIT = False
    print("  RDKit not available -- using known MW table.")


def get_mw_from_smiles(smi):
    if HAS_RDKIT and pd.notna(smi):
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                return Descriptors.MolWt(mol)
        except Exception:
            pass
    return None


# Build a per-solvent MW map using SMILES if possible, else known table
solvent_smiles = df[["Solvent", "SMILES_Solvent"]].drop_duplicates().dropna(subset=["Solvent"])
solvent_mw_map = {}
for _, row in solvent_smiles.iterrows():
    solvent = row["Solvent"]
    mw = get_mw_from_smiles(row["SMILES_Solvent"])
    if mw is None:
        mw = KNOWN_MW.get(solvent.strip().lower())
    if mw is not None:
        solvent_mw_map[solvent] = mw

print(f"  MW resolved for {len(solvent_mw_map)} solvents out of {df['Solvent'].nunique()} unique.")

# Top 5 solvents with density coefficients available
top5_for_check = []
for s in df["Solvent"].value_counts().index:
    s_lower = s.strip().lower()
    if s_lower in density_params and s in solvent_mw_map:
        top5_for_check.append(s)
    if len(top5_for_check) == 5:
        break

print(f"  Top 5 solvents for LogS consistency check: {top5_for_check}")

fig, axes = plt.subplots(2, 5, figsize=(24, 9))

all_residuals = []

for idx, solvent in enumerate(top5_for_check):
    sub = df[(df["Solvent"] == solvent) & df["LogS"].notna() & df["x"].notna() & (df["x"] > 0)].copy()
    s_lower = solvent.strip().lower()
    a, b = density_params[s_lower]
    mw = solvent_mw_map[solvent]

    # density(T) = a*T + b  [g/mL]
    sub["density"] = a * sub["Temperature_K"] + b
    # back-computed LogS = log10(x * density * 1000 / MW)
    sub["LogS_back"] = np.log10(sub["x"] * sub["density"] * 1000.0 / mw)
    sub["residual"] = sub["LogS"] - sub["LogS_back"]

    all_residuals.append(sub[["Solvent", "SMILES_Solute", "Temperature_K", "LogS", "LogS_back", "residual"]])

    rmean = sub["residual"].mean()
    rstd = sub["residual"].std()
    rmedian = sub["residual"].median()

    print(f"\n  {solvent} (n={len(sub):,}, MW={mw:.1f}):")
    print(f"    Residual (precomputed - backcomputed):")
    print(f"      Mean:   {rmean:.4f}")
    print(f"      Median: {rmedian:.4f}")
    print(f"      Std:    {rstd:.4f}")
    print(f"      |res| > 1: {(sub['residual'].abs() > 1).sum()}")
    print(f"      |res| > 0.5: {(sub['residual'].abs() > 0.5).sum()}")

    # Scatter: back-computed vs precomputed
    ax = axes[0, idx]
    ax.scatter(sub["LogS_back"], sub["LogS"], alpha=0.15, s=4, color="steelblue")
    lims = [min(sub["LogS"].min(), sub["LogS_back"].min()) - 0.5,
            max(sub["LogS"].max(), sub["LogS_back"].max()) + 0.5]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Back-computed LogS")
    ax.set_ylabel("Pre-computed LogS")
    ax.set_title(f"{solvent}\n(n={len(sub):,})", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    # Residual histogram
    ax = axes[1, idx]
    ax.hist(sub["residual"], bins=100, color="salmon", edgecolor="none", alpha=0.7)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("Residual (pre - back)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residuals: mean={rmean:.3f}, std={rstd:.3f}", fontsize=9)

fig.suptitle("LogS consistency: pre-computed vs back-computed from mole fraction", fontsize=13)
fig.tight_layout()
savefig(fig, "05_logs_consistency_check.png")

residuals_df = pd.concat(all_residuals, ignore_index=True)

# Large residual entries
subsection("Largest |residual| entries across top 5 solvents")
worst = residuals_df.reindex(residuals_df["residual"].abs().sort_values(ascending=False).index).head(20)
print(worst.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 5. Outlier Candidates
# ═══════════════════════════════════════════════════════════════════════════
section("5. Outlier Candidates (deviation from per-(solute,solvent) median)")

# For each (solute, solvent) group, compute median LogS and deviation
df_with_logs = df[df["LogS"].notna()].copy()
group_stats = df_with_logs.groupby(["SMILES_Solute", "Solvent"])["LogS"].agg(["median", "count", "std"])
group_stats.columns = ["group_median", "group_count", "group_std"]

df_with_logs = df_with_logs.merge(group_stats, left_on=["SMILES_Solute", "Solvent"],
                                   right_index=True, how="left")
df_with_logs["dev_from_median"] = (df_with_logs["LogS"] - df_with_logs["group_median"]).abs()

# Only look at groups with at least 3 data points (otherwise median is meaningless)
df_outlier_candidates = df_with_logs[df_with_logs["group_count"] >= 3].copy()
df_outlier_candidates = df_outlier_candidates.sort_values("dev_from_median", ascending=False)

print(f"Entries in groups with >= 3 measurements: {len(df_outlier_candidates):,}")
print(f"\nTop 50 most extreme outliers:")
cols_show = ["SMILES_Solute", "Solvent", "Temperature_K", "LogS", "group_median",
             "dev_from_median", "group_count"]
if "Source_DOI" in df_with_logs.columns:
    cols_show.append("Source_DOI")
top50 = df_outlier_candidates.head(50)
# Truncate SMILES for display
top50_display = top50[cols_show].copy()
top50_display["SMILES_Solute"] = top50_display["SMILES_Solute"].str[:40]
if "Source_DOI" in top50_display.columns:
    top50_display["Source_DOI"] = top50_display["Source_DOI"].astype(str).str[:40]
print(top50_display.to_string(index=False))

# Source analysis of outliers
if "Source_DOI" in df_outlier_candidates.columns:
    subsection("Sources contributing to top 200 outliers")
    top200_sources = df_outlier_candidates.head(200)["Source_DOI"].value_counts().head(15)
    for src, cnt in top200_sources.items():
        print(f"  {str(src)[:60]:60s} : {cnt}")

# Distribution of deviations
subsection("Deviation statistics")
devs = df_outlier_candidates["dev_from_median"]
print(f"  Deviation > 1 LogS unit: {(devs > 1).sum():,}")
print(f"  Deviation > 2 LogS units: {(devs > 2).sum():,}")
print(f"  Deviation > 3 LogS units: {(devs > 3).sum():,}")
print(f"  Deviation > 5 LogS units: {(devs > 5).sum():,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.hist(devs[devs < 5], bins=200, color="crimson", edgecolor="none", alpha=0.7)
ax.set_xlabel("|LogS - group median|")
ax.set_ylabel("Count")
ax.set_title("Distribution of per-group deviations")
ax.axvline(1, color="black", ls="--", lw=0.8, label="|dev| = 1")
ax.axvline(2, color="gray", ls="--", lw=0.8, label="|dev| = 2")
ax.legend()

ax = axes[1]
ax.hist(devs[devs < 5], bins=200, color="crimson", edgecolor="none", alpha=0.7, cumulative=True, density=True)
ax.set_xlabel("|LogS - group median|")
ax.set_ylabel("Cumulative fraction")
ax.set_title("CDF of per-group deviations")
ax.axvline(1, color="black", ls="--", lw=0.8)

savefig(fig, "06_outlier_deviations.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Solvent Frequency vs Data Quality
# ═══════════════════════════════════════════════════════════════════════════
section("6. Solvent Frequency vs Data Quality")

solvent_stats = df[df["LogS"].notna()].groupby("Solvent").agg(
    n=("LogS", "size"),
    mean_logs=("LogS", "mean"),
    std_logs=("LogS", "std"),
    median_logs=("LogS", "median"),
    min_logs=("LogS", "min"),
    max_logs=("LogS", "max"),
    range_logs=("LogS", lambda x: x.max() - x.min()),
).reset_index()

solvent_stats = solvent_stats.sort_values("n", ascending=False)

print(f"Unique solvents with LogS data: {len(solvent_stats)}")
print(f"\nSolvent summary (top 30):")
print(solvent_stats.head(30).to_string(index=False))

# Rare solvents
rare = solvent_stats[solvent_stats["n"] < 50]
common = solvent_stats[solvent_stats["n"] >= 50]
print(f"\nRare solvents (< 50 entries): {len(rare)}")
print(f"  Mean std of LogS in rare solvents:   {rare['std_logs'].mean():.3f}")
print(f"  Median std of LogS in rare solvents: {rare['std_logs'].median():.3f}")
print(f"Common solvents (>= 50 entries): {len(common)}")
print(f"  Mean std of LogS in common solvents:   {common['std_logs'].mean():.3f}")
print(f"  Median std of LogS in common solvents: {common['std_logs'].median():.3f}")

# Statistical test
valid_rare = rare["std_logs"].dropna()
valid_common = common["std_logs"].dropna()
if len(valid_rare) > 2 and len(valid_common) > 2:
    u_stat, u_pval = stats.mannwhitneyu(valid_rare, valid_common, alternative="greater")
    print(f"\n  Mann-Whitney U test (rare std > common std): U={u_stat:.1f}, p={u_pval:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(solvent_stats["n"], solvent_stats["std_logs"], alpha=0.6, s=25, color="purple")
ax.set_xscale("log")
ax.set_xlabel("Number of entries (log scale)")
ax.set_ylabel("Std of LogS")
ax.set_title("Solvent frequency vs LogS variance")
# label top solvents
for _, row in solvent_stats.head(5).iterrows():
    ax.annotate(row["Solvent"], (row["n"], row["std_logs"]),
                fontsize=7, alpha=0.8)

ax = axes[1]
ax.scatter(solvent_stats["n"], solvent_stats["range_logs"], alpha=0.6, s=25, color="darkgreen")
ax.set_xscale("log")
ax.set_xlabel("Number of entries (log scale)")
ax.set_ylabel("Range of LogS")
ax.set_title("Solvent frequency vs LogS range")
for _, row in solvent_stats.head(5).iterrows():
    ax.annotate(row["Solvent"], (row["n"], row["range_logs"]),
                fontsize=7, alpha=0.8)

savefig(fig, "07_solvent_freq_vs_quality.png")

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
section("SUMMARY OF KEY FINDINGS")

print(textwrap.dedent("""
    See figures in: {fig_dir}

    1. LogS distribution:
       - Check printed KDE peaks for multimodality
       - Extreme tails identified above
       - Per-solvent distributions may differ significantly in shape

    2. Temperature:
       - Fraction at exactly 298.15 K reported above
       - Fractional-K analysis reveals Celsius-origin bias (.15 endings)
       - Suspicious low/high temperatures flagged

    3. Mole fraction:
       - Entries with x > 1, x = 0, x < 0 flagged as data errors
       - log10(x) distribution shape reported

    4. LogS consistency:
       - Residuals between pre-computed and back-computed LogS reported
       - Systematic offsets and large-residual entries identified

    5. Outliers:
       - Top 50 per-group outliers listed with sources
       - Source concentration in outliers may indicate quality issues

    6. Solvent quality:
       - Rare vs common solvent variance compared
       - Statistical test result reported above
""".format(fig_dir=FIG_DIR)))

print("=" * 80)
print("EDA complete.")
print("=" * 80)
