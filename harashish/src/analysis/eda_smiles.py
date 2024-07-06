"""
EDA: SMILES quality analysis for BigSolDBv2.1
Analyzes canonicalization merging, failed SMILES, salts, solvent quality,
cross-identifier consistency, MW distribution, and multi-source counts.
"""

import pandas as pd
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonicalize_no_stereo(smi):
    """Canonicalize SMILES with tautomer standardization, stripping stereochemistry."""
    if pd.isna(smi) or str(smi).strip() == "":
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        # Remove stereochemistry
        Chem.RemoveStereochemistry(mol)
        # Tautomer canonicalization
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = r"C:\Users\vansh\OneDrive\Desktop\Molmerger_Anon\sc3-benchmark\data\raw\bigsoldb_raw\BigSolDBv2.1.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# =========================================================================
# 1. SOLUTE CANONICALIZATION MERGING
# =========================================================================
section("1. SOLUTE SMILES CANONICALIZATION & MERGING")

print("Canonicalizing all SMILES_Solute (tautomer-standardized, no stereo)...")
raw_solutes = df["SMILES_Solute"].values
unique_raw = set(str(s) for s in raw_solutes if pd.notna(s))
print(f"Unique raw SMILES_Solute: {len(unique_raw):,}")

# Build mapping: raw -> canonical
raw_to_canon = {}
for smi in unique_raw:
    raw_to_canon[smi] = canonicalize_no_stereo(smi)

# Apply to dataframe
df["canon_solute"] = df["SMILES_Solute"].map(lambda s: raw_to_canon.get(str(s)) if pd.notna(s) else None)

n_canon_ok = df["canon_solute"].notna().sum()
unique_canon = set(c for c in df["canon_solute"] if pd.notna(c))
print(f"Successfully canonicalized: {n_canon_ok:,} / {len(df):,} rows")
print(f"Unique canonical SMILES_Solute: {len(unique_canon):,}")
print(f"Reduction: {len(unique_raw):,} raw -> {len(unique_canon):,} canonical "
      f"({len(unique_raw) - len(unique_canon):,} merged away, "
      f"{100*(len(unique_raw)-len(unique_canon))/len(unique_raw):.1f}%)")

# Merge group distribution: group raw SMILES by their canonical form
canon_to_raws = {}
for raw, canon in raw_to_canon.items():
    if canon is not None:
        canon_to_raws.setdefault(canon, set()).add(raw)

merge_sizes = [len(v) for v in canon_to_raws.values()]
size_counts = Counter(merge_sizes)
print(f"\nMerge group size distribution (canonical -> how many raw SMILES map to it):")
for sz in sorted(size_counts.keys()):
    print(f"  {sz} raw -> 1 canonical: {size_counts[sz]:,} groups")

# Top 20 merge groups
print(f"\nTop 20 merge groups (most raw SMILES merging into one canonical):")
top_merges = sorted(canon_to_raws.items(), key=lambda x: len(x[1]), reverse=True)[:20]
for i, (canon, raws) in enumerate(top_merges, 1):
    # Find compound names for context
    names = df.loc[df["canon_solute"] == canon, "Compound_Name"].dropna().unique()
    name_str = ", ".join(names[:3])
    print(f"\n  [{i}] Canonical: {canon}")
    print(f"      {len(raws)} raw forms -> names: {name_str}")
    for r in sorted(raws)[:5]:
        print(f"        - {r}")
    if len(raws) > 5:
        print(f"        ... and {len(raws)-5} more")


# =========================================================================
# 2. FAILED SMILES (SOLUTE)
# =========================================================================
section("2. FAILED SMILES_Solute (RDKit parse failures)")

failed_mask = df["canon_solute"].isna() & df["SMILES_Solute"].notna()
n_failed = failed_mask.sum()
failed_df = df[failed_mask].copy()
print(f"Failed to parse: {n_failed:,} rows out of {df['SMILES_Solute'].notna().sum():,} non-null")

# Check if systematic by Source (DOI)
if n_failed > 0:
    source_counts = failed_df["Source"].value_counts().head(20)
    print(f"\nTop sources (DOIs) with failing SMILES:")
    for src, cnt in source_counts.items():
        total_from_src = (df["Source"] == src).sum()
        print(f"  {src}: {cnt} failures / {total_from_src} total rows ({100*cnt/total_from_src:.1f}%)")

    print(f"\nFirst 20 failing SMILES_Solute:")
    for i, smi in enumerate(failed_df["SMILES_Solute"].unique()[:20], 1):
        names = failed_df.loc[failed_df["SMILES_Solute"] == smi, "Compound_Name"].unique()
        print(f"  [{i}] {smi}")
        print(f"       Name(s): {', '.join(str(n) for n in names[:3])}")
else:
    print("No failures!")

# Also count NaN/empty SMILES_Solute
n_missing = df["SMILES_Solute"].isna().sum()
print(f"\nMissing (NaN) SMILES_Solute: {n_missing:,}")


# =========================================================================
# 3. SALT / MULTI-COMPONENT SMILES DETECTION
# =========================================================================
section("3. SALT / MULTI-COMPONENT SMILES (contain '.')")

salt_mask = df["SMILES_Solute"].str.contains(r"\.", na=False)
n_salts = salt_mask.sum()
salt_df = df[salt_mask].copy()
print(f"Rows with multi-component SMILES_Solute: {n_salts:,} ({100*n_salts/len(df):.2f}%)")

unique_salt_smi = salt_df["SMILES_Solute"].nunique()
print(f"Unique multi-component SMILES: {unique_salt_smi:,}")

# Parse out fragments and find common counterions
fragment_counter = Counter()
for smi in salt_df["SMILES_Solute"].unique():
    parts = str(smi).split(".")
    for p in parts:
        fragment_counter[p] += 1

print(f"\nMost common fragments in multi-component SMILES:")
for frag, cnt in fragment_counter.most_common(30):
    # Try to identify
    mol = Chem.MolFromSmiles(frag)
    if mol:
        mw = Descriptors.MolWt(mol)
        desc = f"MW={mw:.1f}"
    else:
        desc = "invalid"
    print(f"  {frag}: appears in {cnt} unique multi-component SMILES ({desc})")

print(f"\nExamples of multi-component SMILES (first 15):")
for i, row in salt_df.drop_duplicates("SMILES_Solute").head(15).iterrows():
    print(f"  {row['SMILES_Solute']}")
    print(f"    Name: {row['Compound_Name']}, CAS: {row['CAS']}")


# =========================================================================
# 4. SOLVENT SMILES CANONICALIZATION
# =========================================================================
section("4. SOLVENT SMILES CANONICALIZATION")

raw_solvents = set(str(s) for s in df["SMILES_Solvent"].values if pd.notna(s))
print(f"Unique raw SMILES_Solvent: {len(raw_solvents):,}")

solvent_to_canon = {}
for smi in raw_solvents:
    solvent_to_canon[smi] = canonicalize_no_stereo(smi)

df["canon_solvent"] = df["SMILES_Solvent"].map(
    lambda s: solvent_to_canon.get(str(s)) if pd.notna(s) else None
)

n_solv_ok = df["canon_solvent"].notna().sum()
unique_canon_solv = set(c for c in df["canon_solvent"] if pd.notna(c))
print(f"Successfully canonicalized: {n_solv_ok:,} / {df['SMILES_Solvent'].notna().sum():,}")
print(f"Unique canonical SMILES_Solvent: {len(unique_canon_solv):,}")
print(f"Reduction: {len(raw_solvents):,} raw -> {len(unique_canon_solv):,} canonical "
      f"({len(raw_solvents) - len(unique_canon_solv):,} merged)")

# Solvent merge groups
canon_solv_to_raws = {}
for raw, canon in solvent_to_canon.items():
    if canon is not None:
        canon_solv_to_raws.setdefault(canon, set()).add(raw)

solv_merge_sizes = [len(v) for v in canon_solv_to_raws.values()]
solv_size_counts = Counter(solv_merge_sizes)
print(f"\nSolvent merge group size distribution:")
for sz in sorted(solv_size_counts.keys()):
    print(f"  {sz} raw -> 1 canonical: {solv_size_counts[sz]:,} groups")

# Show top merges for solvents
print(f"\nTop 10 solvent merge groups:")
top_solv_merges = sorted(canon_solv_to_raws.items(), key=lambda x: len(x[1]), reverse=True)[:10]
for i, (canon, raws) in enumerate(top_solv_merges, 1):
    names = df.loc[df["canon_solvent"] == canon, "Solvent"].dropna().unique()
    print(f"  [{i}] Canonical: {canon} (names: {', '.join(str(n) for n in names[:5])})")
    for r in sorted(raws):
        print(f"        - {r}")

# Failed solvent SMILES
solv_failed_mask = df["canon_solvent"].isna() & df["SMILES_Solvent"].notna()
n_solv_failed = solv_failed_mask.sum()
print(f"\nFailed to parse SMILES_Solvent: {n_solv_failed:,} rows")
if n_solv_failed > 0:
    failed_solv = df.loc[solv_failed_mask, "SMILES_Solvent"].unique()
    print(f"Unique failing solvent SMILES: {len(failed_solv)}")
    for s in failed_solv[:20]:
        names = df.loc[df["SMILES_Solvent"] == s, "Solvent"].unique()
        print(f"  {s} -> names: {', '.join(str(n) for n in names[:3])}")

n_solv_missing = df["SMILES_Solvent"].isna().sum()
print(f"Missing (NaN) SMILES_Solvent: {n_solv_missing:,}")


# =========================================================================
# 5. CROSS-IDENTIFIER CONSISTENCY
# =========================================================================
section("5. CROSS-IDENTIFIER CONSISTENCY")

# Group by canonical solute, check CAS, PubChem_CID, Compound_Name consistency
canon_groups = df.dropna(subset=["canon_solute"]).groupby("canon_solute")

cas_inconsistent = []
cid_inconsistent = []
name_inconsistent = []

for canon, grp in canon_groups:
    cas_vals = grp["CAS"].dropna().unique()
    cid_vals = grp["PubChem_CID"].dropna().unique()
    name_vals = grp["Compound_Name"].dropna().unique()

    if len(cas_vals) > 1:
        cas_inconsistent.append((canon, list(cas_vals), len(grp)))
    if len(cid_vals) > 1:
        cid_inconsistent.append((canon, list(cid_vals), len(grp)))
    if len(name_vals) > 1:
        name_inconsistent.append((canon, list(name_vals), len(grp)))

print(f"Canonical solutes with inconsistent CAS:        {len(cas_inconsistent):,}")
print(f"Canonical solutes with inconsistent PubChem_CID: {len(cid_inconsistent):,}")
print(f"Canonical solutes with inconsistent Compound_Name: {len(name_inconsistent):,}")

print(f"\nTop 15 CAS inconsistencies (sorted by # rows):")
for canon, cas_list, nrows in sorted(cas_inconsistent, key=lambda x: -x[2])[:15]:
    names = df.loc[df["canon_solute"] == canon, "Compound_Name"].dropna().unique()
    print(f"  {canon}")
    print(f"    Names: {', '.join(str(n) for n in names[:4])}")
    print(f"    CAS values ({len(cas_list)}): {', '.join(str(c) for c in cas_list[:6])}")
    print(f"    Rows: {nrows}")

print(f"\nTop 15 PubChem_CID inconsistencies:")
for canon, cid_list, nrows in sorted(cid_inconsistent, key=lambda x: -x[2])[:15]:
    names = df.loc[df["canon_solute"] == canon, "Compound_Name"].dropna().unique()
    print(f"  {canon}")
    print(f"    Names: {', '.join(str(n) for n in names[:4])}")
    print(f"    PubChem_CIDs ({len(cid_list)}): {', '.join(str(c) for c in cid_list[:6])}")
    print(f"    Rows: {nrows}")

print(f"\nTop 15 Compound_Name inconsistencies:")
for canon, name_list, nrows in sorted(name_inconsistent, key=lambda x: -x[2])[:15]:
    cas_vals = df.loc[df["canon_solute"] == canon, "CAS"].dropna().unique()
    print(f"  {canon}")
    print(f"    Names ({len(name_list)}): {', '.join(str(n) for n in name_list[:6])}")
    print(f"    CAS: {', '.join(str(c) for c in cas_vals[:4])}")
    print(f"    Rows: {nrows}")


# =========================================================================
# 6. MOLECULAR WEIGHT DISTRIBUTION
# =========================================================================
section("6. MOLECULAR WEIGHT DISTRIBUTION")

# Compute MW for each unique canonical solute
canon_to_mw = {}
for canon in unique_canon:
    try:
        mol = Chem.MolFromSmiles(canon)
        if mol:
            canon_to_mw[canon] = Descriptors.MolWt(mol)
    except Exception:
        pass

mw_values = np.array(list(canon_to_mw.values()))
print(f"MW computed for {len(mw_values):,} / {len(unique_canon):,} canonical solutes")
print(f"\nMW Statistics:")
print(f"  Min:    {mw_values.min():.1f}")
print(f"  Q1:     {np.percentile(mw_values, 25):.1f}")
print(f"  Median: {np.median(mw_values):.1f}")
print(f"  Mean:   {mw_values.mean():.1f}")
print(f"  Q3:     {np.percentile(mw_values, 75):.1f}")
print(f"  Max:    {mw_values.max():.1f}")
print(f"  Std:    {mw_values.std():.1f}")

n_gt500 = (mw_values > 500).sum()
n_gt1000 = (mw_values > 1000).sum()
print(f"\n  MW > 500:  {n_gt500:,} ({100*n_gt500/len(mw_values):.1f}%)")
print(f"  MW > 1000: {n_gt1000:,} ({100*n_gt1000/len(mw_values):.1f}%)")

# MW distribution buckets
bins = [0, 100, 200, 300, 400, 500, 750, 1000, 2000, 5000, float("inf")]
labels = ["0-100", "100-200", "200-300", "300-400", "400-500",
          "500-750", "750-1000", "1000-2000", "2000-5000", "5000+"]
hist, _ = np.histogram(mw_values, bins=bins)
print(f"\nMW distribution:")
for label, count in zip(labels, hist):
    bar = "#" * (count // max(1, max(hist) // 50))
    print(f"  {label:>10s}: {count:>6,} {bar}")

# Top 15 largest molecules
print(f"\nTop 15 largest molecules by MW:")
sorted_mw = sorted(canon_to_mw.items(), key=lambda x: -x[1])
for i, (canon, mw) in enumerate(sorted_mw[:15], 1):
    names = df.loc[df["canon_solute"] == canon, "Compound_Name"].dropna().unique()
    name_str = ", ".join(str(n) for n in names[:2]) if len(names) > 0 else "?"
    # Truncate long SMILES
    smi_display = canon if len(canon) <= 80 else canon[:77] + "..."
    print(f"  [{i}] MW={mw:.1f}  {smi_display}")
    print(f"       Name: {name_str}")


# =========================================================================
# 7. POST-CANONICALIZATION MULTI-SOURCE COUNT
# =========================================================================
section("7. POST-CANONICALIZATION MULTI-SOURCE COUNT")

# Pre-canonicalization: use raw SMILES pairs
print("Pre-canonicalization (raw SMILES pairs):")
pre_groups = df.groupby(["SMILES_Solute", "SMILES_Solvent"])["Source"].nunique()
pre_ge2 = (pre_groups >= 2).sum()
pre_ge3 = (pre_groups >= 3).sum()
pre_ge5 = (pre_groups >= 5).sum()
print(f"  (raw_solute, raw_solvent) pairs with >= 2 distinct DOIs: {pre_ge2:,}")
print(f"  (raw_solute, raw_solvent) pairs with >= 3 distinct DOIs: {pre_ge3:,}")
print(f"  (raw_solute, raw_solvent) pairs with >= 5 distinct DOIs: {pre_ge5:,}")
print(f"  (Reference pre-canon counts: >=2: 791, >=3: 98, >=5: 4)")

# Post-canonicalization: use canonical SMILES pairs
print(f"\nPost-canonicalization (canonical SMILES pairs):")
df_canon = df.dropna(subset=["canon_solute", "canon_solvent"])
post_groups = df_canon.groupby(["canon_solute", "canon_solvent"])["Source"].nunique()
post_ge2 = (post_groups >= 2).sum()
post_ge3 = (post_groups >= 3).sum()
post_ge5 = (post_groups >= 5).sum()
print(f"  (canon_solute, canon_solvent) pairs with >= 2 distinct DOIs: {post_ge2:,}")
print(f"  (canon_solute, canon_solvent) pairs with >= 3 distinct DOIs: {post_ge3:,}")
print(f"  (canon_solute, canon_solvent) pairs with >= 5 distinct DOIs: {post_ge5:,}")

print(f"\nChange from canonicalization:")
print(f"  >= 2 DOIs: {pre_ge2:,} -> {post_ge2:,} (+{post_ge2 - pre_ge2:,})")
print(f"  >= 3 DOIs: {pre_ge3:,} -> {post_ge3:,} (+{post_ge3 - pre_ge3:,})")
print(f"  >= 5 DOIs: {pre_ge5:,} -> {post_ge5:,} (+{post_ge5 - pre_ge5:,})")

# Show examples of pairs that gained sources after canonicalization
print(f"\nTop 10 (canon_solute, canon_solvent) pairs with most distinct DOIs:")
top_pairs = post_groups.sort_values(ascending=False).head(10)
for (csol, csolv), ndoi in top_pairs.items():
    subset = df_canon[(df_canon["canon_solute"] == csol) & (df_canon["canon_solvent"] == csolv)]
    names = subset["Compound_Name"].dropna().unique()
    solvents = subset["Solvent"].dropna().unique()
    n_raw_pairs = subset.groupby(["SMILES_Solute", "SMILES_Solvent"]).ngroups
    csol_disp = csol if len(csol) <= 60 else csol[:57] + "..."
    print(f"  Solute: {csol_disp}")
    print(f"  Solvent: {csolv} ({', '.join(str(s) for s in solvents[:3])})")
    print(f"  DOIs: {ndoi}, Rows: {len(subset)}, Raw SMILES pairs: {n_raw_pairs}")
    print(f"  Names: {', '.join(str(n) for n in names[:3])}")
    print()

# =========================================================================
# SUMMARY
# =========================================================================
section("SUMMARY")
print(f"Total rows:                    {len(df):,}")
print(f"Unique raw SMILES_Solute:      {len(unique_raw):,}")
print(f"Unique canonical SMILES_Solute:{len(unique_canon):,}")
print(f"  Merged away:                 {len(unique_raw) - len(unique_canon):,} ({100*(len(unique_raw)-len(unique_canon))/len(unique_raw):.1f}%)")
print(f"  Failed to parse:             {n_failed:,} rows ({failed_df['SMILES_Solute'].nunique()} unique)")
print(f"  Multi-component (salts):     {n_salts:,} rows ({unique_salt_smi:,} unique)")
print(f"Unique raw SMILES_Solvent:     {len(raw_solvents):,}")
print(f"Unique canonical SMILES_Solvent:{len(unique_canon_solv):,}")
print(f"  Failed solvent parse:        {n_solv_failed:,} rows")
print(f"Cross-ID inconsistencies:")
print(f"  CAS conflicts:               {len(cas_inconsistent):,}")
print(f"  PubChem_CID conflicts:        {len(cid_inconsistent):,}")
print(f"  Name conflicts:              {len(name_inconsistent):,}")
print(f"MW distribution:")
print(f"  Median MW:                   {np.median(mw_values):.1f}")
print(f"  MW > 500:                    {n_gt500:,}")
print(f"  MW > 1000:                   {n_gt1000:,}")
print(f"Multi-source pairs (canon):    >= 2 DOIs: {post_ge2:,}, >= 3: {post_ge3:,}, >= 5: {post_ge5:,}")
print(f"Multi-source pairs (raw ref):  >= 2 DOIs: 791, >= 3: 98, >= 5: 4")
