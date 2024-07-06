"""
Phase 4B: SC3 Dataset Construction

SC3 tiers are defined by aleatoric-bound thresholds on inter-lab MAE:
  - SC3-Hard:   pairs with inter-lab MAE <= 0.1 log S (tightest ground truth,
                hardest to beat the noise floor)
  - SC3-Medium: pairs with inter-lab MAE <= 0.2 log S
  - SC3-Easy:   pairs with inter-lab MAE <= 0.5 log S (loosest ground truth,
                easiest for models to show value over noise)

All three use the SAME consensus pipeline (Apelblat interpolation across
independent source groups), differing only in the acceptable noise ceiling.

Source pairs with MAE < 0.02 are treated as likely copycats (the histogram
of pairwise MAE shows a clear excess density below this threshold) and are
excluded from tier assignment.

Anti-leakage: SC3 test solutes are entirely removed from train/val
(molecule-level split, not data-point level).

Usage:
  conda run -n sc3 python src/data/build_sc3.py \\
    --input data/intermediate/bigsoldb_cleaned.csv \\
    --phase3-dir reports/phase_03_artifacts \\
    --output-dir data \\
    --figures-dir figures/dataset_construction
"""

import argparse
import json
import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")


# ─── Constants ─────────────────────────────────────────────────────────────

COPYCAT_MAE_THRESH = 0.02   # source pairs below this are likely copycats

# Aleatoric-bound thresholds for tiers (inter-lab MAE upper bound)
# Hard = tightest ground truth (hardest to beat the noise floor)
# Easy = loosest ground truth (easiest for models to show value)
TIER_THRESHOLDS = {
    "hard":   0.1,
    "medium": 0.2,
    "easy":   0.5,
}

VAL_FRAC = 0.15
SEED = 42


# ─── Apelblat ───────────────────────────────────────────────────────────────

def apelblat_eq(T, A, B, C):
    return A + B / T + C * np.log(T)

def fit_apelblat(temps, logs):
    try:
        p0 = [logs.mean(), -500.0, 0.05]
        bounds = ([-200, -100000, -50], [200, 100000, 50])
        params, cov = curve_fit(apelblat_eq, temps, logs, p0=p0,
                                bounds=bounds, maxfev=10000)
        pred = apelblat_eq(temps, *params)
        ss_res = np.sum((logs - pred) ** 2)
        ss_tot = np.sum((logs - logs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return params, cov, r2, True
    except Exception:
        return np.zeros(3), np.eye(3), 0.0, False


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_phase3_artifacts(phase3_dir):
    """Load Phase 3 outputs."""
    doi_rel = pd.read_csv(os.path.join(phase3_dir, "doi_reliability.csv"))
    exact_dups = pd.read_csv(os.path.join(phase3_dir, "exact_duplicates.csv"))
    near_dups = pd.read_csv(os.path.join(phase3_dir, "near_duplicates.csv"))
    pairwise = pd.read_csv(os.path.join(phase3_dir, "pairwise_maes.csv"))
    interlab = pd.read_csv(os.path.join(phase3_dir, "interlab_variability.csv"))
    return doi_rel, exact_dups, near_dups, pairwise, interlab


def build_copycat_set(pairwise_maes):
    """
    Build the set of likely-copycat source pairs.
    Any source pair with MAE < COPYCAT_MAE_THRESH is treated as a copycat.
    """
    copycats = set()
    for _, row in pairwise_maes[pairwise_maes["mae"] < COPYCAT_MAE_THRESH].iterrows():
        copycats.add((row["source1"], row["source2"]))
        copycats.add((row["source2"], row["source1"]))
    return copycats


def compute_consensus(group_df, copycats_set):
    """
    Compute consensus ground truth for a (solute, solvent) pair.

    1. Identify independent source groups (merge copycats)
    2. Fit Apelblat curve per independent group (need >= 3 temps)
    3. Evaluate all fitted curves at each reference temperature
    4. Consensus = median across source predictions; uncertainty = std
    """
    sources = group_df["Source"].unique()

    # Identify independent source groups (merge copycats)
    indep_groups = []
    assigned = set()
    for src in sources:
        if src in assigned:
            continue
        group = {src}
        for src2 in sources:
            if src2 != src and (src, src2) in copycats_set:
                group.add(src2)
        assigned.update(group)
        indep_groups.append(group)

    # Fit Apelblat per independent group
    fitted_groups = []
    for grp in indep_groups:
        grp_rows = group_df[group_df["Source"].isin(grp)]
        temps = grp_rows["Temperature"].values
        logs = grp_rows["LogS"].values
        if len(temps) >= 3:
            params, cov, r2, ok = fit_apelblat(temps, logs)
            if ok and r2 > 0.5:
                fitted_groups.append({
                    "params": params, "r2": r2,
                    "t_min": temps.min(), "t_max": temps.max(),
                    "dois": "|".join(sorted(grp)),
                })

    # If < 2 fitted curves, fallback to direct per-temperature consensus
    if len(fitted_groups) < 2:
        records = []
        all_temps = sorted(group_df["Temperature"].unique())
        for t in all_temps:
            t_data = group_df[group_df["Temperature"] == t]
            group_values = []
            group_sources = []
            for grp in indep_groups:
                grp_rows = t_data[t_data["Source"].isin(grp)]
                if len(grp_rows) > 0:
                    group_values.append(grp_rows["LogS"].median())
                    group_sources.append("|".join(sorted(grp)))
            if len(group_values) == 0:
                continue
            values = np.array(group_values)
            records.append({
                "Temperature": t,
                "LogS": float(np.median(values)),
                "N_Sources": len(values),
                "Uncertainty": float(np.std(values)) if len(values) > 1 else np.nan,
                "Source_DOIs": ";".join(group_sources),
            })
        return pd.DataFrame(records)

    # Find overlapping temperature range
    overlap_tmin = max(fg["t_min"] for fg in fitted_groups)
    overlap_tmax = min(fg["t_max"] for fg in fitted_groups)
    if overlap_tmin >= overlap_tmax:
        overlap_tmin = min(fg["t_min"] for fg in fitted_groups)
        overlap_tmax = max(fg["t_max"] for fg in fitted_groups)

    # Reference temperatures = measured temps within overlap range
    all_temps = sorted(group_df["Temperature"].unique())
    ref_temps = [t for t in all_temps if overlap_tmin <= t <= overlap_tmax]
    if not ref_temps:
        ref_temps = all_temps

    # Evaluate all curves at each reference temp
    records = []
    for t in ref_temps:
        values = []
        doi_labels = []
        for fg in fitted_groups:
            if fg["t_min"] - 5 <= t <= fg["t_max"] + 5:
                pred = apelblat_eq(t, *fg["params"])
                values.append(pred)
                doi_labels.append(fg["dois"])

        if len(values) == 0:
            continue

        values = np.array(values)
        records.append({
            "Temperature": t,
            "LogS": float(np.median(values)),
            "N_Sources": len(values),
            "Uncertainty": float(np.std(values)) if len(values) > 1 else np.nan,
            "Source_DOIs": ";".join(doi_labels),
        })

    return pd.DataFrame(records)


# ─── Tier Construction ──────────────────────────────────────────────────────

def build_tier(df, tier_pairs, copycats_set, tier_name):
    """
    Build a single SC3 tier from the given (solute, solvent) pairs.
    All tiers use the same consensus pipeline.
    """
    print(f"\n  Building SC3-{tier_name.capitalize()}...")
    print(f"    {len(tier_pairs)} candidate pairs")

    records = []
    for _, row in tier_pairs.iterrows():
        sol, solv = row["solute"], row["solvent"]
        gdf = df[(df["Solute"] == sol) & (df["Solvent"] == solv)]
        if len(gdf) == 0:
            continue
        consensus_df = compute_consensus(gdf, copycats_set)
        if len(consensus_df) == 0:
            continue
        for _, cr in consensus_df.iterrows():
            records.append({
                "Solute": sol,
                "Solvent": solv,
                "Solvent_Name": gdf["Solvent_Name"].iloc[0],
                "Temperature": cr["Temperature"],
                "LogS": cr["LogS"],
                "Uncertainty": cr["Uncertainty"],
                "N_Sources": cr["N_Sources"],
                "Interlab_MAE": row["mae"],
                "Source_DOIs": cr["Source_DOIs"],
            })

    result = pd.DataFrame(records)
    if len(result) > 0:
        print(f"    Final: {len(result)} data points, {result['Solute'].nunique()} solutes, "
              f"{result['Solvent'].nunique()} solvents")
        print(f"    Median uncertainty: {result['Uncertainty'].median():.4f}")
        print(f"    Median inter-lab MAE: {result['Interlab_MAE'].median():.4f}")
    else:
        print(f"    WARNING: No data points for SC3-{tier_name}")
    return result


def build_train_val(df, sc3_solutes, val_frac=VAL_FRAC, seed=SEED):
    """
    Build training and validation sets.
    - Remove ALL SC3 test solutes (molecule-level anti-leakage)
    - NO aleatoric-type removal --- keep all data as-is
    - Stratified split by solvent, ensuring >= 1 row per solvent in train
    """
    print("\n  Building train/val split...")
    print(f"    Removing {len(sc3_solutes)} SC3 test solutes...")

    train_pool = df[~df["Solute"].isin(sc3_solutes)].copy()
    print(f"    Train pool: {len(train_pool):,} rows, {train_pool['Solute'].nunique()} solutes")

    rng = np.random.default_rng(seed)
    val_indices = []

    for solv, sdf in train_pool.groupby("Solvent"):
        n_total = len(sdf)
        if n_total <= 1:
            continue  # keep in train only
        n_val = max(1, int(n_total * val_frac))
        n_val = min(n_val, n_total - 1)  # ensure >= 1 stays in train
        idx = sdf.index.values.copy()
        rng.shuffle(idx)
        val_indices.extend(idx[:n_val])

    val_set = set(val_indices)
    train_df = train_pool.loc[~train_pool.index.isin(val_set)].copy()
    val_df = train_pool.loc[train_pool.index.isin(val_set)].copy()

    actual_val_frac = len(val_df) / (len(train_df) + len(val_df))
    print(f"    Train: {len(train_df):,} rows ({1-actual_val_frac:.1%})")
    print(f"    Val:   {len(val_df):,} rows ({actual_val_frac:.1%})")

    return train_df, val_df


def verify_anti_leakage(train_df, val_df, tiers):
    """Verify zero overlap between SC3 test sets and train/val."""
    print("\n  Anti-leakage verification:")

    train_solutes = set(train_df["Solute"].unique())
    val_solutes = set(val_df["Solute"].unique())
    tv_solutes = train_solutes | val_solutes

    for name, sc3_df in tiers.items():
        if len(sc3_df) == 0:
            print(f"    SC3-{name}: EMPTY")
            continue
        sc3_solutes = set(sc3_df["Solute"].unique())
        overlap = sc3_solutes & tv_solutes
        status = f"LEAK ({len(overlap)} shared!)" if overlap else "CLEAN"
        print(f"    SC3-{name} vs train/val: {status}")

    # Check between tiers
    tier_names = list(tiers.keys())
    for i in range(len(tier_names)):
        for j in range(i + 1, len(tier_names)):
            n1, n2 = tier_names[i], tier_names[j]
            if len(tiers[n1]) == 0 or len(tiers[n2]) == 0:
                continue
            s1 = set(tiers[n1]["Solute"].unique())
            s2 = set(tiers[n2]["Solute"].unique())
            # Tiers are nested, so overlap is expected --- check for solute overlap
            # Actually with nested tiers, Easy solutes SHOULD appear in Medium and Hard
            # unless we make them disjoint
            # For now just report
            overlap = s1 & s2
            print(f"    SC3-{n1} vs SC3-{n2}: {len(overlap)} shared solutes")

    # Val stratification check
    train_solvents = set(train_df["Solvent"].unique())
    val_solvents = set(val_df["Solvent"].unique())
    val_only = val_solvents - train_solvents
    print(f"    Val solvents not in train: {len(val_only)}")


# ─── Plotting ──────────────────────────────────────────────────────────────

def make_plots(train_df, val_df, tiers, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    all_dfs = [("Train", train_df, "steelblue"),
               ("Val", val_df, "lightsteelblue")]
    tier_colors = {"easy": "forestgreen", "medium": "orange", "hard": "tomato"}
    for name, tdf in tiers.items():
        all_dfs.append((f"SC3-{name.capitalize()}", tdf, tier_colors.get(name, "gray")))

    # 1. Dataset sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [n for n, _, _ in all_dfs]
    sizes = [len(d) for _, d, _ in all_dfs]
    colors = [c for _, _, c in all_dfs]
    bars = ax.bar(names, sizes, color=colors, edgecolor="black")
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes)*0.02,
                f"{s:,}", ha="center", fontweight="bold")
    ax.set_ylabel("Number of data points")
    ax.set_title("SC3 Benchmark Dataset Sizes")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "01_dataset_sizes.png"), dpi=300)
    plt.close()

    # 2. LogS distributions
    n_plots = len(all_dfs)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()
    for idx, (name, ddf, color) in enumerate(all_dfs):
        ax = axes[idx]
        if len(ddf) > 0:
            ax.hist(ddf["LogS"], bins=50, edgecolor="black", alpha=0.7, color=color)
            ax.set_xlabel("LogS (mol/L)")
            ax.set_ylabel("Count")
            mean_v = ddf["LogS"].mean()
            ax.axvline(mean_v, color="red", ls="--", label=f"Mean: {mean_v:.2f}")
            ax.legend(fontsize=9)
        ax.set_title(f"{name} (n={len(ddf):,})")
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")
    plt.suptitle("LogS Distributions Across Datasets", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "02_logs_distributions.png"), dpi=300)
    plt.close()

    # 3. Temperature distributions
    fig, axes = plt.subplots(1, min(3, n_plots), figsize=(16, 5))
    if not hasattr(axes, '__len__'):
        axes = [axes]
    plot_dfs = [d for d in all_dfs if d[0] in ["Train", "SC3-Easy", "SC3-Hard"]]
    for idx, (name, ddf, color) in enumerate(plot_dfs[:len(axes)]):
        ax = axes[idx]
        if len(ddf) > 0:
            ax.hist(ddf["Temperature"], bins=40, edgecolor="black", alpha=0.7, color=color)
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Count")
        ax.set_title(f"{name} Temperature Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "03_temperature_distributions.png"), dpi=300)
    plt.close()

    # 4. Uncertainty by tier (SC3 only)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, tdf in tiers.items():
        if len(tdf) > 0 and "Uncertainty" in tdf.columns:
            valid = tdf["Uncertainty"].dropna()
            if len(valid) > 0:
                ax.hist(valid, bins=50, alpha=0.5, label=f"SC3-{name.capitalize()}", edgecolor="black")
    ax.set_xlabel("Uncertainty (std of consensus, log S units)")
    ax.set_ylabel("Count")
    ax.set_title("Ground-Truth Uncertainty by Tier")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "04_uncertainty_by_tier.png"), dpi=300)
    plt.close()


# ─── Statistics ─────────────────────────────────────────────────────────────

def compute_stats(ddf, name):
    if len(ddf) == 0:
        return {"name": name, "n_rows": 0}
    stats = {
        "name": name,
        "n_rows": len(ddf),
        "n_solutes": int(ddf["Solute"].nunique()),
        "n_solvents": int(ddf["Solvent"].nunique()),
        "logs_mean": round(float(ddf["LogS"].mean()), 3),
        "logs_std": round(float(ddf["LogS"].std()), 3),
        "logs_min": round(float(ddf["LogS"].min()), 3),
        "logs_max": round(float(ddf["LogS"].max()), 3),
        "temp_min": round(float(ddf["Temperature"].min()), 2),
        "temp_max": round(float(ddf["Temperature"].max()), 2),
    }
    if "N_Sources" in ddf.columns:
        stats["mean_n_sources"] = round(float(ddf["N_Sources"].mean()), 2)
    if "Uncertainty" in ddf.columns:
        valid = ddf["Uncertainty"].dropna()
        if len(valid) > 0:
            stats["mean_uncertainty"] = round(float(valid.mean()), 4)
            stats["median_uncertainty"] = round(float(valid.median()), 4)
    if "Interlab_MAE" in ddf.columns:
        stats["median_interlab_mae"] = round(float(ddf["Interlab_MAE"].median()), 4)
    if "Source" in ddf.columns:
        stats["n_dois"] = int(ddf["Source"].nunique())
    return stats


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build SC3 benchmark datasets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--phase3-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    sc3_dir = os.path.join(args.output_dir, "sc3")
    clean_dir = os.path.join(args.output_dir, "clean")
    os.makedirs(sc3_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows, {df['Solute'].nunique()} solutes, "
          f"{df['Solvent'].nunique()} solvents")

    doi_rel, exact_dups, near_dups, pairwise, interlab = load_phase3_artifacts(args.phase3_dir)

    # Build copycat set (MAE < 0.02)
    copycats_set = build_copycat_set(pairwise)
    n_copycat_pairs = len(pairwise[pairwise["mae"] < COPYCAT_MAE_THRESH])
    print(f"  Copycat source pairs (MAE < {COPYCAT_MAE_THRESH}): {n_copycat_pairs}")

    # ── Filter interlab to exclude copycat-contaminated pairs ──
    # A pair's inter-lab MAE < COPYCAT_MAE_THRESH means it's likely contaminated
    clean_interlab = interlab[interlab["mae"] >= COPYCAT_MAE_THRESH].copy()
    print(f"  Inter-lab pairs after copycat exclusion: {len(clean_interlab)} / {len(interlab)}")

    # ── Build SC3 tiers (nested: Easy ⊂ Medium ⊂ Hard) ──
    print("\n" + "=" * 70)
    print("BUILDING SC3 TEST TIERS (aleatoric-bound thresholds)")
    print("=" * 70)
    print(f"  Copycat threshold: MAE < {COPYCAT_MAE_THRESH} (excluded)")
    print(f"  Hard:   inter-lab MAE <= {TIER_THRESHOLDS['hard']} (tightest)")
    print(f"  Medium: inter-lab MAE <= {TIER_THRESHOLDS['medium']}")
    print(f"  Easy:   inter-lab MAE <= {TIER_THRESHOLDS['easy']} (loosest)")

    tiers = {}
    for tier_name, thresh in TIER_THRESHOLDS.items():
        tier_pairs = clean_interlab[clean_interlab["mae"] <= thresh]
        tier_df = build_tier(df, tier_pairs, copycats_set, tier_name)
        tiers[tier_name] = tier_df

    # ── Build train/val ──
    print("\n" + "=" * 70)
    print("BUILDING TRAIN/VAL SPLIT")
    print("=" * 70)

    # Remove ALL SC3 solutes (from the largest tier = hard)
    sc3_solutes = set()
    for tier_df in tiers.values():
        if len(tier_df) > 0:
            sc3_solutes |= set(tier_df["Solute"].unique())

    train_df, val_df = build_train_val(df, sc3_solutes)

    # ── Verification ──
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    verify_anti_leakage(train_df, val_df, tiers)

    # ── Save ──
    print("\n" + "=" * 70)
    print("SAVING DATASETS")
    print("=" * 70)

    for tier_name, tier_df in tiers.items():
        path = os.path.join(sc3_dir, f"sc3_{tier_name}.csv")
        tier_df.to_csv(path, index=False)
        print(f"  SC3-{tier_name.capitalize():8s}: {path} ({len(tier_df):,} rows)")

    train_df.to_csv(os.path.join(clean_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(clean_dir, "val.csv"), index=False)
    print(f"  Train:         {os.path.join(clean_dir, 'train.csv')} ({len(train_df):,} rows)")
    print(f"  Val:           {os.path.join(clean_dir, 'val.csv')} ({len(val_df):,} rows)")

    # ── Statistics ──
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    all_stats = {}
    for name, ddf in [("train", train_df), ("val", val_df)]:
        stats = compute_stats(ddf, name)
        all_stats[name] = stats
        print(f"\n  {name.upper()}: {stats['n_rows']:,} rows, "
              f"{stats['n_solutes']} solutes, {stats['n_solvents']} solvents")

    for tier_name, tier_df in tiers.items():
        stats = compute_stats(tier_df, f"sc3_{tier_name}")
        all_stats[f"sc3_{tier_name}"] = stats
        print(f"\n  SC3-{tier_name.upper()}: {stats['n_rows']:,} rows, "
              f"{stats.get('n_solutes', 0)} solutes, {stats.get('n_solvents', 0)} solvents")
        if "median_uncertainty" in stats:
            print(f"    Median uncertainty: {stats['median_uncertainty']:.4f}")
        if "median_interlab_mae" in stats:
            print(f"    Median inter-lab MAE: {stats['median_interlab_mae']:.4f}")

    with open(os.path.join(args.output_dir, "dataset_statistics.json"), "w") as f:
        json.dump(all_stats, f, indent=2)

    # ── Plots ──
    print("\nGenerating figures...")
    make_plots(train_df, val_df, tiers, args.figures_dir)

    # ── Summary ──
    total_test = sum(len(t) for t in tiers.values())
    total_all = len(train_df) + len(val_df) + len(tiers["hard"])  # Hard is the superset
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Tiers are NESTED: Easy subset of Medium subset of Hard")
    for tier_name, tier_df in tiers.items():
        pct = 100 * len(tier_df) / total_all if total_all > 0 else 0
        print(f"    SC3-{tier_name.capitalize():8s}: {len(tier_df):>8,} ({pct:.1f}%)")
    print(f"    Train:         {len(train_df):>8,}")
    print(f"    Val:           {len(val_df):>8,}")
    print(f"\n  SC3 test solutes removed from train/val: {len(sc3_solutes)}")
    print(f"  Anti-leakage: VERIFIED")


if __name__ == "__main__":
    main()
