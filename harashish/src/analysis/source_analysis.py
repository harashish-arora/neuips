"""
Phase 3: Source Analysis and Inter-Lab Variability

This is the scientific heart of the SC3 benchmark paper. Analyzes:
  1. Grouping by (solute, solvent, source) triples
  2. Copycat detection — exact duplicates and near-duplicates across DOIs
  3. Apelblat / van't Hoff curve fitting per source
  4. True inter-lab variability at reference temperatures
  5. Source (DOI) reliability ranking
  6. Stratified aleatoric limit estimation

Usage:
  conda run -n sc3 python src/analysis/source_analysis.py \
    --input data/intermediate/bigsoldb_cleaned.csv \
    --output-dir reports/phase_03_artifacts \
    --figures-dir figures/source_analysis
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from itertools import combinations
from math import log10

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ─── Constants ───────────────────────────────────────────────────────────────

EXACT_DUP_DECIMALS = 3        # round to N decimals for exact-duplicate check
NEAR_DUP_MAE_THRESH = 0.01    # log S units — suspected duplicate if MAE below this
APELBLAT_MIN_POINTS = 3       # min temps for Apelblat fit
VANTHOFF_POINTS = 2            # exactly 2 temps → van't Hoff
CI_WIDTH_THRESH = 0.3          # max 95% CI half-width for reliable interpolation
TEMP_GAP_MIN = 2.0             # K — min gap between actual measurements for independence
SHAME_THRESH = 0.6             # log S MAE from consensus → Hall of Shame
FAME_MAX_DEV = 0.2             # log S MAE from consensus → Hall of Fame


# ─── Apelblat / van't Hoff fitting ──────────────────────────────────────────

def apelblat_eq(T, A, B, C):
    """ln S = A + B/T + C*ln(T)  (but we work in log10 S)."""
    return A + B / T + C * np.log(T)


def vanthoff_eq(T, A, B):
    """log S = A + B/T  (2-parameter)."""
    return A + B / T


def fit_apelblat(temps, logs):
    """Fit 3-parameter Apelblat. Returns (params, cov, r2, success)."""
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


def fit_vanthoff(temps, logs):
    """Fit 2-parameter van't Hoff (exact with 2 points). Returns (params, cov, r2, success)."""
    try:
        p0 = [logs.mean(), -500.0]
        params, cov = curve_fit(vanthoff_eq, temps, logs, p0=p0, maxfev=5000)
        pred = vanthoff_eq(temps, *params)
        ss_res = np.sum((logs - pred) ** 2)
        ss_tot = np.sum((logs - logs.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0  # exact with 2 pts
        return params, cov, r2, True
    except Exception:
        return np.zeros(2), np.eye(2), 0.0, False


def predict_with_ci(params, cov, T_ref, model="apelblat"):
    """Predict log S and 95% CI at T_ref using fitted params + covariance."""
    if model == "apelblat":
        pred = apelblat_eq(T_ref, *params)
        J = np.array([1.0, 1.0 / T_ref, np.log(T_ref)])
    else:
        pred = vanthoff_eq(T_ref, *params)
        J = np.array([1.0, 1.0 / T_ref])
    var = J @ cov @ J
    ci_half = 1.96 * np.sqrt(max(var, 0))
    return pred, ci_half


# ─── Step 3.1: Group by Solute-Solvent-Source ────────────────────────────────

def group_data(df):
    """Group cleaned data by (solute, solvent) pairs and (solute, solvent, source) triples."""
    pair_groups = {}
    triple_groups = {}

    for (sol, solv), gdf in df.groupby(["Solute", "Solvent"]):
        pair_groups[(sol, solv)] = gdf
        for src, sdf in gdf.groupby("Source"):
            triple_groups[(sol, solv, src)] = sdf.sort_values("Temperature")

    return pair_groups, triple_groups


def pair_source_stats(pair_groups):
    """Compute how many (solute, solvent) pairs have N independent sources."""
    src_counts = {}
    for (sol, solv), gdf in pair_groups.items():
        n = gdf["Source"].nunique()
        src_counts[(sol, solv)] = n

    total = len(src_counts)
    stats = {}
    for n in [1, 2, 3, 4, 5, 10]:
        c = sum(1 for v in src_counts.values() if v >= n)
        stats[f"ge{n}"] = c
        stats[f"ge{n}_pct"] = round(100 * c / total, 2) if total > 0 else 0
    stats["total_pairs"] = total
    return stats, src_counts


# ─── Step 3.2: Copycat Detection ────────────────────────────────────────────

def detect_copycats(pair_groups):
    """
    For each (solute, solvent) pair with ≥2 sources, check for:
      a) Exact duplicates (values identical to N decimal places at identical T)
      b) Near-duplicates (pairwise MAE < threshold)
    Returns lists of flagged pairs.
    """
    exact_dups = []
    near_dups = []
    pairwise_maes = []

    multi_source_pairs = {k: v for k, v in pair_groups.items()
                          if v["Source"].nunique() >= 2}

    for (sol, solv), gdf in multi_source_pairs.items():
        sources = gdf["Source"].unique()
        source_data = {}
        for src in sources:
            sdf = gdf[gdf["Source"] == src].sort_values("Temperature")
            source_data[src] = sdf

        for s1, s2 in combinations(sources, 2):
            d1 = source_data[s1]
            d2 = source_data[s2]

            # Find matching temperatures
            t1 = d1["Temperature"].values
            t2 = d2["Temperature"].values
            l1 = d1["LogS"].values
            l2 = d2["LogS"].values

            shared_temps = np.intersect1d(np.round(t1, 2), np.round(t2, 2))
            if len(shared_temps) == 0:
                continue

            vals1, vals2 = [], []
            for t in shared_temps:
                v1 = l1[np.argmin(np.abs(t1 - t))]
                v2 = l2[np.argmin(np.abs(t2 - t))]
                vals1.append(v1)
                vals2.append(v2)

            vals1 = np.array(vals1)
            vals2 = np.array(vals2)
            mae = np.mean(np.abs(vals1 - vals2))

            pairwise_maes.append({
                "solute": sol, "solvent": solv,
                "source1": s1, "source2": s2,
                "n_shared_temps": len(shared_temps),
                "mae": mae,
            })

            # Exact duplicate: all values match to N decimals
            r1 = np.round(vals1, EXACT_DUP_DECIMALS)
            r2 = np.round(vals2, EXACT_DUP_DECIMALS)
            if np.all(r1 == r2):
                exact_dups.append({
                    "solute": sol, "solvent": solv,
                    "source1": s1, "source2": s2,
                    "n_shared_temps": len(shared_temps),
                    "mae": mae,
                })

            # Near-duplicate
            if mae < NEAR_DUP_MAE_THRESH and not np.all(r1 == r2):
                near_dups.append({
                    "solute": sol, "solvent": solv,
                    "source1": s1, "source2": s2,
                    "n_shared_temps": len(shared_temps),
                    "mae": mae,
                })

    return exact_dups, near_dups, pairwise_maes


# ─── Step 3.3: Fit Curves per Source ─────────────────────────────────────────

def fit_all_triples(triple_groups):
    """
    For each (solute, solvent, source) triple, fit:
      - Apelblat (≥3 temps)
      - van't Hoff (exactly 2 temps)
      - Isolated (1 temp — no fit)
    Returns dict of fit results.
    """
    fits = {}
    stats = {"apelblat": 0, "vanthoff": 0, "isolated": 0, "failed": 0}

    for (sol, solv, src), sdf in triple_groups.items():
        temps = sdf["Temperature"].values
        logs = sdf["LogS"].values
        n = len(temps)

        if n >= APELBLAT_MIN_POINTS:
            params, cov, r2, ok = fit_apelblat(temps, logs)
            if ok and r2 > -1.0:
                fits[(sol, solv, src)] = {
                    "model": "apelblat", "params": params, "cov": cov,
                    "r2": r2, "n_points": n,
                    "t_min": temps.min(), "t_max": temps.max(),
                    "temps": temps, "logs": logs,
                }
                stats["apelblat"] += 1
            else:
                fits[(sol, solv, src)] = {
                    "model": "failed", "n_points": n,
                    "temps": temps, "logs": logs,
                }
                stats["failed"] += 1

        elif n == VANTHOFF_POINTS:
            params, cov, r2, ok = fit_vanthoff(temps, logs)
            if ok:
                fits[(sol, solv, src)] = {
                    "model": "vanthoff", "params": params, "cov": cov,
                    "r2": r2, "n_points": n,
                    "t_min": temps.min(), "t_max": temps.max(),
                    "temps": temps, "logs": logs,
                }
                stats["vanthoff"] += 1
            else:
                fits[(sol, solv, src)] = {
                    "model": "failed", "n_points": n,
                    "temps": temps, "logs": logs,
                }
                stats["failed"] += 1

        else:  # 1 point
            fits[(sol, solv, src)] = {
                "model": "isolated", "n_points": n,
                "temps": temps, "logs": logs,
            }
            stats["isolated"] += 1

    return fits, stats


# ─── Step 3.4: Inter-Lab Variability ────────────────────────────────────────

def compute_interlab_variability(pair_groups, fits, copycat_pairs):
    """
    For each (solute, solvent) pair with ≥2 truly independent sources:
      - Choose reference temps where multiple sources have data/reliable interpolation
      - Compute inter-lab MAE, RMSE, std at each T_ref
    Returns per-pair and aggregate statistics.
    """
    # Build set of copycat source pairs to exclude
    copycat_set = set()
    for d in copycat_pairs:
        copycat_set.add((d["solute"], d["solvent"], d["source1"], d["source2"]))
        copycat_set.add((d["solute"], d["solvent"], d["source2"], d["source1"]))

    pair_results = []

    multi_pairs = {k: v for k, v in pair_groups.items()
                   if v["Source"].nunique() >= 2}

    for (sol, solv), gdf in multi_pairs.items():
        sources = gdf["Source"].unique()

        # Get fitted sources for this pair
        fitted_sources = {}
        for src in sources:
            key = (sol, solv, src)
            if key in fits and fits[key]["model"] in ("apelblat", "vanthoff"):
                fitted_sources[src] = fits[key]

        if len(fitted_sources) < 2:
            # Try direct comparison at shared temperatures (unfitted)
            # Use raw values at matching temperatures instead
            source_raw = {}
            for src in sources:
                sdf = gdf[gdf["Source"] == src]
                source_raw[src] = dict(zip(sdf["Temperature"].values, sdf["LogS"].values))

            # Find independent source pairs (excluding copycats)
            indep_sources = []
            for s1, s2 in combinations(sources, 2):
                if (sol, solv, s1, s2) not in copycat_set:
                    indep_sources.append((s1, s2))

            if not indep_sources:
                continue

            # Direct comparison at shared temperatures
            # Only compare if sources have measurements with temp gap ≥ TEMP_GAP_MIN
            deviations = []
            for s1, s2 in indep_sources:
                t1_all = sorted(source_raw[s1].keys())
                t2_all = sorted(source_raw[s2].keys())
                # Check independence: at least one pair of measurements ≥2K apart
                has_gap = any(
                    abs(ta - tb) >= TEMP_GAP_MIN
                    for ta in t1_all for tb in t2_all
                )
                if not has_gap and len(t1_all) > 1 and len(t2_all) > 1:
                    continue  # skip — may be same-day measurements
                shared = set(source_raw[s1].keys()) & set(source_raw[s2].keys())
                for t in shared:
                    deviations.append(abs(source_raw[s1][t] - source_raw[s2][t]))

            if deviations:
                pair_results.append({
                    "solute": sol, "solvent": solv,
                    "n_sources": len(sources),
                    "n_indep_pairs": len(indep_sources),
                    "method": "direct",
                    "n_comparisons": len(deviations),
                    "mae": np.mean(deviations),
                    "rmse": np.sqrt(np.mean(np.array(deviations) ** 2)),
                    "std": np.std(deviations),
                    "max_dev": np.max(deviations),
                })
            continue

        # Filter to independent source pairs
        indep_fitted = {}
        for s1, s2 in combinations(fitted_sources.keys(), 2):
            if (sol, solv, s1, s2) not in copycat_set:
                indep_fitted[s1] = fitted_sources[s1]
                indep_fitted[s2] = fitted_sources[s2]

        if len(indep_fitted) < 2:
            continue

        # Find reference temperatures: common range across sources
        all_tmin = max(f["t_min"] for f in indep_fitted.values())
        all_tmax = min(f["t_max"] for f in indep_fitted.values())
        if all_tmin >= all_tmax:
            # No overlapping range — try direct comparison
            continue

        # Sample reference temperatures within the overlap
        t_refs = np.arange(
            np.ceil(all_tmin / 5) * 5,
            np.floor(all_tmax / 5) * 5 + 1,
            5.0
        )
        if len(t_refs) == 0:
            t_refs = np.array([(all_tmin + all_tmax) / 2])

        # Predict at each T_ref and compute inter-source variability
        deviations = []
        for t_ref in t_refs:
            preds = []
            for src, fdata in indep_fitted.items():
                pred, ci = predict_with_ci(
                    fdata["params"], fdata["cov"], t_ref, fdata["model"]
                )
                if ci < CI_WIDTH_THRESH:
                    preds.append(pred)

            if len(preds) >= 2:
                preds = np.array(preds)
                for p1, p2 in combinations(preds, 2):
                    deviations.append(abs(p1 - p2))

        if deviations:
            deviations = np.array(deviations)
            pair_results.append({
                "solute": sol, "solvent": solv,
                "n_sources": len(sources),
                "n_indep_pairs": len(indep_fitted),
                "method": "interpolated",
                "n_comparisons": len(deviations),
                "mae": np.mean(deviations),
                "rmse": np.sqrt(np.mean(deviations ** 2)),
                "std": np.std(deviations),
                "max_dev": np.max(deviations),
            })

    return pair_results


# ─── Step 3.5: Source Reliability Ranking ────────────────────────────────────

def rank_sources(pair_groups, fits, copycat_pairs):
    """
    For each DOI, compute its average deviation from the consensus
    across all overlapping pairs.
    """
    copycat_set = set()
    for d in copycat_pairs:
        copycat_set.add((d["solute"], d["solvent"], d["source1"], d["source2"]))
        copycat_set.add((d["solute"], d["solvent"], d["source2"], d["source1"]))

    doi_deviations = defaultdict(list)
    doi_pair_count = defaultdict(int)

    multi_pairs = {k: v for k, v in pair_groups.items()
                   if v["Source"].nunique() >= 2}

    for (sol, solv), gdf in multi_pairs.items():
        sources = gdf["Source"].unique()
        if len(sources) < 2:
            continue

        # Compute consensus: median across all sources at shared temps
        source_data = {}
        for src in sources:
            sdf = gdf[gdf["Source"] == src]
            source_data[src] = dict(zip(sdf["Temperature"].values, sdf["LogS"].values))

        # For each shared temperature, compute median
        all_temps = set()
        for d in source_data.values():
            all_temps.update(d.keys())

        for src in sources:
            other_srcs = [s for s in sources if s != src
                          and (sol, solv, src, s) not in copycat_set]
            if not other_srcs:
                continue

            # Compare this source against others at shared temps
            deviations = []
            for t in source_data[src]:
                others_at_t = [source_data[s][t] for s in other_srcs if t in source_data[s]]
                if others_at_t:
                    consensus = np.median(others_at_t)
                    deviations.append(abs(source_data[src][t] - consensus))

            if deviations:
                doi_deviations[src].extend(deviations)
                doi_pair_count[src] += 1

    # Compute per-DOI statistics
    doi_stats = []
    for doi in doi_deviations:
        devs = doi_deviations[doi]
        doi_stats.append({
            "doi": doi,
            "n_overlapping_pairs": doi_pair_count[doi],
            "n_comparisons": len(devs),
            "mae_from_consensus": np.mean(devs),
            "median_dev": np.median(devs),
            "max_dev": np.max(devs),
            "std_dev": np.std(devs),
        })

    doi_stats.sort(key=lambda x: x["mae_from_consensus"])
    return doi_stats


# ─── Step 3.6: Stratified Aleatoric Limit ───────────────────────────────────

def stratified_aleatoric(pair_results, src_counts):
    """
    Compute aleatoric limit stratified by number of independent sources:
      - Easy: ≥5 sources
      - Medium: 3-4 sources
      - Hard: 2 sources
    """
    tiers = {
        "easy_ge5": [],
        "medium_3_4": [],
        "hard_2": [],
        "all": [],
    }

    for pr in pair_results:
        key = (pr["solute"], pr["solvent"])
        n_src = src_counts.get(key, pr["n_sources"])
        tiers["all"].append(pr["mae"])

        if n_src >= 5:
            tiers["easy_ge5"].append(pr["mae"])
        elif n_src >= 3:
            tiers["medium_3_4"].append(pr["mae"])
        elif n_src >= 2:
            tiers["hard_2"].append(pr["mae"])

    results = {}
    for tier, values in tiers.items():
        if values:
            values = np.array(values)
            results[tier] = {
                "n_pairs": len(values),
                "mean_mae": round(float(np.mean(values)), 4),
                "median_mae": round(float(np.median(values)), 4),
                "std_mae": round(float(np.std(values)), 4),
                "p25": round(float(np.percentile(values, 25)), 4),
                "p75": round(float(np.percentile(values, 75)), 4),
                "p90": round(float(np.percentile(values, 90)), 4),
                "rmse": round(float(np.sqrt(np.mean(values ** 2))), 4),
            }
        else:
            results[tier] = {"n_pairs": 0}

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def make_plots(pair_results, doi_stats, pairwise_maes, aleatoric, fit_stats,
               pair_source_info, figures_dir):
    """Generate all Phase 3 figures."""
    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Pairwise MAE distribution (copycat detection) ──
    if pairwise_maes:
        maes = [d["mae"] for d in pairwise_maes]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(maes, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(NEAR_DUP_MAE_THRESH, color="red", ls="--", lw=2,
                   label=f"Near-dup threshold ({NEAR_DUP_MAE_THRESH})")
        ax.set_xlabel("Pairwise MAE between sources (log S units)")
        ax.set_ylabel("Count")
        ax.set_title("Pairwise MAE Distribution Between Sources\n(shared temperature comparisons)")
        ax.legend()
        ax.set_xlim(0, min(3.0, np.percentile(maes, 99)))
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "01_pairwise_mae_distribution.png"), dpi=300)
        plt.close()

        # Zoomed in on low-MAE region
        fig, ax = plt.subplots(figsize=(10, 6))
        low_maes = [m for m in maes if m < 0.1]
        if low_maes:
            ax.hist(low_maes, bins=50, edgecolor="black", alpha=0.7, color="coral")
            ax.axvline(NEAR_DUP_MAE_THRESH, color="red", ls="--", lw=2,
                       label=f"Near-dup threshold ({NEAR_DUP_MAE_THRESH})")
            ax.set_xlabel("Pairwise MAE (log S units)")
            ax.set_ylabel("Count")
            ax.set_title("Zoomed: Low-MAE Region (potential copycats)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "02_pairwise_mae_zoomed.png"), dpi=300)
        plt.close()

    # ── 2. Inter-lab MAE distribution ──
    if pair_results:
        interlab_maes = [pr["mae"] for pr in pair_results]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(interlab_maes, bins=60, edgecolor="black", alpha=0.7, color="mediumpurple")
        mean_mae = np.mean(interlab_maes)
        median_mae = np.median(interlab_maes)
        ax.axvline(mean_mae, color="red", ls="--", lw=2, label=f"Mean: {mean_mae:.3f}")
        ax.axvline(median_mae, color="green", ls="--", lw=2, label=f"Median: {median_mae:.3f}")
        ax.axvline(0.6, color="orange", ls=":", lw=2, label="Palmer & Mitchell (0.6)")
        ax.set_xlabel("Inter-lab MAE (log S units)")
        ax.set_ylabel("Count (solute-solvent pairs)")
        ax.set_title("Distribution of Inter-Lab MAE\n(truly independent sources only)")
        ax.legend()
        ax.set_xlim(0, min(5.0, np.percentile(interlab_maes, 99)))
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "03_interlab_mae_distribution.png"), dpi=300)
        plt.close()

    # ── 3. Source reliability histogram ──
    if doi_stats:
        doi_maes = [d["mae_from_consensus"] for d in doi_stats]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(doi_maes, bins=50, edgecolor="black", alpha=0.7, color="teal")
        ax.axvline(SHAME_THRESH, color="red", ls="--", lw=2,
                   label=f"Shame threshold ({SHAME_THRESH})")
        ax.axvline(FAME_MAX_DEV, color="green", ls="--", lw=2,
                   label=f"Fame threshold ({FAME_MAX_DEV})")
        ax.set_xlabel("Average MAE from consensus (log S units)")
        ax.set_ylabel("Count (DOIs)")
        ax.set_title("Source Reliability: Per-DOI MAE from Consensus")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "04_source_reliability_histogram.png"), dpi=300)
        plt.close()

    # ── 4. Fit type breakdown ──
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(fit_stats.keys())
    values = list(fit_stats.values())
    colors = ["steelblue", "coral", "lightgray", "tomato"]
    ax.bar(labels, values, color=colors[:len(labels)], edgecolor="black")
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, str(v), ha="center", fontweight="bold")
    ax.set_ylabel("Count (source triples)")
    ax.set_title("Curve Fit Results by Model Type")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "05_fit_type_breakdown.png"), dpi=300)
    plt.close()

    # ── 5. Stratified aleatoric limit ──
    if aleatoric:
        tier_names = []
        tier_means = []
        tier_medians = []
        tier_p90s = []
        for tier in ["easy_ge5", "medium_3_4", "hard_2", "all"]:
            if tier in aleatoric and aleatoric[tier].get("n_pairs", 0) > 0:
                nice = {"easy_ge5": "Easy (≥5 src)", "medium_3_4": "Medium (3-4)",
                         "hard_2": "Hard (2)", "all": "All"}
                tier_names.append(nice.get(tier, tier))
                tier_means.append(aleatoric[tier]["mean_mae"])
                tier_medians.append(aleatoric[tier]["median_mae"])
                tier_p90s.append(aleatoric[tier]["p90"])

        if tier_names:
            x = np.arange(len(tier_names))
            w = 0.25
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x - w, tier_means, w, label="Mean MAE", color="steelblue", edgecolor="black")
            ax.bar(x, tier_medians, w, label="Median MAE", color="coral", edgecolor="black")
            ax.bar(x + w, tier_p90s, w, label="P90 MAE", color="mediumpurple", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(tier_names)
            ax.set_ylabel("MAE (log S units)")
            ax.set_title("Stratified Aleatoric Limit Estimates")
            ax.legend()
            ax.axhline(0.6, color="orange", ls=":", lw=2, alpha=0.7, label="Palmer & Mitchell")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "06_stratified_aleatoric_limit.png"), dpi=300)
            plt.close()

    # ── 6. Source count distribution ──
    if pair_source_info:
        src_vals = list(pair_source_info.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        max_src = min(max(src_vals), 15)
        bins = np.arange(0.5, max_src + 1.5, 1)
        ax.hist(src_vals, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Number of independent sources (DOIs)")
        ax.set_ylabel("Count (solute-solvent pairs)")
        ax.set_title("Distribution of Source Coverage per Pair")
        ax.set_xticks(range(1, int(max_src) + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "07_source_count_distribution.png"), dpi=300)
        plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Source analysis and inter-lab variability.")
    parser.add_argument("--input", required=True, help="Cleaned BigSolDB CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for artifacts (JSON, CSV)")
    parser.add_argument("--figures-dir", required=True, help="Directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Load ──
    print("Loading cleaned data...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows, {df['Solute'].nunique()} solutes, "
          f"{df['Solvent'].nunique()} solvents, {df['Source'].nunique()} DOIs")

    # ── Step 3.1: Group ──
    print("\nStep 3.1: Grouping by (solute, solvent) and (solute, solvent, source)...")
    pair_groups, triple_groups = group_data(df)
    src_stats, src_counts = pair_source_stats(pair_groups)
    print(f"  Total pairs: {src_stats['total_pairs']:,}")
    print(f"  ≥2 DOIs: {src_stats['ge2']:,} ({src_stats['ge2_pct']}%)")
    print(f"  ≥3 DOIs: {src_stats['ge3']:,} ({src_stats['ge3_pct']}%)")
    print(f"  ≥5 DOIs: {src_stats['ge5']:,} ({src_stats['ge5_pct']}%)")
    print(f"  Total triples: {len(triple_groups):,}")

    # ── Step 3.2: Copycat detection ──
    print("\nStep 3.2: Detecting copycats...")
    exact_dups, near_dups, pairwise_maes = detect_copycats(pair_groups)
    all_copycats = exact_dups + near_dups
    print(f"  Exact duplicates: {len(exact_dups)} source pairs")
    print(f"  Near-duplicates (MAE < {NEAR_DUP_MAE_THRESH}): {len(near_dups)} source pairs")
    print(f"  Total pairwise comparisons: {len(pairwise_maes)}")
    if pairwise_maes:
        all_maes = [d["mae"] for d in pairwise_maes]
        print(f"  Overall pairwise MAE: mean={np.mean(all_maes):.4f}, "
              f"median={np.median(all_maes):.4f}")

    # Save copycat results
    if exact_dups:
        pd.DataFrame(exact_dups).to_csv(
            os.path.join(args.output_dir, "exact_duplicates.csv"), index=False)
    if near_dups:
        pd.DataFrame(near_dups).to_csv(
            os.path.join(args.output_dir, "near_duplicates.csv"), index=False)
    if pairwise_maes:
        pd.DataFrame(pairwise_maes).to_csv(
            os.path.join(args.output_dir, "pairwise_maes.csv"), index=False)

    # ── Step 3.3: Curve fitting ──
    print("\nStep 3.3: Fitting thermodynamic curves per source...")
    fits, fit_stats = fit_all_triples(triple_groups)
    print(f"  Apelblat (≥3 temps): {fit_stats['apelblat']:,}")
    print(f"  van't Hoff (2 temps): {fit_stats['vanthoff']:,}")
    print(f"  Isolated (1 temp): {fit_stats['isolated']:,}")
    print(f"  Failed fits: {fit_stats['failed']:,}")

    # Apelblat R² distribution
    apelblat_r2 = [f["r2"] for f in fits.values()
                   if f["model"] == "apelblat"]
    if apelblat_r2:
        print(f"  Apelblat R² — mean: {np.mean(apelblat_r2):.4f}, "
              f"median: {np.median(apelblat_r2):.4f}, "
              f"≥0.95: {sum(1 for r in apelblat_r2 if r >= 0.95)}, "
              f"≥0.99: {sum(1 for r in apelblat_r2 if r >= 0.99)}")

    # ── Step 3.4: Inter-lab variability ──
    print("\nStep 3.4: Computing inter-lab variability...")
    pair_results = compute_interlab_variability(pair_groups, fits, all_copycats)
    print(f"  Pairs with inter-lab comparison: {len(pair_results)}")
    if pair_results:
        all_interlab = [pr["mae"] for pr in pair_results]
        print(f"  Inter-lab MAE — mean: {np.mean(all_interlab):.4f}, "
              f"median: {np.median(all_interlab):.4f}, "
              f"std: {np.std(all_interlab):.4f}")
        method_counts = defaultdict(int)
        for pr in pair_results:
            method_counts[pr["method"]] += 1
        print(f"  Methods: {dict(method_counts)}")

    pd.DataFrame(pair_results).to_csv(
        os.path.join(args.output_dir, "interlab_variability.csv"), index=False)

    # ── Step 3.5: Source reliability ──
    print("\nStep 3.5: Ranking source reliability...")
    doi_stats = rank_sources(pair_groups, fits, all_copycats)
    print(f"  DOIs with overlap data: {len(doi_stats)}")
    if doi_stats:
        hall_of_fame = [d for d in doi_stats if d["mae_from_consensus"] <= FAME_MAX_DEV]
        hall_of_shame = [d for d in doi_stats if d["mae_from_consensus"] >= SHAME_THRESH]
        print(f"  Hall of Fame (MAE ≤ {FAME_MAX_DEV}): {len(hall_of_fame)} DOIs")
        print(f"  Hall of Shame (MAE ≥ {SHAME_THRESH}): {len(hall_of_shame)} DOIs")

        print("\n  Top 10 most reliable DOIs:")
        for d in doi_stats[:10]:
            print(f"    {d['doi']:45s} MAE={d['mae_from_consensus']:.4f} "
                  f"({d['n_overlapping_pairs']} pairs, {d['n_comparisons']} comparisons)")

        print(f"\n  Top 10 least reliable DOIs:")
        for d in doi_stats[-10:]:
            print(f"    {d['doi']:45s} MAE={d['mae_from_consensus']:.4f} "
                  f"({d['n_overlapping_pairs']} pairs, {d['n_comparisons']} comparisons)")

    pd.DataFrame(doi_stats).to_csv(
        os.path.join(args.output_dir, "doi_reliability.csv"), index=False)

    # ── Step 3.6: Stratified aleatoric limit ──
    print("\nStep 3.6: Computing stratified aleatoric limits...")
    aleatoric = stratified_aleatoric(pair_results, src_counts)
    for tier, vals in aleatoric.items():
        if vals.get("n_pairs", 0) > 0:
            print(f"  {tier:15s}: n={vals['n_pairs']:4d}, "
                  f"mean={vals['mean_mae']:.4f}, median={vals['median_mae']:.4f}, "
                  f"P90={vals['p90']:.4f}")
        else:
            print(f"  {tier:15s}: n=0 (insufficient data)")

    with open(os.path.join(args.output_dir, "aleatoric_limits.json"), "w") as f:
        json.dump(aleatoric, f, indent=2)

    # ── Save summary ──
    summary = {
        "dataset": {
            "rows": len(df),
            "solutes": int(df["Solute"].nunique()),
            "solvents": int(df["Solvent"].nunique()),
            "dois": int(df["Source"].nunique()),
        },
        "source_coverage": src_stats,
        "copycat_detection": {
            "exact_duplicates": len(exact_dups),
            "near_duplicates": len(near_dups),
            "total_pairwise_comparisons": len(pairwise_maes),
        },
        "curve_fitting": fit_stats,
        "interlab_variability": {
            "n_pairs_compared": len(pair_results),
            "mean_mae": round(float(np.mean([pr["mae"] for pr in pair_results])), 4) if pair_results else None,
            "median_mae": round(float(np.median([pr["mae"] for pr in pair_results])), 4) if pair_results else None,
        },
        "source_reliability": {
            "n_dois_ranked": len(doi_stats),
            "hall_of_fame": len([d for d in doi_stats if d["mae_from_consensus"] <= FAME_MAX_DEV]),
            "hall_of_shame": len([d for d in doi_stats if d["mae_from_consensus"] >= SHAME_THRESH]),
        },
        "aleatoric_limits": aleatoric,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ──
    print("\nGenerating figures...")
    make_plots(pair_results, doi_stats, pairwise_maes, aleatoric, fit_stats,
               src_counts, args.figures_dir)

    print(f"\nPhase 3 complete. Artifacts saved to: {args.output_dir}")
    print(f"Figures saved to: {args.figures_dir}")


if __name__ == "__main__":
    main()
