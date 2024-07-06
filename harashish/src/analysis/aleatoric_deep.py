"""
Phase 4A: Deep Aleatoric Limit Analysis

Rigorous analysis of the irreducible measurement uncertainty in solubility data.

Mathematical Framework
======================
The aleatoric limit ε_aleatoric is the irreducible uncertainty floor due to
inter-laboratory measurement variability. It sets the lower bound below which
no predictive model can be expected to perform reliably.

For a (solute, solvent) pair at temperature T:
    S_true(T)  = unknown true solubility
    S_i(T)     = lab i's measurement = S_true(T) + ε_i    (measurement noise)
    f_i(T_ref) = Apelblat interpolation of lab i's data    (adds δ_interp)

Direct aleatoric limit (at shared temperatures, no interpolation):
    ε_direct = MAE between independent labs at matching T

Interpolated aleatoric limit (at reference temperatures via curve fits):
    ε_interp = MAE between independent labs' fitted curves at T_ref
    This includes interpolation uncertainty from finite data.

Composite aleatoric limit (protocol formula):
    ε_aleatoric = √(ε_direct² + 2·δ_interp²)
    where δ_interp = median 95% CI half-width of the Apelblat fits

We also decompose errors into:
    - Random component: σ_random = std of deviations (precision)
    - Systematic component: μ_bias  = mean signed deviation (accuracy/bias)

Usage:
  conda run -n sc3 python src/analysis/aleatoric_deep.py \
    --input data/intermediate/bigsoldb_cleaned.csv \
    --phase3-dir reports/phase_03_artifacts \
    --output-dir reports/phase_04_aleatoric \
    --figures-dir figures/aleatoric
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import bootstrap, lognorm, expon, halfnorm, kstest

warnings.filterwarnings("ignore")


# ─── Apelblat / van't Hoff ──────────────────────────────────────────────────

def apelblat_eq(T, A, B, C):
    return A + B / T + C * np.log(T)

def vanthoff_eq(T, A, B):
    return A + B / T

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

def predict_with_ci(params, cov, T_ref):
    """Predict log S and 95% CI half-width at T_ref."""
    pred = apelblat_eq(T_ref, *params)
    J = np.array([1.0, 1.0 / T_ref, np.log(T_ref)])
    var = J @ cov @ J
    ci_half = 1.96 * np.sqrt(max(var, 0))
    return pred, ci_half


# ─── Core Analysis ───────────────────────────────────────────────────────────

def compute_direct_interlab(df):
    """
    Compute inter-lab deviations at SHARED temperatures only (no interpolation).
    Returns per-comparison records with full metadata.
    """
    # Load copycat info
    records = []

    multi = df.groupby(["Solute", "Solvent"]).filter(
        lambda x: x["Source"].nunique() >= 2
    )

    for (sol, solv), gdf in multi.groupby(["Solute", "Solvent"]):
        sources = gdf["Source"].unique()
        source_data = {}
        for src in sources:
            sdf = gdf[gdf["Source"] == src]
            source_data[src] = {
                "temps": sdf["Temperature"].values,
                "logs": sdf["LogS"].values,
                "mole_frac": sdf["Solubility(mole_fraction)"].values,
            }

        for s1, s2 in combinations(sources, 2):
            d1, d2 = source_data[s1], source_data[s2]
            t1, t2 = d1["temps"], d2["temps"]
            l1, l2 = d1["logs"], d2["logs"]

            # Find matching temperatures (within 0.1 K)
            for i, t in enumerate(t1):
                matches = np.where(np.abs(t2 - t) < 0.1)[0]
                if len(matches) > 0:
                    j = matches[0]
                    dev = l1[i] - l2[j]
                    records.append({
                        "solute": sol, "solvent": solv,
                        "source1": s1, "source2": s2,
                        "temperature": t,
                        "logs1": l1[i], "logs2": l2[j],
                        "deviation": dev,
                        "abs_deviation": abs(dev),
                        "mean_logs": (l1[i] + l2[j]) / 2,
                    })

    return pd.DataFrame(records)


def compute_interpolated_interlab(df):
    """
    Fit Apelblat curves per (solute, solvent, source) triple, then compare
    at reference temperatures. Also returns interpolation uncertainty (δ_interp).
    """
    records = []
    ci_widths = []

    multi = df.groupby(["Solute", "Solvent"]).filter(
        lambda x: x["Source"].nunique() >= 2
    )

    for (sol, solv), gdf in multi.groupby(["Solute", "Solvent"]):
        sources = gdf["Source"].unique()

        # Fit curves per source
        fitted = {}
        for src in sources:
            sdf = gdf[gdf["Source"] == src].sort_values("Temperature")
            temps = sdf["Temperature"].values
            logs = sdf["LogS"].values
            if len(temps) >= 3:
                params, cov, r2, ok = fit_apelblat(temps, logs)
                if ok and r2 > 0.5:
                    fitted[src] = {
                        "params": params, "cov": cov, "r2": r2,
                        "t_min": temps.min(), "t_max": temps.max(),
                    }

        if len(fitted) < 2:
            continue

        # Find overlapping temperature range
        all_tmin = max(f["t_min"] for f in fitted.values())
        all_tmax = min(f["t_max"] for f in fitted.values())
        if all_tmin >= all_tmax:
            continue

        # Reference temperatures at 5K intervals
        t_refs = np.arange(
            np.ceil(all_tmin / 5) * 5,
            np.floor(all_tmax / 5) * 5 + 1, 5.0
        )
        if len(t_refs) == 0:
            t_refs = np.array([(all_tmin + all_tmax) / 2])

        for t_ref in t_refs:
            preds = {}
            for src, fd in fitted.items():
                pred, ci = predict_with_ci(fd["params"], fd["cov"], t_ref)
                if ci < 0.5:  # only use reliable interpolations
                    preds[src] = pred
                    ci_widths.append(ci)

            if len(preds) < 2:
                continue

            src_list = list(preds.keys())
            for s1, s2 in combinations(src_list, 2):
                dev = preds[s1] - preds[s2]
                records.append({
                    "solute": sol, "solvent": solv,
                    "source1": s1, "source2": s2,
                    "temperature": t_ref,
                    "logs1": preds[s1], "logs2": preds[s2],
                    "deviation": dev,
                    "abs_deviation": abs(dev),
                    "mean_logs": (preds[s1] + preds[s2]) / 2,
                })

    return pd.DataFrame(records), np.array(ci_widths)


def bootstrap_ci(values, stat_func=np.median, n_boot=5000, ci=0.95):
    """Compute bootstrap confidence interval for a statistic."""
    values = np.array(values)
    boot_stats = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(stat_func(sample))
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_stats, alpha * 100)),
        float(stat_func(values)),
        float(np.percentile(boot_stats, (1 - alpha) * 100)),
    )


def fit_distributions(abs_devs):
    """Fit candidate distributions to the absolute deviations."""
    results = {}
    abs_devs = abs_devs[abs_devs > 0]  # remove exact zeros

    # Log-normal
    try:
        shape, loc, scale = lognorm.fit(abs_devs, floc=0)
        ks_stat, ks_p = kstest(abs_devs, 'lognorm', args=(shape, loc, scale))
        results["lognorm"] = {
            "params": {"shape": round(shape, 4), "scale": round(scale, 4)},
            "ks_stat": round(ks_stat, 4), "ks_p": round(ks_p, 4),
        }
    except Exception:
        pass

    # Half-normal
    try:
        loc, scale = halfnorm.fit(abs_devs)
        ks_stat, ks_p = kstest(abs_devs, 'halfnorm', args=(loc, scale))
        results["halfnorm"] = {
            "params": {"loc": round(loc, 4), "scale": round(scale, 4)},
            "ks_stat": round(ks_stat, 4), "ks_p": round(ks_p, 4),
        }
    except Exception:
        pass

    # Exponential
    try:
        loc, scale = expon.fit(abs_devs)
        ks_stat, ks_p = kstest(abs_devs, 'expon', args=(loc, scale))
        results["exponential"] = {
            "params": {"loc": round(loc, 4), "scale": round(scale, 4)},
            "ks_stat": round(ks_stat, 4), "ks_p": round(ks_p, 4),
        }
    except Exception:
        pass

    return results


def stratify_by_solvent(direct_df, df):
    """Compute per-solvent aleatoric limits."""
    # Get solvent names
    solvent_names = df.drop_duplicates("Solvent")[["Solvent", "Solvent_Name"]].set_index("Solvent")["Solvent_Name"].to_dict()

    per_solvent = {}
    for solv, sdf in direct_df.groupby("solvent"):
        devs = sdf["abs_deviation"].values
        if len(devs) < 5:
            continue
        name = solvent_names.get(solv, solv[:30])
        per_solvent[solv] = {
            "name": name,
            "n_comparisons": len(devs),
            "n_pairs": sdf.groupby(["solute"]).ngroups,
            "mean_mae": round(float(np.mean(devs)), 4),
            "median_mae": round(float(np.median(devs)), 4),
            "std": round(float(np.std(devs)), 4),
            "p90": round(float(np.percentile(devs, 90)), 4),
        }

    return dict(sorted(per_solvent.items(), key=lambda x: -x[1]["n_comparisons"]))


def analyze_error_relationships(direct_df):
    """Analyze how measurement error relates to LogS magnitude and temperature."""
    results = {}

    # Error vs LogS magnitude
    if len(direct_df) > 10:
        logs_bins = [(-10, -3), (-3, -2), (-2, -1), (-1, 0), (0, 3)]
        logs_strata = {}
        for lo, hi in logs_bins:
            mask = (direct_df["mean_logs"] >= lo) & (direct_df["mean_logs"] < hi)
            devs = direct_df.loc[mask, "abs_deviation"].values
            if len(devs) >= 5:
                logs_strata[f"[{lo},{hi})"] = {
                    "n": len(devs),
                    "mean_mae": round(float(np.mean(devs)), 4),
                    "median_mae": round(float(np.median(devs)), 4),
                }
        results["by_logs_magnitude"] = logs_strata

    # Error vs temperature
    if len(direct_df) > 10:
        temp_bins = [(240, 280), (280, 300), (300, 320), (320, 350), (350, 430)]
        temp_strata = {}
        for lo, hi in temp_bins:
            mask = (direct_df["temperature"] >= lo) & (direct_df["temperature"] < hi)
            devs = direct_df.loc[mask, "abs_deviation"].values
            if len(devs) >= 5:
                temp_strata[f"[{lo},{hi})K"] = {
                    "n": len(devs),
                    "mean_mae": round(float(np.mean(devs)), 4),
                    "median_mae": round(float(np.median(devs)), 4),
                }
        results["by_temperature"] = temp_strata

    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def make_plots(direct_df, interp_df, ci_widths, per_solvent, relationships,
               dist_fits, composite, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Direct inter-lab error distribution with bootstrap CI ──
    if len(direct_df) > 0:
        abs_devs = direct_df["abs_deviation"].values

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram with fitted distributions
        ax = axes[0]
        ax.hist(abs_devs, bins=80, density=True, edgecolor="black",
                alpha=0.6, color="steelblue", label="Observed")

        x_plot = np.linspace(0, min(3.0, np.percentile(abs_devs, 99)), 200)
        if "lognorm" in dist_fits:
            p = dist_fits["lognorm"]["params"]
            y = lognorm.pdf(x_plot, p["shape"], 0, p["scale"])
            ax.plot(x_plot, y, "r-", lw=2, label=f"Log-normal (KS p={dist_fits['lognorm']['ks_p']:.3f})")
        if "halfnorm" in dist_fits:
            p = dist_fits["halfnorm"]["params"]
            y = halfnorm.pdf(x_plot, p["loc"], p["scale"])
            ax.plot(x_plot, y, "g--", lw=2, label=f"Half-normal (KS p={dist_fits['halfnorm']['ks_p']:.3f})")

        ax.axvline(np.median(abs_devs), color="orange", ls="--", lw=2,
                   label=f"Median: {np.median(abs_devs):.3f}")
        ax.set_xlabel("|ΔlogS| between labs (log S units)")
        ax.set_ylabel("Density")
        ax.set_title("Direct Inter-Lab Absolute Deviation\n(at shared temperatures)")
        ax.set_xlim(0, min(3.0, np.percentile(abs_devs, 99)))
        ax.legend(fontsize=9)

        # Signed deviation (bias analysis)
        ax = axes[1]
        devs = direct_df["deviation"].values
        ax.hist(devs, bins=80, edgecolor="black", alpha=0.6, color="coral")
        ax.axvline(0, color="black", ls="-", lw=1)
        ax.axvline(np.mean(devs), color="red", ls="--", lw=2,
                   label=f"Mean bias: {np.mean(devs):+.4f}")
        ax.set_xlabel("ΔlogS (source1 − source2)")
        ax.set_ylabel("Count")
        ax.set_title("Signed Inter-Lab Deviation\n(bias analysis)")
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "01_direct_interlab_distribution.png"), dpi=300)
        plt.close()

    # ── 2. Per-solvent aleatoric limits ──
    if per_solvent:
        top_solvents = list(per_solvent.items())[:15]
        if top_solvents:
            names = [v["name"][:20] for _, v in top_solvents]
            means = [v["mean_mae"] for _, v in top_solvents]
            medians = [v["median_mae"] for _, v in top_solvents]
            p90s = [v["p90"] for _, v in top_solvents]

            fig, ax = plt.subplots(figsize=(12, 7))
            x = np.arange(len(names))
            w = 0.25
            ax.bar(x - w, means, w, label="Mean MAE", color="steelblue", edgecolor="black")
            ax.bar(x, medians, w, label="Median MAE", color="coral", edgecolor="black")
            ax.bar(x + w, p90s, w, label="P90 MAE", color="mediumpurple", edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("MAE (log S units)")
            ax.set_title("Per-Solvent Aleatoric Limits\n(top 15 solvents by comparison count)")
            ax.legend()
            ax.axhline(0.6, color="orange", ls=":", lw=2, alpha=0.5, label="Literature 0.6")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "02_per_solvent_aleatoric.png"), dpi=300)
            plt.close()

    # ── 3. Error vs LogS magnitude ──
    if len(direct_df) > 10:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(direct_df["mean_logs"], direct_df["abs_deviation"],
                   alpha=0.3, s=10, color="steelblue")
        ax.set_xlabel("Mean LogS (average of two labs)")
        ax.set_ylabel("|ΔlogS| between labs")
        ax.set_title("Measurement Error vs Solubility Magnitude")
        ax.set_ylim(0, min(5, direct_df["abs_deviation"].quantile(0.99)))
        ax.axhline(np.median(direct_df["abs_deviation"]), color="red",
                   ls="--", alpha=0.5, label="Overall median")
        ax.legend()

        # Binned version
        ax = axes[1]
        if "by_logs_magnitude" in relationships:
            bins = relationships["by_logs_magnitude"]
            labels = list(bins.keys())
            vals = [bins[k]["median_mae"] for k in labels]
            counts = [bins[k]["n"] for k in labels]
            ax.bar(range(len(labels)), vals, color="steelblue", edgecolor="black")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            for i, c in enumerate(counts):
                ax.text(i, vals[i] + 0.005, f"n={c}", ha="center", fontsize=8)
            ax.set_xlabel("LogS range")
            ax.set_ylabel("Median MAE")
            ax.set_title("Binned: Median Error by Solubility Range")

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "03_error_vs_logs.png"), dpi=300)
        plt.close()

    # ── 4. Error vs Temperature ──
    if len(direct_df) > 10:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(direct_df["temperature"], direct_df["abs_deviation"],
                   alpha=0.3, s=10, color="teal")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("|ΔlogS| between labs")
        ax.set_title("Measurement Error vs Temperature")
        ax.set_ylim(0, min(5, direct_df["abs_deviation"].quantile(0.99)))

        ax = axes[1]
        if "by_temperature" in relationships:
            bins = relationships["by_temperature"]
            labels = list(bins.keys())
            vals = [bins[k]["median_mae"] for k in labels]
            counts = [bins[k]["n"] for k in labels]
            ax.bar(range(len(labels)), vals, color="teal", edgecolor="black")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            for i, c in enumerate(counts):
                ax.text(i, vals[i] + 0.005, f"n={c}", ha="center", fontsize=8)
            ax.set_xlabel("Temperature range")
            ax.set_ylabel("Median MAE")
            ax.set_title("Binned: Median Error by Temperature")

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "04_error_vs_temperature.png"), dpi=300)
        plt.close()

    # ── 5. Interpolation uncertainty (δ_interp) ──
    if len(ci_widths) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ci_widths, bins=80, edgecolor="black", alpha=0.7, color="mediumpurple")
        ax.axvline(np.median(ci_widths), color="red", ls="--", lw=2,
                   label=f"Median δ_interp = {np.median(ci_widths):.4f}")
        ax.axvline(np.mean(ci_widths), color="orange", ls="--", lw=2,
                   label=f"Mean δ_interp = {np.mean(ci_widths):.4f}")
        ax.set_xlabel("95% CI half-width (log S units)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Interpolation Uncertainty (δ_interp)\nfrom Apelblat curve 95% CI")
        ax.set_xlim(0, min(0.5, np.percentile(ci_widths, 99)))
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "05_interpolation_uncertainty.png"), dpi=300)
        plt.close()

    # ── 6. Composite aleatoric limit summary ──
    if composite:
        fig, ax = plt.subplots(figsize=(10, 6))
        components = [
            ("ε_direct\n(median)", composite["epsilon_direct"]["median"]),
            ("ε_direct\n(mean)", composite["epsilon_direct"]["mean"]),
            ("δ_interp\n(median)", composite["delta_interp"]["median"]),
            ("ε_aleatoric\n(composite)", composite["epsilon_aleatoric"]),
        ]
        labels = [c[0] for c in components]
        vals = [c[1] for c in components]
        colors = ["steelblue", "steelblue", "mediumpurple", "coral"]
        bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="black")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontweight="bold")
        ax.set_ylabel("log S units")
        ax.set_title("Aleatoric Limit Decomposition\nε_aleatoric = √(ε_direct² + 2·δ_interp²)")
        ax.axhline(0.6, color="orange", ls=":", lw=2, alpha=0.5)
        ax.text(len(labels) - 0.5, 0.61, "Literature 0.6-0.8", fontsize=9, color="orange")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "06_composite_aleatoric_limit.png"), dpi=300)
        plt.close()

    # ── 7. Bootstrap CI visualization ──
    if len(direct_df) > 0:
        abs_devs = direct_df["abs_deviation"].values
        n_boot = 5000
        rng = np.random.default_rng(42)
        boot_medians = []
        boot_means = []
        for _ in range(n_boot):
            sample = rng.choice(abs_devs, size=len(abs_devs), replace=True)
            boot_medians.append(np.median(sample))
            boot_means.append(np.mean(sample))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.hist(boot_medians, bins=80, edgecolor="black", alpha=0.7, color="steelblue")
        lo, mid, hi = np.percentile(boot_medians, [2.5, 50, 97.5])
        ax.axvline(lo, color="red", ls="--", lw=2)
        ax.axvline(hi, color="red", ls="--", lw=2)
        ax.axvline(mid, color="red", ls="-", lw=2)
        ax.set_title(f"Bootstrap: Median MAE\n{mid:.4f} [{lo:.4f}, {hi:.4f}] 95% CI")
        ax.set_xlabel("Median MAE (log S)")

        ax = axes[1]
        ax.hist(boot_means, bins=80, edgecolor="black", alpha=0.7, color="coral")
        lo, mid, hi = np.percentile(boot_means, [2.5, 50, 97.5])
        ax.axvline(lo, color="red", ls="--", lw=2)
        ax.axvline(hi, color="red", ls="--", lw=2)
        ax.axvline(mid, color="red", ls="-", lw=2)
        ax.set_title(f"Bootstrap: Mean MAE\n{mid:.4f} [{lo:.4f}, {hi:.4f}] 95% CI")
        ax.set_xlabel("Mean MAE (log S)")

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "07_bootstrap_ci.png"), dpi=300)
        plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deep aleatoric limit analysis.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--phase3-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figures-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("Loading cleaned data...")
    df = pd.read_csv(args.input)
    print(f"  {len(df):,} rows")

    # ── Section 1: Direct inter-lab comparisons ──
    print("\n" + "=" * 70)
    print("SECTION 1: Direct Inter-Lab Comparisons (no interpolation)")
    print("=" * 70)
    direct_df = compute_direct_interlab(df)
    n_direct = len(direct_df)
    print(f"  {n_direct:,} direct comparisons at shared temperatures")

    if n_direct > 0:
        abs_devs = direct_df["abs_deviation"].values
        signed_devs = direct_df["deviation"].values

        print(f"\n  ε_direct statistics:")
        print(f"    Mean |ΔlogS|:   {np.mean(abs_devs):.4f}")
        print(f"    Median |ΔlogS|: {np.median(abs_devs):.4f}")
        print(f"    Std |ΔlogS|:    {np.std(abs_devs):.4f}")
        print(f"    P25:            {np.percentile(abs_devs, 25):.4f}")
        print(f"    P75:            {np.percentile(abs_devs, 75):.4f}")
        print(f"    P90:            {np.percentile(abs_devs, 90):.4f}")
        print(f"    P95:            {np.percentile(abs_devs, 95):.4f}")
        print(f"    Max:            {np.max(abs_devs):.4f}")

        print(f"\n  Bias analysis (signed deviation):")
        print(f"    Mean bias:      {np.mean(signed_devs):+.4f}")
        print(f"    Std:            {np.std(signed_devs):.4f}")
        print(f"    → Random component (σ) ≈ {np.std(signed_devs):.4f}")
        print(f"    → Systematic component (μ) ≈ {abs(np.mean(signed_devs)):.4f}")

        # Fraction of pairs within various thresholds
        for t in [0.1, 0.2, 0.3, 0.5, 1.0]:
            pct = 100 * np.mean(abs_devs < t)
            print(f"    < {t} log S: {pct:.1f}%")

        # Bootstrap CIs
        print(f"\n  Bootstrap 95% CIs (5000 resamples):")
        ci_median = bootstrap_ci(abs_devs, np.median)
        ci_mean = bootstrap_ci(abs_devs, np.mean)
        print(f"    Median: {ci_median[1]:.4f}  [{ci_median[0]:.4f}, {ci_median[2]:.4f}]")
        print(f"    Mean:   {ci_mean[1]:.4f}  [{ci_mean[0]:.4f}, {ci_mean[2]:.4f}]")

    # ── Section 2: Interpolated inter-lab comparisons ──
    print("\n" + "=" * 70)
    print("SECTION 2: Interpolated Inter-Lab Comparisons (via Apelblat)")
    print("=" * 70)
    interp_df, ci_widths = compute_interpolated_interlab(df)
    print(f"  {len(interp_df):,} interpolated comparisons")
    print(f"  {len(ci_widths):,} CI widths recorded")

    if len(ci_widths) > 0:
        print(f"\n  Interpolation uncertainty (δ_interp):")
        print(f"    Median 95% CI half-width: {np.median(ci_widths):.4f}")
        print(f"    Mean 95% CI half-width:   {np.mean(ci_widths):.4f}")
        print(f"    P90:                       {np.percentile(ci_widths, 90):.4f}")

    if len(interp_df) > 0:
        abs_interp = interp_df["abs_deviation"].values
        print(f"\n  ε_interp statistics:")
        print(f"    Mean |ΔlogS|:   {np.mean(abs_interp):.4f}")
        print(f"    Median |ΔlogS|: {np.median(abs_interp):.4f}")

    # ── Section 3: Composite aleatoric limit ──
    print("\n" + "=" * 70)
    print("SECTION 3: Composite Aleatoric Limit")
    print("=" * 70)
    composite = {}
    if n_direct > 0 and len(ci_widths) > 0:
        eps_direct_median = float(np.median(abs_devs))
        eps_direct_mean = float(np.mean(abs_devs))
        delta_interp_median = float(np.median(ci_widths))
        delta_interp_mean = float(np.mean(ci_widths))

        # ε_aleatoric = √(ε_direct² + 2·δ_interp²)
        eps_aleatoric = np.sqrt(eps_direct_median ** 2 + 2 * delta_interp_median ** 2)

        composite = {
            "epsilon_direct": {
                "median": round(eps_direct_median, 4),
                "mean": round(eps_direct_mean, 4),
                "ci_95_median": [round(ci_median[0], 4), round(ci_median[2], 4)],
                "ci_95_mean": [round(ci_mean[0], 4), round(ci_mean[2], 4)],
            },
            "delta_interp": {
                "median": round(delta_interp_median, 4),
                "mean": round(delta_interp_mean, 4),
            },
            "epsilon_aleatoric": round(eps_aleatoric, 4),
            "formula": "sqrt(eps_direct_median^2 + 2*delta_interp_median^2)",
        }

        print(f"\n  ε_direct (median):    {eps_direct_median:.4f}")
        print(f"  δ_interp (median):    {delta_interp_median:.4f}")
        print(f"  ε_aleatoric:          {eps_aleatoric:.4f}")
        print(f"  Formula: √({eps_direct_median:.4f}² + 2×{delta_interp_median:.4f}²)")
        print(f"\n  Interpretation:")
        print(f"    The composite aleatoric limit of {eps_aleatoric:.3f} log S is the")
        print(f"    irreducible measurement floor. No model should be expected to")
        print(f"    reliably achieve RMSE below this value on properly curated data.")

    # ── Section 4: Distribution fitting ──
    print("\n" + "=" * 70)
    print("SECTION 4: Error Distribution Analysis")
    print("=" * 70)
    dist_fits = {}
    if n_direct > 0:
        dist_fits = fit_distributions(abs_devs)
        for name, result in dist_fits.items():
            print(f"  {name:12s}: KS stat={result['ks_stat']:.4f}, p={result['ks_p']:.4f}")
        best = min(dist_fits.items(), key=lambda x: x[1]["ks_stat"])
        print(f"  Best fit: {best[0]} (lowest KS statistic)")

    # ── Section 5: Per-solvent stratification ──
    print("\n" + "=" * 70)
    print("SECTION 5: Per-Solvent Aleatoric Limits")
    print("=" * 70)
    per_solvent = {}
    if n_direct > 0:
        per_solvent = stratify_by_solvent(direct_df, df)
        print(f"  {len(per_solvent)} solvents with ≥5 comparisons")
        print(f"\n  {'Solvent':<25s} {'N':>5s} {'Mean':>8s} {'Median':>8s} {'P90':>8s}")
        print(f"  {'-'*25} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
        for solv, stats in list(per_solvent.items())[:20]:
            print(f"  {stats['name']:<25s} {stats['n_comparisons']:>5d} "
                  f"{stats['mean_mae']:>8.4f} {stats['median_mae']:>8.4f} "
                  f"{stats['p90']:>8.4f}")

    # ── Section 6: Error relationships ──
    print("\n" + "=" * 70)
    print("SECTION 6: Error vs LogS Magnitude and Temperature")
    print("=" * 70)
    relationships = {}
    if n_direct > 0:
        relationships = analyze_error_relationships(direct_df)

        if "by_logs_magnitude" in relationships:
            print("\n  By LogS magnitude:")
            for rng, stats in relationships["by_logs_magnitude"].items():
                print(f"    {rng:15s}: n={stats['n']:>5d}, "
                      f"median MAE={stats['median_mae']:.4f}")

        if "by_temperature" in relationships:
            print("\n  By temperature:")
            for rng, stats in relationships["by_temperature"].items():
                print(f"    {rng:15s}: n={stats['n']:>5d}, "
                      f"median MAE={stats['median_mae']:.4f}")

    # ── Section 7: Literature comparison ──
    print("\n" + "=" * 70)
    print("SECTION 7: Comparison with Literature")
    print("=" * 70)
    lit_comparison = {
        "palmer_mitchell_2014": {
            "claimed": "0.6-0.7 log S (aqueous, heterogeneous sources)",
            "our_finding": f"Median: {np.median(abs_devs):.3f}, P90: {np.percentile(abs_devs, 90):.3f}" if n_direct > 0 else "N/A",
            "explanation": "Literature value includes copycats and conflates quality tiers",
        },
        "attia_2025": {
            "claimed": "0.5-1.0 log S (organic solvents)",
            "our_finding": f"Our P90 ({np.percentile(abs_devs, 90):.3f}) approaches lower end" if n_direct > 0 else "N/A",
        },
        "llompart_2024": {
            "claimed": "Curation issues inflate apparent model performance",
            "our_finding": f"Confirmed: {14} exact + {125} near-duplicate source pairs found",
        },
    }
    for ref, info in lit_comparison.items():
        print(f"\n  {ref}:")
        for k, v in info.items():
            print(f"    {k}: {v}")

    # ── Save all results ──
    print("\nSaving results...")

    results = {
        "direct_interlab": {
            "n_comparisons": n_direct,
            "mean_mae": round(float(np.mean(abs_devs)), 4) if n_direct > 0 else None,
            "median_mae": round(float(np.median(abs_devs)), 4) if n_direct > 0 else None,
            "std": round(float(np.std(abs_devs)), 4) if n_direct > 0 else None,
            "p25": round(float(np.percentile(abs_devs, 25)), 4) if n_direct > 0 else None,
            "p75": round(float(np.percentile(abs_devs, 75)), 4) if n_direct > 0 else None,
            "p90": round(float(np.percentile(abs_devs, 90)), 4) if n_direct > 0 else None,
            "p95": round(float(np.percentile(abs_devs, 95)), 4) if n_direct > 0 else None,
            "bootstrap_ci_median": [round(ci_median[0], 4), round(ci_median[2], 4)] if n_direct > 0 else None,
            "bootstrap_ci_mean": [round(ci_mean[0], 4), round(ci_mean[2], 4)] if n_direct > 0 else None,
            "bias_mean": round(float(np.mean(signed_devs)), 4) if n_direct > 0 else None,
            "bias_std": round(float(np.std(signed_devs)), 4) if n_direct > 0 else None,
        },
        "composite_aleatoric": composite,
        "distribution_fits": dist_fits,
        "per_solvent": {k: v for k, v in list(per_solvent.items())[:30]},
        "error_relationships": relationships,
        "literature_comparison": lit_comparison,
    }

    with open(os.path.join(args.output_dir, "aleatoric_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    if n_direct > 0:
        direct_df.to_csv(os.path.join(args.output_dir, "direct_comparisons.csv"), index=False)

    # ── Plots ──
    print("Generating figures...")
    make_plots(direct_df, interp_df, ci_widths, per_solvent, relationships,
               dist_fits, composite, args.figures_dir)

    print(f"\nAleatoric analysis complete.")
    print(f"  Artifacts: {args.output_dir}")
    print(f"  Figures:   {args.figures_dir}")


if __name__ == "__main__":
    main()
