#!/usr/bin/env python
"""
Q4 analysis: how much more data, until smarter deeper models can perform
close to the aleatoric limit?

Methodology (follows Chen et al. NeurIPS 2023, Attia et al. Nat. Commun. 2025,
"Scaling Laws for Neural Material Models" 2025):

For each model, fit the standard saturating power law

    RMSE(N) = a * N^(-b) + c

where:
  - N = training set size (rows)
  - a > 0  : data prefactor (RMSE at the smallest N modeled)
  - b > 0  : scaling exponent (larger b = faster gains from more data)
  - c >= 0 : irreducible floor (aleatoric + representation bias)

We anchor `c` against an externally-estimated aleatoric RMSE from
duplicate (solute, solvent, temperature) measurements in the SC3
benchmark dataset.

For each model we then answer:
  Q4a. What is the model's intrinsic floor c_model?
  Q4b. How much additional data N* would close 95% of the gap to c_model?
  Q4c. How much data N** would reach within 0.05 logS of the *aleatoric*
       floor c_aleatoric (capped at infinity if c_model > c_aleatoric)?

Outputs:
  figures/q4_scaling_law_eval.png    log-log fit + extrapolation
  figures/q4_scaling_law_panel.png   eval / ood / sc3_gold side by side
  results/q4_summary.json            Fitted parameters and N* / N** per model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VANSH_ROOT = HERE.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.data import load_all_splits  # noqa: E402

RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

EVAL_SPLITS = ["eval", "ood", "sc3_gold"]

PLOT_METHODS = ["lgb_rdkit", "fastprop", "fastprop_big", "fastprop_xl"]

METHOD_LABELS = {
    "lgb_rdkit": "LightGBM (RDKit)",
    "fastprop": "FastProp (310K params)",
    "fastprop_big": "FastProp-Big (1.7M)",
    "fastprop_xl": "FastProp-XL (9M)",
}
METHOD_COLORS = {
    "lgb_rdkit": "#1f77b4",
    "fastprop": "#ffbb78",
    "fastprop_big": "#ff7f0e",
    "fastprop_xl": "#d62728",
}


# ---------------------------------------------------------------------------
# Aleatoric floor estimation
# ---------------------------------------------------------------------------

def _duplicate_pair_stats(df, T_decimals: int = 0) -> dict:
    """Estimate the irreducible (aleatoric) error floor from duplicate
    (Solute, Solvent, round(T)) measurements in one DataFrame.

    Method
    ------
    For every duplicate group g with measurements {y_i}, the per-observation
    residual against an oracle that predicts the group mean mu_g is
    (y_i - mu_g).  Pooled across all duplicate observations, this gives:

        aleatoric_RMSE = sqrt( mean_i ( y_i - mu_g(i) )^2 )
        aleatoric_MAE  = mean_i | y_i - mu_g(i) |

    These are directly comparable to the model RMSE and MAE on the same
    split: they bound from below what *any* model (regardless of capacity
    or training-set size) can achieve on this data.  This is the standard
    estimator (Attia et al., Nat. Commun. 2025; "interlaboratory dispersion").

    We weight by *observation*, not by group, so that large duplicate
    groups (more replicates) carry proportionally more weight, just like
    they would in a model evaluation.
    """
    df = df.copy()
    df["T_round"] = df["Temperature"].round(T_decimals)
    sq_resids, abs_resids = [], []
    pair_stds = []  # Bessel-corrected per-group stds, used only for diagnostics
    n_obs_total = 0
    n_groups = 0
    for _, vals in df.groupby(["Solute", "Solvent", "T_round"])["LogS"]:
        v = vals.values
        if len(v) >= 2:
            mu = v.mean()
            sq_resids.extend((v - mu) ** 2)
            abs_resids.extend(np.abs(v - mu))
            pair_stds.append(float(np.std(v, ddof=1)))
            n_obs_total += len(v)
            n_groups += 1
    if not sq_resids:
        return {"n_triples": 0}
    sq = np.asarray(sq_resids)
    ab = np.asarray(abs_resids)
    ps = np.asarray(pair_stds)
    return {
        "n_triples": n_groups,
        "n_rows_in_dups": n_obs_total,
        "rmse_floor": float(np.sqrt(sq.mean())),
        "mae_floor":  float(ab.mean()),
        # Per-group Bessel std summary stats (helpful for spotting outliers)
        "std_p50": float(np.percentile(ps, 50)),
        "std_p90": float(np.percentile(ps, 90)),
        "std_p99": float(np.percentile(ps, 99)),
        "std_max": float(ps.max()),
    }


def estimate_aleatoric(splits) -> dict:
    """Estimate the aleatoric (irreducible) RMSE floor for each evaluation
    split.  Two parallel estimates per split:

      - duplicate_pairs : pooled std-dev across (Solute, Solvent, T) triples
        that appear more than once in the same split.  Direct measurement
        of inter-source experimental noise.
      - sc3_*_sigma     : sigma-quadrature from the curator-supplied
        per-row sigma (only available on the SC3 tiers).
    """
    out = {"per_split": {}}

    for s in ["train", "eval", "ood", "sc3_gold", "sc3_silver", "sc3_bronze"]:
        if s not in splits:
            continue
        per_split = {"duplicate_pairs": _duplicate_pair_stats(splits[s])}
        df = splits[s]
        if "Uncertainty" in df.columns:
            unc = df["Uncertainty"].values
            valid = unc[~np.isnan(unc) & (unc > 0)]
            if len(valid) > 0:
                per_split["sigma_quadrature"] = {
                    "n_with_unc": int(len(valid)),
                    "sigma_p50": float(np.median(valid)),
                    "sigma_mean": float(np.mean(valid)),
                    "rmse_floor": float(np.sqrt(np.mean(valid ** 2))),
                }
        out["per_split"][s] = per_split

    # Combined-pool inter-source aleatoric (legacy / robustness check).
    big = pd.concat([splits[s] for s in ["train", "eval", "ood"]],
                    ignore_index=True)
    out["combined_train_eval_ood_duplicates"] = _duplicate_pair_stats(big)

    # Convenience top-level shortcuts used elsewhere in the script.
    if "eval" in out["per_split"] and "duplicate_pairs" in out["per_split"]["eval"]:
        out["duplicate_pairs"] = out["per_split"]["eval"]["duplicate_pairs"].get("rmse_floor")
    if "sc3_gold" in out["per_split"] and "sigma_quadrature" in out["per_split"]["sc3_gold"]:
        out["sc3_gold_sigma"] = out["per_split"]["sc3_gold"]["sigma_quadrature"].get("rmse_floor")
    return out


def floor_for_split(aleatoric: dict, split: str) -> float | None:
    """Pick the most appropriate aleatoric floor for one evaluation split.

    Preference order:
      1. duplicate-pair pooled RMSE measured *on that same split*
      2. sc3 sigma-quadrature floor (only valid for sc3_* splits)
      3. None
    """
    per = aleatoric.get("per_split", {}).get(split, {})
    dup = per.get("duplicate_pairs", {})
    if dup.get("rmse_floor") is not None:
        return float(dup["rmse_floor"])
    sig = per.get("sigma_quadrature", {})
    if sig.get("rmse_floor") is not None:
        return float(sig["rmse_floor"])
    return None


# ---------------------------------------------------------------------------
# Scaling-law fit: RMSE(N) = a * N^(-b) + c
# ---------------------------------------------------------------------------

def fit_scaling_law(n_train: np.ndarray, rmse: np.ndarray,
                    fix_c: float | None = None) -> dict:
    """Fit RMSE(N) = a * N^(-b) + c with non-linear least squares.

    If `fix_c` is given, c is held fixed to that value (e.g. an externally
    estimated aleatoric floor).
    """
    from scipy.optimize import curve_fit

    n = np.asarray(n_train, dtype=float)
    r = np.asarray(rmse, dtype=float)
    mask = np.isfinite(n) & np.isfinite(r) & (n > 0) & (r > 0)
    n, r = n[mask], r[mask]
    if len(n) < 3:
        return {"ok": False, "reason": f"only {len(n)} valid points"}

    # Initial guesses from the endpoints: at small N, RMSE ~ a * N^-b + c.
    a0 = float(r.max() - r.min()) * float(n.min()) ** 0.3
    b0 = 0.3
    c0 = float(r.min()) * 0.9 if fix_c is None else fix_c

    try:
        if fix_c is None:
            def _f(n, a, b, c):
                return a * np.power(n, -b) + c
            p0 = [a0, b0, c0]
            bounds = ([1e-6, 0.01, 0.0], [1e6, 5.0, max(r.max(), 1.0)])
            popt, pcov = curve_fit(_f, n, r, p0=p0, bounds=bounds, maxfev=20000)
            a, b, c = popt
        else:
            def _f(n, a, b):
                return a * np.power(n, -b) + fix_c
            p0 = [a0, b0]
            bounds = ([1e-6, 0.01], [1e6, 5.0])
            popt, pcov = curve_fit(_f, n, r, p0=p0, bounds=bounds, maxfev=20000)
            a, b = popt
            c = fix_c
    except Exception as e:
        return {"ok": False, "reason": str(e)}

    pred = a * np.power(n, -b) + c
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return {
        "ok": True, "a": float(a), "b": float(b), "c": float(c),
        "r2": float(r2), "n_points": int(len(n)),
        "fix_c": fix_c is not None,
    }


def n_required(a: float, b: float, c: float, target_rmse: float) -> float | None:
    """Solve a * N^(-b) + c = target_rmse for N (>0).  None if unreachable."""
    if target_rmse <= c:
        return None
    return float((a / (target_rmse - c)) ** (1.0 / b))


# ---------------------------------------------------------------------------
# Summary loader (mean RMSE + n_train per fraction per method)
# ---------------------------------------------------------------------------

def load_curve(method: str, split: str):
    sp = RESULTS_DIR / method / "summary.json"
    if not sp.exists():
        return None
    s = json.load(open(sp))
    rows = []
    for f_str, payload in sorted(s["by_fraction"].items(), key=lambda kv: float(kv[0])):
        n = payload.get("n_train_mean")
        agg = payload.get("aggregated", {}).get(split, {})
        rmse = agg.get("RMSE_mean")
        if n is None or rmse is None:
            continue
        rows.append((float(f_str), n, rmse))
    if not rows:
        return None
    arr = np.array(rows)
    return {"frac": arr[:, 0], "n": arr[:, 1], "rmse": arr[:, 2]}


def load_train_curve(method: str):
    """Return the model's BEST-EPOCH train-RMSE per training-set size,
    pulled from the per-fraction `diagnostics._train_RMSE_at_best` field
    written by `run_data_scaling.py`.  Only available for fastprop variants
    (LightGBM does not save a train-RMSE diagnostic).
    """
    sp = RESULTS_DIR / method / "summary.json"
    if not sp.exists():
        return None
    s = json.load(open(sp))
    rows = []
    for f_str, payload in sorted(s["by_fraction"].items(), key=lambda kv: float(kv[0])):
        diag = payload.get("diagnostics") or {}
        n = payload.get("n_train_mean")
        tr = diag.get("_train_RMSE_at_best")
        if n is None or tr is None:
            continue
        rows.append((float(f_str), n, tr))
    if not rows:
        return None
    arr = np.array(rows)
    return {"frac": arr[:, 0], "n": arr[:, 1], "rmse": arr[:, 2]}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _curve_and_fit_for(summary: dict, method: str, split: str):
    """Return (curve, fit) for either a val-split (eval/ood/sc3_*) or for
    the special 'train' panel.  Returns (None, None) if absent."""
    if split == "train":
        return (summary.get("train_curves", {}).get(method),
                summary.get("train_fits", {}).get(method))
    return (summary["curves"].get(f"{method}::{split}"),
            summary["fits"].get(f"{method}::{split}"))


def _panel_y_range(summary: dict, split: str, extra_pad: float = 0.04):
    """Compute a tight linear y-range for one split panel.

    Includes (a) all observed RMSE points, (b) the fitted floor `c`,
    (c) the per-split aleatoric floor.  Pads slightly on both ends.
    """
    floor_split = "train" if split == "train" else split
    floor = floor_for_split(summary["aleatoric"], floor_split)
    vals = []
    for method in PLOT_METHODS:
        curve, fit = _curve_and_fit_for(summary, method, split)
        if curve is None:
            continue
        vals.extend(curve["rmse"])
        if fit and fit.get("ok"):
            vals.append(fit["c"])
    if floor is not None:
        vals.append(floor)
    if not vals:
        return (0.0, 1.0)
    lo = max(0.0, min(vals) - extra_pad)
    hi = max(vals) + extra_pad
    return (lo, hi)


def plot_panel(summary: dict, splits_to_plot=("train", *EVAL_SPLITS),
               out_path: Path | None = None,
               extrapolate_decades: float = 1.5):
    """4-panel Q4 figure: train + eval + ood + sc3_gold.

    - X axis: log10 (training rows N).
    - Y axis: linear RMSE (logS), independent per panel for readability.
    - Solid line in the observed N range, dashed beyond (extrapolation).
    - Per-split aleatoric floor shown as a black dashed horizontal line.
    - Compact legend with one row per method (fit summary inline).
    """
    n_panels = len(splits_to_plot)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4.6 * n_panels, 4.8),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    panel_titles = {
        "train": "(a) train RMSE @ best epoch",
        "eval":  "(b) eval RMSE  (in-distribution)",
        "ood":   "(c) ood RMSE   (long-tail solvents)",
        "sc3_gold": "(d) sc3_gold RMSE  (consensus tier)",
    }

    for ax, split in zip(axes, splits_to_plot):
        # Plot scatter + fit per method
        method_handles = []
        for method in PLOT_METHODS:
            curve, fit = _curve_and_fit_for(summary, method, split)
            if curve is None:
                continue
            ns = np.asarray(curve["n"], dtype=float)
            rs = np.asarray(curve["rmse"], dtype=float)
            color = METHOD_COLORS.get(method, "k")
            label = METHOD_LABELS.get(method, method)

            ax.scatter(ns, rs, s=55, color=color, zorder=4,
                       edgecolor="white", linewidth=1.2)

            if fit and fit.get("ok"):
                a, b, c = fit["a"], fit["b"], fit["c"]
                n_min = float(ns.min())
                n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
                xs = np.logspace(np.log10(n_min), np.log10(n_max), 300)
                ys = a * np.power(xs, -b) + c
                in_range = xs <= float(ns.max()) * 1.05
                line, = ax.plot(xs[in_range], ys[in_range], "-",
                                color=color, lw=2.2, zorder=3)
                ax.plot(xs[~in_range], ys[~in_range], "--",
                        color=color, lw=1.4, alpha=0.55, zorder=3)
                method_handles.append(
                    (line, f"{label}   b={b:.2f},  c={c:.3f}")
                )

        # Per-split aleatoric floor
        floor_split = "train" if split == "train" else split
        split_floor = floor_for_split(summary["aleatoric"], floor_split)
        floor_handles = []
        if split_floor is not None:
            line = ax.axhline(split_floor, color="k", ls="--", lw=1.4,
                              alpha=0.85, zorder=2)
            floor_handles.append(
                (line, f"{floor_split} aleatoric floor = {split_floor:.3f}")
            )

        # Cosmetics
        y_lo, y_hi = _panel_y_range(summary, split)
        ax.set_xscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("training rows  $N$", fontsize=11)
        if split == splits_to_plot[0]:
            ax.set_ylabel("RMSE  (logS)", fontsize=11)
        ax.set_title(panel_titles.get(split, split), fontsize=12)
        ax.grid(True, which="major", axis="both", alpha=0.3)
        ax.grid(True, which="minor", axis="x", alpha=0.15)

        handles = method_handles + floor_handles
        if handles:
            ax.legend([h for h, _ in handles], [t for _, t in handles],
                      fontsize=8, loc="lower right", framealpha=0.92,
                      borderaxespad=0.4, handlelength=2.0)

    fig.suptitle(
        r"Q4: Data scaling fits   $\mathrm{RMSE}(N) = a \cdot N^{-b} + c$"
        "    (solid = observed range,  dashed = extrapolation,  black dashed = aleatoric floor)",
        fontsize=11.5, y=1.02)
    fig.tight_layout()
    out = out_path or (FIG_DIR / "q4_scaling_law_panel.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")

    # Single-panel eval-only version (keeps the same style)
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.2))
    plot_split_solo(ax2, summary, "eval", extrapolate_decades=extrapolate_decades)
    out2 = FIG_DIR / "q4_scaling_law_eval.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=170, bbox_inches="tight")
    plt.close(fig2)
    print(f"  wrote {out2}")

    # Train vs val overlay
    fig3, ax3 = plt.subplots(figsize=(8.5, 6))
    plot_train_vs_val(ax3, summary, extrapolate_decades=extrapolate_decades)
    out3 = FIG_DIR / "q4_train_vs_val.png"
    fig3.tight_layout()
    fig3.savefig(out3, dpi=170, bbox_inches="tight")
    plt.close(fig3)
    print(f"  wrote {out3}")


def plot_split_solo(ax, summary: dict, split: str,
                     extrapolate_decades: float = 1.5):
    handles = []
    for method in PLOT_METHODS:
        curve, fit = _curve_and_fit_for(summary, method, split)
        if curve is None:
            continue
        ns = np.asarray(curve["n"], dtype=float)
        rs = np.asarray(curve["rmse"], dtype=float)
        color = METHOD_COLORS.get(method, "k")
        label = METHOD_LABELS.get(method, method)
        ax.scatter(ns, rs, s=70, color=color, zorder=4,
                   edgecolor="white", linewidth=1.3)
        if fit and fit.get("ok"):
            a, b, c = fit["a"], fit["b"], fit["c"]
            n_min = float(ns.min())
            n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
            xs = np.logspace(np.log10(n_min), np.log10(n_max), 300)
            ys = a * np.power(xs, -b) + c
            in_range = xs <= float(ns.max()) * 1.05
            line, = ax.plot(xs[in_range], ys[in_range], "-",
                            color=color, lw=2.2, zorder=3)
            ax.plot(xs[~in_range], ys[~in_range], "--",
                    color=color, lw=1.5, alpha=0.6, zorder=3)
            handles.append((line, f"{label}   b={b:.2f},  c={c:.3f}"))

    split_floor = floor_for_split(summary["aleatoric"], split)
    if split_floor is not None:
        line = ax.axhline(split_floor, color="k", ls="--", lw=1.5,
                          alpha=0.85, zorder=2)
        handles.append((line, f"{split} aleatoric floor = {split_floor:.3f}"))

    y_lo, y_hi = _panel_y_range(summary, split)
    ax.set_xscale("log")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("training rows  $N$", fontsize=12)
    ax.set_ylabel(f"{split}  RMSE  (logS)", fontsize=12)
    ax.set_title(
        rf"Q4 scaling-law fit on {split}:  $\mathrm{{RMSE}}(N) = a \cdot N^{{-b}} + c$",
        fontsize=12)
    ax.grid(True, which="major", axis="both", alpha=0.3)
    ax.grid(True, which="minor", axis="x", alpha=0.15)
    if handles:
        ax.legend([h for h, _ in handles], [t for _, t in handles],
                  fontsize=9.5, loc="lower right", framealpha=0.92,
                  borderaxespad=0.4)


def plot_train_vs_val(ax, summary: dict, val_split: str = "eval",
                       extrapolate_decades: float = 1.5):
    """Overlay each model's train RMSE@best (solid + circles) and
    val RMSE (dotted + squares) so the train-val gap is immediately
    visible.  Linear y-axis with tight bounds.
    """
    handles = []
    all_y = []

    for method in PLOT_METHODS:
        color = METHOD_COLORS.get(method, "k")
        label = METHOD_LABELS.get(method, method)

        train_curve = summary.get("train_curves", {}).get(method)
        val_curve = summary["curves"].get(f"{method}::{val_split}")

        if train_curve is not None:
            ns = np.asarray(train_curve["n"], dtype=float)
            rs = np.asarray(train_curve["rmse"], dtype=float)
            all_y.extend(rs)
            ax.scatter(ns, rs, s=55, color=color, zorder=4,
                       edgecolor="white", linewidth=1.2, marker="o")
            tfit = summary.get("train_fits", {}).get(method)
            if tfit and tfit.get("ok"):
                a, b, c = tfit["a"], tfit["b"], tfit["c"]
                all_y.append(c)
                n_min = float(ns.min())
                n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
                xs = np.logspace(np.log10(n_min), np.log10(n_max), 300)
                ys = a * np.power(xs, -b) + c
                in_range = xs <= float(ns.max()) * 1.05
                line, = ax.plot(xs[in_range], ys[in_range], "-",
                                color=color, lw=2.0, zorder=3)
                ax.plot(xs[~in_range], ys[~in_range], "--",
                        color=color, lw=1.4, alpha=0.55, zorder=3)
                handles.append(
                    (line, f"{label}  train   b={b:.2f},  c={c:.3f}")
                )

        if val_curve is not None:
            ns = np.asarray(val_curve["n"], dtype=float)
            rs = np.asarray(val_curve["rmse"], dtype=float)
            all_y.extend(rs)
            ax.scatter(ns, rs, s=55, color=color, alpha=0.55, zorder=3,
                       edgecolor="white", linewidth=1.2, marker="s")
            vfit = summary["fits"].get(f"{method}::{val_split}")
            if vfit and vfit.get("ok"):
                a, b, c = vfit["a"], vfit["b"], vfit["c"]
                all_y.append(c)
                n_min = float(ns.min())
                n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
                xs = np.logspace(np.log10(n_min), np.log10(n_max), 300)
                ys = a * np.power(xs, -b) + c
                in_range = xs <= float(ns.max()) * 1.05
                line, = ax.plot(xs[in_range], ys[in_range], ":",
                                color=color, lw=2.0, alpha=0.85, zorder=3)
                ax.plot(xs[~in_range], ys[~in_range], ":",
                        color=color, lw=1.3, alpha=0.5, zorder=3)
                handles.append(
                    (line, f"{label}  {val_split:<5}  b={b:.2f},  c={c:.3f}")
                )

    train_floor = floor_for_split(summary["aleatoric"], "train")
    val_floor = floor_for_split(summary["aleatoric"], val_split)
    if train_floor is not None:
        line = ax.axhline(train_floor, color="k", ls="--", lw=1.4, alpha=0.85)
        all_y.append(train_floor)
        handles.append((line, f"train aleatoric = {train_floor:.3f}"))
    if val_floor is not None and (train_floor is None
                                   or abs(val_floor - train_floor) > 0.02):
        line = ax.axhline(val_floor, color="0.4", ls="-.", lw=1.3, alpha=0.85)
        all_y.append(val_floor)
        handles.append((line, f"{val_split} aleatoric = {val_floor:.3f}"))

    if all_y:
        lo = max(0.0, min(all_y) - 0.04)
        hi = max(all_y) + 0.04
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_xlabel("training rows  $N$", fontsize=12)
    ax.set_ylabel("RMSE  (logS)", fontsize=12)
    ax.set_title(
        f"Q4: train RMSE@best (●, solid)  vs  {val_split} RMSE (■, dotted)",
        fontsize=12)
    ax.grid(True, which="major", axis="both", alpha=0.3)
    ax.grid(True, which="minor", axis="x", alpha=0.15)
    if handles:
        ax.legend([h for h, _ in handles], [t for _, t in handles],
                  fontsize=8.5, loc="lower right", framealpha=0.92,
                  ncol=1, borderaxespad=0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading splits & estimating aleatoric floor (per split)...")
    splits = load_all_splits(verbose=False)
    aleatoric = estimate_aleatoric(splits)

    print()
    print(f"{'split':<12} {'n_dup_triples':>14} {'n_dup_obs':>10} "
          f"{'aleatoric_RMSE':>15} {'aleatoric_MAE':>14}")
    print("-" * 75)
    for s, info in aleatoric["per_split"].items():
        dup = info.get("duplicate_pairs", {})
        if dup.get("n_triples"):
            print(f"{s:<12} {dup['n_triples']:>14d} {dup['n_rows_in_dups']:>10d} "
                  f"{dup['rmse_floor']:>15.4f} {dup['mae_floor']:>14.4f}")
        else:
            print(f"{s:<12} {'(no dup triples)':>14}")
    print()

    summary = {
        "aleatoric": aleatoric,
        "curves": {},
        "train_curves": {},
        "fits": {},
        "train_fits": {},
        "extrapolations": {},
    }

    # `train` is treated like an extra split for plotting purposes only.
    plot_panels = ["train", *EVAL_SPLITS]

    print(f"{'method':<14} {'split':<10} {'a':>10} {'b':>6} {'c_fit':>7} "
          f"{'c_floor':>8} {'r2':>5} {'N* (95% gap)':>14} {'N** (floor+0.05)':>18}")
    print("-" * 100)

    for method in PLOT_METHODS:
        # ---- Per-split (val/test) curves and fits ----
        for split in EVAL_SPLITS:
            curve = load_curve(method, split)
            if curve is None:
                continue
            summary["curves"][f"{method}::{split}"] = {
                "n": curve["n"].tolist(), "rmse": curve["rmse"].tolist(),
            }
            fit_free = fit_scaling_law(curve["n"], curve["rmse"])
            summary["fits"][f"{method}::{split}"] = fit_free
            if not fit_free.get("ok"):
                print(f"{method:<14} {split:<10}  fit failed: {fit_free.get('reason')}")
                continue

            a, b, c = fit_free["a"], fit_free["b"], fit_free["c"]
            r2 = fit_free["r2"]
            split_floor = floor_for_split(aleatoric, split)

            rmse_max = float(np.max(curve["rmse"]))
            t95 = c + 0.05 * (rmse_max - c)
            n_star_95 = n_required(a, b, c, t95)
            n_star_floor = None
            if split_floor is not None:
                t_floor = split_floor + 0.05
                n_star_floor = n_required(a, b, c, t_floor)

            summary["extrapolations"][f"{method}::{split}"] = {
                "rmse_at_max_N": float(curve["rmse"][-1]),
                "max_N": float(curve["n"][-1]),
                "fit_floor_c": c,
                "split_aleatoric_floor": split_floor,
                "gap_c_minus_floor": (c - split_floor) if split_floor is not None else None,
                "n_to_close_95pct_gap_to_model_floor": n_star_95,
                "target_rmse_for_95pct": t95,
                "n_to_reach_split_aleatoric_plus_0.05": n_star_floor,
                "target_rmse_for_split_aleatoric": (split_floor + 0.05) if split_floor is not None else None,
            }
            n_star_95_str = f"{n_star_95:.2e}" if n_star_95 else "  achieved"
            n_star_floor_str = (f"{n_star_floor:.2e}" if n_star_floor
                                else (" unreachable" if split_floor is not None else "    n/a"))
            split_floor_str = f"{split_floor:.4f}" if split_floor is not None else "  n/a"
            print(f"{method:<14} {split:<10} {a:>10.3f} {b:>6.3f} {c:>7.4f} "
                  f"{split_floor_str:>8} {r2:>5.2f} {n_star_95_str:>14} {n_star_floor_str:>18}")

        # ---- Train-RMSE curve + fit (only fastprop variants save this) ----
        train_curve = load_train_curve(method)
        if train_curve is None:
            continue
        summary["train_curves"][method] = {
            "n": train_curve["n"].tolist(), "rmse": train_curve["rmse"].tolist(),
        }
        fit_train = fit_scaling_law(train_curve["n"], train_curve["rmse"])
        summary["train_fits"][method] = fit_train
        if fit_train.get("ok"):
            a, b, c = fit_train["a"], fit_train["b"], fit_train["c"]
            print(f"{method:<14} {'train':<10} {a:>10.3f} {b:>6.3f} {c:>7.4f} "
                  f"{aleatoric['per_split']['train']['duplicate_pairs']['rmse_floor']:>8.4f} "
                  f"{fit_train['r2']:>5.2f} {' (train)':>14} {' (train)':>18}")

    out_json = RESULTS_DIR / "q4_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved {out_json}")

    print("\nGenerating panel plot...")
    plot_panel(summary)
    print("Done.")


if __name__ == "__main__":
    main()
