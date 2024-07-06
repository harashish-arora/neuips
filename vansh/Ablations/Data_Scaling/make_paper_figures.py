#!/usr/bin/env python
"""
Generate the publication-quality figures for the data-scaling ablation
section of the SC3 paper.  Writes both PDF (vector, paper-bound) and PNG
(rendered preview) versions directly into the paper figures directory.

Figures
-------
fig_data_scaling_curves.{pdf,png}
    3 panels (eval, ood, sc3_gold).  Per-method RMSE vs. training-set
    fraction with the per-split aleatoric floor as a horizontal reference.

fig_data_scaling_lawfits.{pdf,png}
    4 panels (train, eval, ood, sc3_gold).  Same data plus a power-law
    fit  RMSE(N) = a*N^{-b} + c  per method, extrapolated 1.5 decades
    past the largest observed N (dashed segments).  Linear y-axis,
    fit summary in each panel's legend.

fig_data_scaling_train_vs_val.{pdf,png}
    Single overlay panel: per-method train RMSE@best vs eval RMSE on
    one set of axes, both with their own power-law fits and aleatoric
    floors.  Makes the train/eval gap and overfitting story explicit.

The script reads the same JSON summaries that the live analysis script
(`q4_scaling_analysis.py`) writes, so it is always in sync with the
on-disk experimental results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

HERE = Path(__file__).resolve().parent
VANSH_ROOT = HERE.parent.parent
sys.path.insert(0, str(VANSH_ROOT))
sys.path.insert(0, str(HERE))

from q4_scaling_analysis import (  # noqa: E402
    estimate_aleatoric, fit_scaling_law, floor_for_split, load_curve,
    load_train_curve, n_required, EVAL_SPLITS, PLOT_METHODS,
)
from sc3_bench.data import load_all_splits  # noqa: E402

PAPER_FIG_DIR = Path("/DATATWO/users/solubility/Solubility/sc3_benchmark_data_curation_v2/paper/figures")
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot styling: NeurIPS-consistent
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.framealpha": 0.93,
    "legend.borderpad": 0.4,
    "lines.linewidth": 1.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
})

# Method styling -- matched across all three figures, family-coloured.
METHOD_LABEL = {
    "lgb_rdkit":    "LightGBM (RDKit)",
    "fastprop":     "FastProp (310K)",
    "fastprop_big": "FastProp-Big (1.7M)",
    "fastprop_xl":  "FastProp-XL (9M)",
}
METHOD_COLOR = {
    "lgb_rdkit":    "#1f77b4",  # blue: tabular tree family
    "fastprop":     "#ffbb78",  # light orange: small descriptor MLP
    "fastprop_big": "#ff7f0e",  # mid orange: medium descriptor MLP
    "fastprop_xl":  "#d62728",  # red: large descriptor MLP
}
METHOD_MARKER = {
    "lgb_rdkit":    "o",
    "fastprop":     "s",
    "fastprop_big": "D",
    "fastprop_xl":  "^",
}

PANEL_TITLES = {
    "train":    "train RMSE @ best epoch",
    "eval":     "eval RMSE   (in-distribution)",
    "ood":      "ood RMSE    (long-tail solvents)",
    "sc3_gold": "sc3_gold RMSE  (consensus tier)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, name: str) -> None:
    """Save fig as both PDF (paper-bound) and PNG (preview) in paper/figures/."""
    pdf = PAPER_FIG_DIR / f"{name}.pdf"
    png = PAPER_FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")


def _collect_curves(splits_to_plot, include_train: bool = False):
    """Gather the per-method, per-split curves and fits.  Anchors the fits
    to the same on-disk power-law fits so the paper figure agrees
    bit-for-bit with the q4_summary.json table.
    """
    out = {"curves": {}, "fits": {}, "train_curves": {}, "train_fits": {}}
    for m in PLOT_METHODS:
        for s in splits_to_plot:
            if s == "train":
                continue
            curve = load_curve(m, s)
            if curve is None:
                continue
            ns = curve["n"]
            rs = curve["rmse"]
            fit = fit_scaling_law(ns, rs)
            out["curves"][f"{m}::{s}"] = {"n": ns, "rmse": rs}
            out["fits"][f"{m}::{s}"] = fit
        if include_train:
            tc = load_train_curve(m)
            if tc is None:
                continue
            tfit = fit_scaling_law(tc["n"], tc["rmse"])
            out["train_curves"][m] = {"n": tc["n"], "rmse": tc["rmse"]}
            out["train_fits"][m] = tfit
    return out


def _curve_for(curves: dict, method: str, split: str):
    if split == "train":
        return curves["train_curves"].get(method), curves["train_fits"].get(method)
    return curves["curves"].get(f"{method}::{split}"), curves["fits"].get(f"{method}::{split}")


def _xticks_thousands(ax):
    """Rotate-free k/M formatting for the log-N axis."""
    def _fmt(x, _pos):
        if x >= 1e6:
            return f"{x/1e6:g}M"
        if x >= 1e3:
            return f"{x/1e3:g}k"
        return f"{x:g}"
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_fmt))


def _panel_y_range(curves: dict, split: str, aleatoric: dict,
                   pad: float = 0.04):
    floor_split = "train" if split == "train" else split
    floor = floor_for_split(aleatoric, floor_split)
    vals = []
    for m in PLOT_METHODS:
        curve, fit = _curve_for(curves, m, split)
        if curve is None:
            continue
        vals.extend(curve["rmse"])
        if fit and fit.get("ok"):
            vals.append(fit["c"])
    if floor is not None:
        vals.append(floor)
    if not vals:
        return (0.0, 1.0)
    lo = max(0.0, min(vals) - pad)
    hi = max(vals) + pad
    return (lo, hi)


# ---------------------------------------------------------------------------
# Figure 1: data scaling curves (no fit, eval/ood/sc3_gold)
# ---------------------------------------------------------------------------

def fig_scaling_curves(curves: dict, aleatoric: dict, name: str):
    """Three-panel data-scaling plot, one panel per evaluation split,
    points + connecting line per method, per-split aleatoric horizontal
    line.  The single most-cited figure in the paper.
    """
    splits = ["eval", "ood", "sc3_gold"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), sharey=False)

    for ax, split in zip(axes, splits):
        for m in PLOT_METHODS:
            curve, _fit = _curve_for(curves, m, split)
            if curve is None:
                continue
            ns = np.asarray(curve["n"], dtype=float)
            rs = np.asarray(curve["rmse"], dtype=float)
            ax.plot(ns, rs, "-", color=METHOD_COLOR[m], lw=1.6, alpha=0.9,
                    zorder=2)
            ax.scatter(ns, rs, marker=METHOD_MARKER[m], s=42,
                       color=METHOD_COLOR[m], edgecolor="white",
                       linewidth=0.9, zorder=4, label=METHOD_LABEL[m])

        floor = floor_for_split(aleatoric, split)
        if floor is not None:
            ax.axhline(floor, color="0.15", ls="--", lw=1.2, alpha=0.85,
                       zorder=1,
                       label=rf"aleatoric  $\varepsilon_A = {floor:.3f}$")

        y_lo, y_hi = _panel_y_range(curves, split, aleatoric)
        ax.set_xscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(left=2.5e3)
        _xticks_thousands(ax)

        ax.set_xlabel(r"training rows  $N$")
        if split == "eval":
            ax.set_ylabel(r"RMSE  ($\log_{10} S$)")
        ax.set_title(PANEL_TITLES[split], fontsize=9.5)
        ax.legend(loc="lower left", ncol=1, handletextpad=0.4)

    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 2: scaling-law fits with extrapolation
# ---------------------------------------------------------------------------

def fig_scaling_lawfits(curves: dict, aleatoric: dict, name: str,
                        extrapolate_decades: float = 1.5):
    """Four-panel power-law-fit plot.  Each panel: data points + per-method
    fit  RMSE(N) = a*N^{-b} + c, solid in observed range and dashed beyond,
    plus the per-split aleatoric floor.  The fit equation appears in
    the suptitle; per-method (b, c) appear in the legend.
    """
    splits = ["train", "eval", "ood", "sc3_gold"]
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.6), sharey=False)

    for ax, split in zip(axes, splits):
        legend_handles = []

        for m in PLOT_METHODS:
            curve, fit = _curve_for(curves, m, split)
            if curve is None:
                continue
            ns = np.asarray(curve["n"], dtype=float)
            rs = np.asarray(curve["rmse"], dtype=float)
            color = METHOD_COLOR[m]

            ax.scatter(ns, rs, marker=METHOD_MARKER[m], s=40, color=color,
                       edgecolor="white", linewidth=0.9, zorder=4)

            if fit and fit.get("ok"):
                a, b, c = fit["a"], fit["b"], fit["c"]
                n_min = float(ns.min())
                n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
                xs = np.logspace(np.log10(n_min), np.log10(n_max), 250)
                ys = a * np.power(xs, -b) + c
                in_obs = xs <= float(ns.max()) * 1.05
                line, = ax.plot(xs[in_obs], ys[in_obs], "-", color=color,
                                lw=1.7, zorder=3)
                ax.plot(xs[~in_obs], ys[~in_obs], "--", color=color,
                        lw=1.2, alpha=0.55, zorder=3)
                legend_handles.append(
                    (line,
                     rf"{METHOD_LABEL[m]}:  $b={b:.2f}$, $c={c:.3f}$")
                )

        floor_split = "train" if split == "train" else split
        floor = floor_for_split(aleatoric, floor_split)
        if floor is not None:
            line = ax.axhline(floor, color="0.15", ls="--", lw=1.2,
                              alpha=0.85, zorder=2)
            legend_handles.append(
                (line,
                 rf"aleatoric  $\varepsilon_A = {floor:.3f}$")
            )

        y_lo, y_hi = _panel_y_range(curves, split, aleatoric)
        ax.set_xscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(left=2.5e3)
        _xticks_thousands(ax)
        ax.set_xlabel(r"training rows  $N$")
        if split == "train":
            ax.set_ylabel(r"RMSE  ($\log_{10} S$)")
        ax.set_title(PANEL_TITLES[split], fontsize=9.5)
        if legend_handles:
            ax.legend([h for h, _ in legend_handles],
                      [t for _, t in legend_handles],
                      loc="lower right", handletextpad=0.4, handlelength=1.6)

    fig.suptitle(
        r"Power-law fits   $\mathrm{RMSE}(N) = a \cdot N^{-b} + c$"
        r"   (solid = observed range,  dashed = extrapolation,  "
        r"black dashed = per-split aleatoric floor)",
        fontsize=9.5, y=1.03)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 3: train vs eval overlay (overfitting diagnostic)
# ---------------------------------------------------------------------------

def fig_train_vs_val(curves: dict, aleatoric: dict, name: str,
                     val_split: str = "eval",
                     extrapolate_decades: float = 1.5):
    """Single-panel overlay of each model's train RMSE@best (circles, solid)
    and val RMSE on `val_split` (squares, dotted).  Both have their power-law
    fit drawn through.  The train/eval aleatoric floors are horizontal
    reference lines.  Makes the train-eval gap and overfitting story
    immediately readable.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    legend_handles = []

    for m in PLOT_METHODS:
        color = METHOD_COLOR[m]
        label = METHOD_LABEL[m]

        # Train side
        train_curve, tfit = _curve_for(curves, m, "train")
        if train_curve is not None:
            ns = np.asarray(train_curve["n"], dtype=float)
            rs = np.asarray(train_curve["rmse"], dtype=float)
            ax.scatter(ns, rs, marker="o", s=36, color=color,
                       edgecolor="white", linewidth=0.9, zorder=4)
            if tfit and tfit.get("ok"):
                a, b, c = tfit["a"], tfit["b"], tfit["c"]
                n_min = float(ns.min())
                n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
                xs = np.logspace(np.log10(n_min), np.log10(n_max), 300)
                ys = a * np.power(xs, -b) + c
                in_obs = xs <= float(ns.max()) * 1.05
                line, = ax.plot(xs[in_obs], ys[in_obs], "-", color=color,
                                lw=1.7, zorder=3)
                ax.plot(xs[~in_obs], ys[~in_obs], "--", color=color,
                        lw=1.2, alpha=0.55, zorder=3)
                legend_handles.append(
                    (line,
                     rf"{label}  train: $b={b:.2f}$, $c={c:.3f}$")
                )

        # Val side
        val_curve, vfit = _curve_for(curves, m, val_split)
        if val_curve is not None:
            ns = np.asarray(val_curve["n"], dtype=float)
            rs = np.asarray(val_curve["rmse"], dtype=float)
            ax.scatter(ns, rs, marker="s", s=36, color=color, alpha=0.55,
                       edgecolor="white", linewidth=0.9, zorder=3)
            if vfit and vfit.get("ok"):
                a, b, c = vfit["a"], vfit["b"], vfit["c"]
                n_min = float(ns.min())
                n_max = float(ns.max()) * (10.0 ** extrapolate_decades)
                xs = np.logspace(np.log10(n_min), np.log10(n_max), 300)
                ys = a * np.power(xs, -b) + c
                in_obs = xs <= float(ns.max()) * 1.05
                line, = ax.plot(xs[in_obs], ys[in_obs], ":", color=color,
                                lw=1.7, alpha=0.85, zorder=3)
                ax.plot(xs[~in_obs], ys[~in_obs], ":", color=color, lw=1.2,
                        alpha=0.45, zorder=3)
                legend_handles.append(
                    (line,
                     rf"{label}  {val_split}: $b={b:.2f}$, $c={c:.3f}$")
                )

    train_floor = floor_for_split(aleatoric, "train")
    val_floor = floor_for_split(aleatoric, val_split)
    if train_floor is not None:
        line = ax.axhline(train_floor, color="0.15", ls="--", lw=1.3,
                          alpha=0.85, zorder=2)
        legend_handles.append(
            (line, rf"aleatoric  $\varepsilon_A^{{\mathrm{{train}}}} = {train_floor:.3f}$")
        )
    if val_floor is not None and (train_floor is None
                                   or abs(val_floor - train_floor) > 0.02):
        line = ax.axhline(val_floor, color="0.45", ls="-.", lw=1.2,
                          alpha=0.85, zorder=2)
        legend_handles.append(
            (line, rf"aleatoric  $\varepsilon_A^{{\mathrm{{{val_split}}}}} = {val_floor:.3f}$")
        )

    # Y range across all plotted points + fit asymptotes + floors
    yvals = []
    for m in PLOT_METHODS:
        for s in ("train", val_split):
            curve, fit = _curve_for(curves, m, s)
            if curve is None:
                continue
            yvals.extend(curve["rmse"])
            if fit and fit.get("ok"):
                yvals.append(fit["c"])
    if train_floor is not None:
        yvals.append(train_floor)
    if val_floor is not None:
        yvals.append(val_floor)
    if yvals:
        ax.set_ylim(max(0.0, min(yvals) - 0.04), max(yvals) + 0.04)

    ax.set_xscale("log")
    ax.set_xlim(left=2.5e3)
    _xticks_thousands(ax)
    ax.set_xlabel(r"training rows  $N$")
    ax.set_ylabel(r"RMSE  ($\log_{10} S$)")
    ax.set_title(
        f"Train RMSE@best (circles, solid) vs. {val_split} RMSE "
        "(squares, dotted)  --  "
        "power-law fits and per-split aleatoric floors",
        fontsize=9.5)
    ax.legend([h for h, _ in legend_handles], [t for _, t in legend_handles],
              loc="lower right", ncol=1, handletextpad=0.5,
              handlelength=1.8)

    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading SC3 splits and computing per-split aleatoric floors...")
    splits = load_all_splits(verbose=False)
    aleatoric = estimate_aleatoric(splits)

    print("\nPer-split aleatoric RMSE floors (from duplicate (s,v,T) measurements):")
    for s in ["train", "eval", "ood", "sc3_gold"]:
        f = floor_for_split(aleatoric, s)
        print(f"  {s:10s}  rmse={f:.4f}  logS")

    print("\nLoading scaling curves and refitting power laws...")
    curves = _collect_curves(("train", *EVAL_SPLITS), include_train=True)

    print("\nGenerating publication figures into paper/figures/ ...")
    fig_scaling_curves(curves, aleatoric,
                       name="fig_data_scaling_curves")
    fig_scaling_lawfits(curves, aleatoric,
                        name="fig_data_scaling_lawfits")
    fig_train_vs_val(curves, aleatoric,
                     name="fig_data_scaling_train_vs_val")
    print("\nDone.")


if __name__ == "__main__":
    main()
