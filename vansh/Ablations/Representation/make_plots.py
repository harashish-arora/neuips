#!/usr/bin/env python
"""
Plot the representation-ablation results from results/<featurizer>/summary.json.

Produces three things:

  figures/representation_<metric>.png       Grouped bar chart, one bar per
                                            featurizer for each split.
  figures/representation_panel_<metric>.png Per-split bar charts side-by-side.
  results/representation_table_<metric>.csv Long-format CSV for the paper.

Usage:
  python make_plots.py                       # default: RMSE
  python make_plots.py --metric MAE          # plot a different metric
  python make_plots.py --metrics RMSE MAE PS_RMSE   # write all three at once
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

EVAL_SPLITS = ["eval", "ood", "sc3_gold", "sc3_silver", "sc3_bronze"]

# Featurizer display labels.  ``n_features_*`` tags get filled from each
# summary's diagnostics so the bar label shows the dimensionality.
FEATURIZER_LABELS = {
    "rdkit":        "RDKit 2D desc.",
    "morgan":       "Morgan ECFP4 (1024)",
    "dissolvr":     "Dissolvr (RDKit+MOSE+Joback+Abr.)",
    "mordred":      "Mordred 2D",
    "maccs":        "MACCS keys",
    "atompair":     "Atom-Pair FP (1024)",
    "abraham_only": "Abraham-only (5 LSER)",
}

# Distinct family-coded colours for the bars.
FEATURIZER_COLORS = {
    "rdkit":        "#1f77b4",
    "dissolvr":     "#2ca02c",
    "mordred":      "#9467bd",
    "morgan":       "#ff7f0e",
    "atompair":     "#d62728",
    "maccs":        "#8c564b",
    "abraham_only": "#7f7f7f",
}

PLOT_FEATURIZERS = list(FEATURIZER_LABELS.keys())


def _load_summaries(only: list | None = None) -> dict:
    """Return {featurizer: summary dict} for every results/<f>/summary.json."""
    summaries: dict = {}
    if not RESULTS_DIR.exists():
        return summaries
    if only is not None:
        for f in only:
            sp = RESULTS_DIR / f / "summary.json"
            if sp.exists():
                with open(sp) as fh:
                    summaries[f] = json.load(fh)
        return summaries
    for d in sorted(RESULTS_DIR.iterdir()):
        sp = d / "summary.json"
        if sp.exists():
            with open(sp) as fh:
                summaries[d.name] = json.load(fh)
    return summaries


def _value(summary: dict, split: str, metric: str) -> tuple[float, float, int]:
    """Return (mean, std, n) for a (split, metric); NaN/0/0 if missing."""
    agg = summary.get("aggregated", {}).get(split, {})
    m = agg.get(f"{metric}_mean", float("nan"))
    s = agg.get(f"{metric}_std", 0.0)
    n = agg.get(f"{metric}_n", 0)
    return float(m), float(s), int(n)


def _plot_grouped(summaries: dict, splits: list, metric: str,
                  ascending: bool, out_path: Path):
    """One figure: grouped bars (one group per featurizer, one bar per split)."""
    feats = [f for f in PLOT_FEATURIZERS if f in summaries]
    if not feats:
        print("  No featurizers to plot.")
        return
    n_groups = len(feats)
    n_bars = len(splits)
    width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_groups + 2), 5.5))
    cmap = plt.cm.tab10
    for i, sn in enumerate(splits):
        means, stds = [], []
        for f in feats:
            m, s, _ = _value(summaries[f], sn, metric)
            means.append(m); stds.append(s)
        bars = ax.bar(x + (i - (n_bars - 1) / 2) * width, means, width,
                      yerr=stds, label=sn, color=cmap(i),
                      capsize=3, edgecolor="black", linewidth=0.5)
        # annotate small mean number above each bar
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=90 if n_bars > 3 else 0)

    # Sort labels for the x-tick text by ``ascending`` of the *eval* split mean
    eval_means = [_value(summaries[f], "eval", metric)[0] for f in feats]
    order = np.argsort(eval_means)
    if not ascending:
        order = order[::-1]
    feats_sorted = [feats[i] for i in order]

    # Re-do the bars in the sorted order so the figure is easier to read.
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * n_groups + 2), 5.5))
    x = np.arange(n_groups)
    for i, sn in enumerate(splits):
        means, stds = [], []
        for f in feats_sorted:
            m, s, _ = _value(summaries[f], sn, metric)
            means.append(m); stds.append(s)
        bars = ax.bar(x + (i - (n_bars - 1) / 2) * width, means, width,
                      yerr=stds, label=sn, color=cmap(i),
                      capsize=3, edgecolor="black", linewidth=0.5)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7,
                        rotation=90 if n_bars > 3 else 0)

    ax.set_xticks(x)
    tick_labels = []
    for f in feats_sorted:
        n_feat = summaries[f].get("diagnostics", {}).get("_n_features")
        suffix = f"\n({int(n_feat)} feats)" if n_feat else ""
        tick_labels.append(f"{FEATURIZER_LABELS.get(f, f)}{suffix}")
    ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f"Representation ablation (LightGBM, fixed HPs):  {metric} by featurizer\n"
                 "(sorted by eval-split " + ("ascending" if ascending else "descending") + ")")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="split", loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _plot_split_panel(summaries: dict, splits: list, metric: str,
                      ascending: bool, out_path: Path):
    """One panel per split: all featurizers ranked left -> right by performance."""
    feats = [f for f in PLOT_FEATURIZERS if f in summaries]
    if not feats:
        return

    n_panels = len(splits)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5.0), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, sn in zip(axes, splits):
        triples = []
        for f in feats:
            m, s, _ = _value(summaries[f], sn, metric)
            triples.append((f, m, s))
        triples.sort(key=lambda t: (np.inf if np.isnan(t[1]) else t[1]),
                     reverse=not ascending)
        names = [t[0] for t in triples]
        means = [t[1] for t in triples]
        stds = [t[2] for t in triples]
        colors = [FEATURIZER_COLORS.get(n, "#333") for n in names]
        x = np.arange(len(names))
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=3,
                      edgecolor="black", linewidth=0.5)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURIZER_LABELS.get(n, n) for n in names],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{sn}  ({metric})")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Representation ablation: {metric} per split (LightGBM fixed)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _write_csv(summaries: dict, splits: list, metric: str, out_path: Path):
    feats = [f for f in PLOT_FEATURIZERS if f in summaries]
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["featurizer", "n_features", "split",
                    f"{metric}_mean", f"{metric}_std", "n_seeds"])
        for f in feats:
            n_feat = summaries[f].get("diagnostics", {}).get("_n_features", "")
            for sn in splits:
                m, s, n = _value(summaries[f], sn, metric)
                if np.isnan(m):
                    continue
                w.writerow([f, n_feat, sn, m, s, n])
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="RMSE",
                        help="Single metric to plot.  Use --metrics for multiple.")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="If set, generate plots/CSVs for each of these metrics "
                             "(e.g. --metrics RMSE MAE R2 PS_RMSE).")
    parser.add_argument("--splits", nargs="+", default=EVAL_SPLITS,
                        choices=EVAL_SPLITS,
                        help="Splits to include (default: all 5).")
    parser.add_argument("--featurizers", nargs="+", default=PLOT_FEATURIZERS,
                        help="Featurizers to include in the plot.")
    parser.add_argument("--all", action="store_true",
                        help="Use every featurizer that has a summary, ignore --featurizers.")
    args = parser.parse_args()

    summaries = _load_summaries(only=None if args.all else args.featurizers)
    if not summaries:
        print(f"No summaries under {RESULTS_DIR}. Run run_representation.py first.")
        return

    print("Featurizers with summaries:")
    for f, s in summaries.items():
        n_seeds = len(s.get("seeds", []))
        n_feat = s.get("diagnostics", {}).get("_n_features", "?")
        print(f"  {f:14s}  n_seeds={n_seeds}  n_features={n_feat}")

    metrics = args.metrics or [args.metric]
    # For RMSE/MAE/PS_RMSE smaller is better -> ascending sort.
    ascending_for = {
        "RMSE": True, "MAE": True, "PS_RMSE": True, "Z_RMSE": True,
        "R2": False, "PS_R2": False, "f_aleatoric": False,
    }
    for metric in metrics:
        ascending = ascending_for.get(metric, True)
        _plot_grouped(summaries, args.splits, metric, ascending,
                      FIG_DIR / f"representation_grouped_{metric}.png")
        _plot_split_panel(summaries, args.splits, metric, ascending,
                          FIG_DIR / f"representation_panel_{metric}.png")
        _write_csv(summaries, args.splits, metric,
                   RESULTS_DIR / f"representation_table_{metric}.csv")


if __name__ == "__main__":
    main()
