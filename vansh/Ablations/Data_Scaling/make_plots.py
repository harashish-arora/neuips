#!/usr/bin/env python
"""
Plot the data-scaling curves from the per-run JSONs in results/.

Reads results/<method>/summary.json and produces:

  figures/data_scaling_eval.png       RMSE vs. data fraction on `eval`
  figures/data_scaling_ood.png        RMSE vs. data fraction on `ood`
  figures/data_scaling_sc3_gold.png   RMSE vs. data fraction on `sc3_gold`
  figures/data_scaling_panel.png      3-panel side-by-side (eval, ood, sc3_gold)
  results/data_scaling_table.csv      Long-format table (method, fraction, split, RMSE_mean, RMSE_std)

Usage:
    python make_plots.py                         # default: RMSE
    python make_plots.py --metric MAE            # plot a different metric
    python make_plots.py --log-x                 # log-scale x-axis (rows)
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

EVAL_SPLITS = ["eval", "ood", "sc3_gold"]

METHOD_LABELS = {
    "lgb_rdkit": "LightGBM (RDKit)",
    "fastprop": "FastProp (~310K params)",
    "fastprop_big": "FastProp-Big (~1.7M params)",
    "fastprop_xl": "FastProp-XL (~9M params)",
    "molmerger": "MolMerger (AttentiveFP)",
}
METHOD_COLORS = {
    "lgb_rdkit": "#1f77b4",
    "fastprop": "#ffbb78",        # light orange
    "fastprop_big": "#ff7f0e",    # mid orange
    "fastprop_xl": "#d62728",     # red
    "molmerger": "#2ca02c",
}
METHOD_MARKERS = {
    "lgb_rdkit": "o",
    "fastprop": "s",
    "fastprop_big": "D",
    "fastprop_xl": "P",
    "molmerger": "^",
}

# Methods to include in the plot (in this order). Drop molmerger by default.
PLOT_METHODS = ["lgb_rdkit", "fastprop", "fastprop_big", "fastprop_xl"]


def _load_summaries(only: list | None = None) -> dict:
    """Return {method: summary dict} for every results/<method>/summary.json.

    If `only` is given, restrict to that ordered list of methods (and preserve order).
    """
    summaries = {}
    if not RESULTS_DIR.exists():
        return summaries
    if only is not None:
        for m in only:
            sp = RESULTS_DIR / m / "summary.json"
            if sp.exists():
                with open(sp) as f:
                    summaries[m] = json.load(f)
        return summaries
    for d in sorted(RESULTS_DIR.iterdir()):
        sp = d / "summary.json"
        if sp.exists():
            with open(sp) as f:
                summaries[d.name] = json.load(f)
    return summaries


def _curve(summary: dict, split: str, metric: str):
    """Return (fractions, n_train, mean, std) arrays for a method/split/metric."""
    fracs, ns, means, stds = [], [], [], []
    for f_str, payload in sorted(summary["by_fraction"].items(), key=lambda kv: float(kv[0])):
        agg = payload.get("aggregated", {}).get(split, {})
        m = agg.get(f"{metric}_mean")
        if m is None:
            continue
        fracs.append(float(f_str))
        ns.append(payload.get("n_train_mean") or 0)
        means.append(m)
        stds.append(agg.get(f"{metric}_std", 0.0))
    return np.asarray(fracs), np.asarray(ns), np.asarray(means), np.asarray(stds)


def _diag_curve(summary: dict, key: str):
    """Return (fractions, values) for a top-level diagnostic key."""
    fracs, vals = [], []
    for f_str, payload in sorted(summary["by_fraction"].items(), key=lambda kv: float(kv[0])):
        v = payload.get("diagnostics", {}).get(key)
        if v is None:
            continue
        fracs.append(float(f_str))
        vals.append(v)
    return np.asarray(fracs), np.asarray(vals)


def _plot_split(ax, summaries: dict, split: str, metric: str,
                use_n_train: bool, log_x: bool, y_clip: float | None = None):
    """Plot one split's curve. y_clip excludes failed runs (RMSE > y_clip)
    from the line so a single divergent point doesn't squash the y-axis;
    the failed point is plotted as a faded x-marker for transparency.
    """
    for method, summary in summaries.items():
        fracs, ns, means, stds = _curve(summary, split, metric)
        if len(fracs) == 0:
            continue
        x = ns if use_n_train else fracs * 100
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        marker = METHOD_MARKERS.get(method, "o")
        if y_clip is not None and metric in ("RMSE", "MAE"):
            ok_mask = means <= y_clip
            bad_mask = ~ok_mask
            ax.errorbar(x[ok_mask], means[ok_mask], yerr=stds[ok_mask], label=label,
                        marker=marker, markersize=7, linewidth=2, capsize=3, color=color)
            if bad_mask.any():
                ax.scatter(x[bad_mask], np.full(bad_mask.sum(), y_clip * 0.97),
                           marker="x", s=80, color=color, alpha=0.6)
        else:
            ax.errorbar(x, means, yerr=stds, label=label,
                        marker=marker, markersize=7, linewidth=2,
                        capsize=3, color=color)
    if log_x:
        ax.set_xscale("log")
    if y_clip is not None and metric in ("RMSE", "MAE"):
        ax.set_ylim(top=y_clip)
    ax.set_xlabel("training rows" if use_n_train else "fraction of training data (%)")
    ax.set_ylabel(metric)
    ax.set_title(f"{split}  ({metric} vs. data)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="RMSE",
                        help="Metric to plot (RMSE, MAE, R2, PS_RMSE, ...).")
    parser.add_argument("--log-x", action="store_true",
                        help="Use log-scale x-axis.")
    parser.add_argument("--use-n-train", action="store_true",
                        help="Use training row count on x instead of fraction.")
    parser.add_argument("--methods", nargs="+", default=PLOT_METHODS,
                        help="Methods to include in the plot (default: lgb + 3 fastprop sizes).")
    parser.add_argument("--all", action="store_true",
                        help="Plot every method that has results (overrides --methods).")
    args = parser.parse_args()

    summaries = _load_summaries(only=None if args.all else args.methods)
    if not summaries:
        print(f"No summaries found under {RESULTS_DIR}.")
        return

    print("Methods with summaries:")
    for m, s in summaries.items():
        print(f"  {m}: {len(s['by_fraction'])} fractions x {len(s['seeds'])} seeds")

    # Cap RMSE/MAE plots at 1.5 so an occasional failed run doesn't squash
    # the visible range. Failed points are shown as a faded 'x' at the top.
    Y_CLIP = 1.5 if args.metric in ("RMSE", "MAE") else None

    for split in EVAL_SPLITS:
        fig, ax = plt.subplots(figsize=(7, 5))
        _plot_split(ax, summaries, split, args.metric, args.use_n_train, args.log_x, Y_CLIP)
        out = FIG_DIR / f"data_scaling_{split}_{args.metric}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"  wrote {out}")

    fig, axes = plt.subplots(1, len(EVAL_SPLITS), figsize=(6 * len(EVAL_SPLITS), 5),
                             sharey=False)
    for ax, split in zip(axes, EVAL_SPLITS):
        _plot_split(ax, summaries, split, args.metric, args.use_n_train, args.log_x, Y_CLIP)
    out = FIG_DIR / f"data_scaling_panel_{args.metric}.png"
    fig.suptitle(f"Data scaling: {args.metric} vs. training data", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- Overfitting diagnostic plot: train_RMSE_at_best vs val (eval) RMSE ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, summary in summaries.items():
        fracs_v, ns, val_means, _ = _curve(summary, "eval", "RMSE")
        fracs_t, train_vals = _diag_curve(summary, "_train_RMSE_at_best")
        if len(val_means) == 0:
            continue
        x_v = ns if args.use_n_train else fracs_v * 100
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        marker = METHOD_MARKERS.get(method, "o")
        ax.plot(x_v, val_means, marker=marker, markersize=7, linewidth=2,
                color=color, label=f"{label} (val)")
        if len(train_vals) > 0:
            x_t = ns[:len(train_vals)] if args.use_n_train else fracs_t * 100
            ax.plot(x_t, train_vals, marker=marker, markersize=5, linewidth=1.5,
                    linestyle="--", color=color, alpha=0.7,
                    label=f"{label} (train)")
    if args.log_x:
        ax.set_xscale("log")
    ax.set_xlabel("training rows" if args.use_n_train else "fraction of training data (%)")
    ax.set_ylabel("RMSE")
    ax.set_title("Train vs Val RMSE  (overfitting diagnostic)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    out = FIG_DIR / "data_scaling_overfit_diag.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")

    csv_path = RESULTS_DIR / f"data_scaling_table_{args.metric}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "fraction", "n_train", "split",
                    f"{args.metric}_mean", f"{args.metric}_std", "n_seeds"])
        for method, summary in summaries.items():
            for f_str, payload in sorted(summary["by_fraction"].items(),
                                         key=lambda kv: float(kv[0])):
                n_tr = payload.get("n_train_mean")
                for sn in EVAL_SPLITS:
                    agg = payload.get("aggregated", {}).get(sn, {})
                    if f"{args.metric}_mean" not in agg:
                        continue
                    w.writerow([
                        method, f_str, n_tr, sn,
                        agg[f"{args.metric}_mean"],
                        agg.get(f"{args.metric}_std", ""),
                        agg.get(f"{args.metric}_n", ""),
                    ])
    print(f"  wrote {csv_path}")


if __name__ == "__main__":
    main()
