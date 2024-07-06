"""
Generate plots and a clean CSV table for the Transfer-Learning ablation.

Reads `results/<protocol>__<variant>/frac_<f>/seed_<s>.json` for each run,
aggregates mean ± std across seeds, and produces:

  figures/transfer_panel_<metric>.png      3-panel (eval, ood, sc3_gold)
  figures/transfer_grouped_<metric>.png    grouped bars at each fraction
  figures/transfer_data_efficiency.png     line plot RMSE vs fraction
  results/transfer_summary_<metric>.csv    long-format summary table

Usage:
  python make_plots.py
  python make_plots.py --metric MAE
  python make_plots.py --metrics RMSE PS_RMSE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ABLATIONS_TRANSFER_DIR / "results"
FIGURES_DIR = ABLATIONS_TRANSFER_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SPLITS = ("eval", "ood", "sc3_gold")
DEFAULT_METRIC = "RMSE"

# Use distinct, paper-quality colours; reuse the SC3 palette where possible
COLOURS = {
    ("scratch", "full"):       "#777777",  # neutral grey
    ("scratch", "head_only"):  "#bbbbbb",  # light grey (sanity)
    ("qm",      "full"):       "#1f77b4",  # blue
    ("qm",      "head_only"):  "#9bcae1",  # light blue
}
LABELS = {
    ("scratch", "full"):       "Scratch (full)",
    ("scratch", "head_only"):  "Scratch (head-only, sanity)",
    ("qm",      "full"):       "QM-pretrain → SC3 (full)",
    ("qm",      "head_only"):  "QM-pretrain → SC3 (head-only)",
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_long() -> pd.DataFrame:
    """Read every per-run JSON into one long-format DataFrame.

    Columns: protocol, variant, fraction, seed, plus <split>_<metric>.
    """
    rows: list[dict] = []
    for p in sorted(RESULTS_DIR.rglob("seed_*.json")):
        proto, variant = p.parent.parent.name.split("__")
        frac = float(p.parent.name.replace("frac_", ""))
        seed = int(p.stem.replace("seed_", ""))
        with open(p) as f:
            d = json.load(f)
        row = {"protocol": proto, "variant": variant,
               "fraction": frac, "seed": seed,
               "n_train": d.get("_n_train", d.get("n_train"))}
        for split, m in d.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        row[f"{split}__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, metric: str, splits=DEFAULT_SPLITS) -> pd.DataFrame:
    """Return per-(proto, variant, fraction) mean + std for each <split>__<metric>."""
    cols = [f"{s}__{metric}" for s in splits if f"{s}__{metric}" in df.columns]
    grouped = df.groupby(["protocol", "variant", "fraction"])
    out = grouped[cols].agg(["mean", "std"]).reset_index()
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _panel_plot(agg: pd.DataFrame, metric: str, splits=DEFAULT_SPLITS,
                fastprop_baseline: dict | None = None) -> Path:
    """3-panel line plot: one panel per split, line per (proto, variant)."""
    fractions_sorted = sorted(agg["fraction"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 4.0),
                             sharey=False)
    if len(splits) == 1:
        axes = [axes]

    # Order lines: scratch/full and qm/full first; qm/head_only third;
    # skip scratch/head_only (broken sanity).
    order = [("scratch", "full"), ("qm", "full"), ("qm", "head_only")]
    for ax, split in zip(axes, splits):
        for proto, variant in order:
            sub = agg[(agg["protocol"] == proto) & (agg["variant"] == variant)]
            sub = sub.sort_values("fraction")
            if sub.empty:
                continue
            x = sub["fraction"].values
            mean_col = (f"{split}__{metric}", "mean")
            std_col  = (f"{split}__{metric}", "std")
            if mean_col not in sub.columns:
                continue
            y  = sub[mean_col].values
            yerr = sub[std_col].values
            ax.errorbar(x, y, yerr=yerr, marker="o",
                        color=COLOURS[(proto, variant)],
                        label=LABELS[(proto, variant)],
                        linewidth=2.0, markersize=6, capsize=3)
        if fastprop_baseline is not None and split in fastprop_baseline:
            ax.axhline(fastprop_baseline[split], color="k", ls="--", lw=1.2,
                       alpha=0.7, label="FastProp (full SC3 train)")
        ax.set_title(split.replace("_", " ").upper())
        ax.set_xlabel("SC3 train fraction")
        ax.set_ylabel(f"{metric}  (logS units)")
        ax.set_xscale("log")
        ax.set_xticks(fractions_sorted)
        ax.set_xticklabels([f"{f:g}" for f in fractions_sorted])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Q3 Transfer Learning: {metric} vs SC3 train fraction "
                 f"(mean ± std over 3 seeds)", fontsize=11)
    fig.tight_layout()
    out_path = FIGURES_DIR / f"transfer_panel_{metric}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _grouped_bar_plot(agg: pd.DataFrame, metric: str,
                      splits=DEFAULT_SPLITS) -> Path:
    """Grouped bar chart: x=fraction, hue=(proto, variant), one panel per split."""
    fractions_sorted = sorted(agg["fraction"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(5.0 * len(splits), 4.0),
                             sharey=False)
    if len(splits) == 1:
        axes = [axes]

    order = [("scratch", "full"), ("qm", "full"), ("qm", "head_only")]
    width = 0.8 / max(len(order), 1)

    for ax, split in zip(axes, splits):
        x = np.arange(len(fractions_sorted))
        for i, (proto, variant) in enumerate(order):
            sub = agg[(agg["protocol"] == proto) & (agg["variant"] == variant)]
            if sub.empty:
                continue
            sub = sub.set_index("fraction")
            mean_col = (f"{split}__{metric}", "mean")
            std_col  = (f"{split}__{metric}", "std")
            if mean_col not in sub.columns:
                continue
            ys  = [sub.loc[f, mean_col] for f in fractions_sorted]
            yerr = [sub.loc[f, std_col]  for f in fractions_sorted]
            ax.bar(x + i * width - 0.4 + width / 2, ys, width=width,
                   yerr=yerr, capsize=3,
                   color=COLOURS[(proto, variant)],
                   label=LABELS[(proto, variant)],
                   edgecolor="black", linewidth=0.5)
        ax.set_title(split.replace("_", " ").upper())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f:g}" for f in fractions_sorted])
        ax.set_xlabel("SC3 train fraction")
        ax.set_ylabel(f"{metric}  (logS units)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Q3 Transfer Learning ({metric})  —  3 seeds", fontsize=11)
    fig.tight_layout()
    out_path = FIGURES_DIR / f"transfer_grouped_{metric}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _data_efficiency_plot(agg: pd.DataFrame, splits=DEFAULT_SPLITS) -> Path:
    """One plot showing RMSE on each split for scratch/full vs qm/full."""
    fractions_sorted = sorted(agg["fraction"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(4.5 * len(splits), 4.0),
                             sharey=False)
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        for proto, c, label in [
            ("scratch", "#777777", "Scratch"),
            ("qm",      "#1f77b4", "QM-pretrained"),
        ]:
            sub = agg[(agg["protocol"] == proto) & (agg["variant"] == "full")]
            sub = sub.sort_values("fraction")
            if sub.empty:
                continue
            x = sub["fraction"].values
            y    = sub[(f"{split}__RMSE", "mean")].values
            yerr = sub[(f"{split}__RMSE", "std")].values
            ax.errorbar(x, y, yerr=yerr, marker="o",
                        color=c, label=label, linewidth=2.2,
                        markersize=7, capsize=3)
        ax.set_xscale("log")
        ax.set_xticks(fractions_sorted)
        ax.set_xticklabels([f"{int(100*f)}%" for f in fractions_sorted])
        ax.set_title(f"{split.replace('_', ' ').upper()}  RMSE")
        ax.set_xlabel("SC3 train fraction (log)")
        ax.set_ylabel("RMSE  (logS)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc="upper right")
    fig.suptitle("Q3: Does QM-pretraining give data-efficient SC3 fine-tuning?",
                 fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "transfer_data_efficiency.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _write_summary_csv(agg: pd.DataFrame, metric: str) -> Path:
    df = agg.copy()
    df.columns = ["__".join([str(c) for c in col if c]).strip("_")
                  for col in df.columns]
    out_path = RESULTS_DIR / f"transfer_summary_{metric}.csv"
    df.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(metrics: list[str], splits: tuple[str, ...]) -> None:
    long = _load_long()
    if long.empty:
        print("[plots] no results found; nothing to do.")
        return
    print(f"[plots] loaded {len(long)} runs")

    # FastProp main-paper baselines (mean across 5 seeds in vansh/results/fastprop)
    fastprop_baseline = None
    fp_summary = ABLATIONS_TRANSFER_DIR.parent.parent / "results" / "fastprop" / "summary.json"
    if fp_summary.exists():
        with open(fp_summary) as f:
            fp = json.load(f)
        fastprop_baseline = {
            s: fp["aggregated"].get(s, {}).get("RMSE_mean")
            for s in splits
            if fp["aggregated"].get(s, {}).get("RMSE_mean") is not None
        }
        print(f"[plots] FastProp main baseline (RMSE): "
              f"{ {k: round(v, 4) for k, v in fastprop_baseline.items()} }")

    for metric in metrics:
        agg = _agg(long, metric, splits=splits)
        out_csv = _write_summary_csv(agg, metric)
        print(f"[plots] wrote {out_csv}")

        out1 = _panel_plot(agg, metric, splits=splits,
                           fastprop_baseline=fastprop_baseline if metric == "RMSE" else None)
        print(f"[plots] wrote {out1}")
        out2 = _grouped_bar_plot(agg, metric, splits=splits)
        print(f"[plots] wrote {out2}")
        if metric == "RMSE":
            out3 = _data_efficiency_plot(agg, splits=splits)
            print(f"[plots] wrote {out3}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default=None,
                   help="Single metric (overrides --metrics).")
    p.add_argument("--metrics", nargs="+", default=["RMSE", "PS_RMSE"],
                   help="Metrics to plot (default: RMSE PS_RMSE).")
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    args = p.parse_args()
    metrics = [args.metric] if args.metric else list(args.metrics)
    main(metrics, tuple(args.splits))
