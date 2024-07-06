"""
Plots and summary tables for the 298 K-locked transfer-learning ablation.

Reads `results_298k/<approach>/<protocol>__<variant>/frac_<f>/seed_<s>.json`
for each `(approach, protocol, variant, fraction, seed)` cell, aggregates
mean ± std across seeds, and produces:

  figures/transfer_298k_panel_<metric>.png        3-panel (eval/ood/sc3_gold)
  figures/transfer_298k_filter_<metric>.png       single approach (filter)
  figures/transfer_298k_interp_<metric>.png       single approach (interp)
  figures/transfer_298k_compare_T.png             multi-T vs 298 K side-by-side
  results/transfer_298k_summary_<metric>.csv      long-format CSV

Usage:
  python make_plots_298k.py                   # default RMSE + PS_RMSE
  python make_plots_298k.py --metric MAE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ABLATIONS_TRANSFER_DIR / "results_298k"
RESULTS_MAIN_DIR = ABLATIONS_TRANSFER_DIR / "results"
FIGURES_DIR = ABLATIONS_TRANSFER_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SPLITS = ("eval", "ood", "sc3_gold")
DEFAULT_METRIC = "RMSE"
APPROACHES = ("filter", "interp")

COLOURS = {
    ("scratch", "full"):       "#777777",
    ("scratch", "head_only"):  "#bbbbbb",
    ("qm",      "full"):       "#1f77b4",
    ("qm",      "head_only"):  "#9bcae1",
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

def _load_long(approach: str | None = None, base: Path = RESULTS_DIR) -> pd.DataFrame:
    """Read every per-run JSON for one approach (or all approaches) into long-format."""
    if approach:
        glob_root = base / approach
    else:
        glob_root = base
    rows: list[dict] = []
    for p in sorted(glob_root.rglob("seed_*.json")):
        parts = p.relative_to(base).parts
        # parts = (<approach>, <proto>__<var>, frac_<f>, seed_<s>.json)
        if len(parts) != 4:
            continue
        a, prov, fr, sd = parts
        proto, variant = prov.split("__")
        frac = float(fr.replace("frac_", ""))
        seed = int(sd.replace("seed_", "").replace(".json", ""))
        with open(p) as f:
            d = json.load(f)
        row = {"approach": a, "protocol": proto, "variant": variant,
               "fraction": frac, "seed": seed,
               "n_train": d.get("_n_train", d.get("n_train"))}
        for split, m in d.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        row[f"{split}__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, metric: str, splits=DEFAULT_SPLITS,
         group_keys=("approach", "protocol", "variant", "fraction")) -> pd.DataFrame:
    cols = [f"{s}__{metric}" for s in splits if f"{s}__{metric}" in df.columns]
    g = df.groupby(list(group_keys))
    return g[cols].agg(["mean", "std", "count"]).reset_index()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _panel(agg: pd.DataFrame, metric: str, approach: str, splits=DEFAULT_SPLITS,
           order=(("scratch", "full"), ("qm", "full"), ("qm", "head_only"))
           ) -> Path:
    fractions_sorted = sorted(agg["fraction"].unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 4.0))
    if len(splits) == 1:
        axes = [axes]
    for ax, split in zip(axes, splits):
        for proto, variant in order:
            sub = agg[(agg["approach"] == approach)
                      & (agg["protocol"] == proto)
                      & (agg["variant"] == variant)].sort_values("fraction")
            if sub.empty:
                continue
            x = sub["fraction"].values
            y_col = (f"{split}__{metric}", "mean")
            yerr_col = (f"{split}__{metric}", "std")
            if y_col not in sub.columns:
                continue
            y = sub[y_col].values
            yerr = sub[yerr_col].values
            ax.errorbar(x, y, yerr=yerr, marker="o",
                        color=COLOURS[(proto, variant)],
                        label=LABELS[(proto, variant)],
                        linewidth=2.0, markersize=6, capsize=3)
        ax.set_title(split.replace("_", " ").upper())
        ax.set_xlabel("SC3 train fraction")
        ax.set_ylabel(f"{metric}  (logS units)")
        ax.set_xscale("log")
        ax.set_xticks(fractions_sorted)
        ax.set_xticklabels([f"{f:g}" for f in fractions_sorted])
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Q3 Transfer Learning, 298 K-locked ({approach}): "
                 f"{metric} vs SC3 train fraction (mean ± std, 5 seeds)", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / f"transfer_298k_{approach}_{metric}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _two_approach_panel(agg: pd.DataFrame, metric: str, splits=DEFAULT_SPLITS) -> Path:
    """Side-by-side rows: top=filter, bottom=interp; cols = splits."""
    fractions_sorted = sorted(agg["fraction"].unique())
    fig, axes = plt.subplots(2, len(splits),
                             figsize=(4.6 * len(splits), 7.5),
                             sharey="col")
    for r, approach in enumerate(APPROACHES):
        for c, split in enumerate(splits):
            ax = axes[r, c]
            for proto, variant in [("scratch", "full"), ("qm", "full")]:
                sub = agg[(agg["approach"] == approach)
                          & (agg["protocol"] == proto)
                          & (agg["variant"] == variant)].sort_values("fraction")
                if sub.empty:
                    continue
                x = sub["fraction"].values
                y_col = (f"{split}__{metric}", "mean")
                yerr_col = (f"{split}__{metric}", "std")
                if y_col not in sub.columns:
                    continue
                y = sub[y_col].values
                yerr = sub[yerr_col].values
                ax.errorbar(x, y, yerr=yerr, marker="o",
                            color=COLOURS[(proto, variant)],
                            label=LABELS[(proto, variant)],
                            linewidth=2.0, markersize=6, capsize=3)
            ax.set_xscale("log")
            ax.set_xticks(fractions_sorted)
            ax.set_xticklabels([f"{f:g}" for f in fractions_sorted])
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(split.replace("_", " ").upper())
            if c == 0:
                ax.set_ylabel(f"{approach.upper()}\n{metric}  (logS units)")
            if r == 1:
                ax.set_xlabel("SC3 train fraction")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Q3 Transfer Learning, 298 K-locked: {metric} vs SC3 train fraction",
                 fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / f"transfer_298k_panel_{metric}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _multi_T_compare(metric: str = "RMSE", splits=DEFAULT_SPLITS) -> Path | None:
    """Compare three datasets: full multi-T (results/), filter (298 K real),
    and interp (298 K Apelblat-evaluated).  Row 1: scratch only.  Row 2: qm only.
    Each cell shows the mean RMSE at the three SC3 fractions."""

    long_main = _load_long_main()
    long_298 = _load_long(base=RESULTS_DIR)
    if long_main.empty or long_298.empty:
        return None

    agg_main = _agg(long_main, metric,
                    group_keys=("protocol", "variant", "fraction"))
    agg_298 = _agg(long_298, metric)

    fractions_sorted = sorted(set(long_main["fraction"]).intersection(long_298["fraction"]))
    fig, axes = plt.subplots(2, len(splits),
                             figsize=(4.6 * len(splits), 7.5),
                             sharey="col")
    proto_to_row = {"scratch": 0, "qm": 1}
    for proto, r in proto_to_row.items():
        for c, split in enumerate(splits):
            ax = axes[r, c]
            x = np.array(fractions_sorted)
            for tag, agg, color, label in [
                ("multi_T", agg_main, "#cb4b16", "Multi-T SC3 (original)"),
                ("filter",  agg_298,  "#268bd2", "298 K-only (filter)"),
                ("interp",  agg_298,  "#2aa198", "298 K interpolated"),
            ]:
                sub = agg[(agg.get("protocol", agg.get("approach", None)) == proto)
                          & (agg["variant"] == "full")]
                if "approach" in agg.columns:
                    if tag != "multi_T":
                        sub = sub[sub["approach"] == tag]
                    else:
                        continue
                else:
                    if tag != "multi_T":
                        continue
                sub = sub.sort_values("fraction")
                y_col = (f"{split}__{metric}", "mean")
                yerr_col = (f"{split}__{metric}", "std")
                if y_col not in sub.columns or sub.empty:
                    continue
                # Filter to common fractions
                sub = sub[sub["fraction"].isin(fractions_sorted)]
                if sub.empty:
                    continue
                ax.errorbar(sub["fraction"].values,
                            sub[y_col].values,
                            yerr=sub[yerr_col].values,
                            marker="o", color=color, label=label,
                            linewidth=2.0, markersize=6, capsize=3)
            ax.set_xscale("log")
            ax.set_xticks(fractions_sorted)
            ax.set_xticklabels([f"{f:g}" for f in fractions_sorted])
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(split.replace("_", " ").upper())
            if c == 0:
                ax.set_ylabel(f"{proto}\n{metric}  (logS units)")
            if r == 1:
                ax.set_xlabel("SC3 train fraction")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Multi-T vs 298 K-locked SC3:  scratch (top) and QM-pretrain (bot)",
                 fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / f"transfer_298k_compare_T_{metric}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def _load_long_main() -> pd.DataFrame:
    """Long-format DataFrame from `results/` (the multi-T main experiment)."""
    rows: list[dict] = []
    for p in sorted(RESULTS_MAIN_DIR.rglob("seed_*.json")):
        parts = p.relative_to(RESULTS_MAIN_DIR).parts
        if len(parts) != 3:
            continue
        prov, fr, sd = parts
        if "__" not in prov:
            continue
        proto, variant = prov.split("__")
        frac = float(fr.replace("frac_", ""))
        seed = int(sd.replace("seed_", "").replace(".json", ""))
        with open(p) as f:
            d = json.load(f)
        row = {"protocol": proto, "variant": variant,
               "fraction": frac, "seed": seed}
        for split, m in d.items():
            if isinstance(m, dict):
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        row[f"{split}__{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(metrics: list[str], splits: tuple[str, ...]):
    long = _load_long()
    if long.empty:
        print("[plots] no 298k results yet"); return
    print(f"[plots] loaded {len(long)} 298k runs across approaches: "
          f"{sorted(long['approach'].unique())}")
    for metric in metrics:
        agg = _agg(long, metric, splits=splits)
        # Per-approach panel
        for approach in long["approach"].unique():
            out = _panel(agg, metric, approach, splits=splits)
            print(f"[plots] wrote {out}")
        # 2-row panel (filter top / interp bot)
        out = _two_approach_panel(agg, metric, splits=splits)
        print(f"[plots] wrote {out}")
        # Long-format CSV
        df_csv = agg.copy()
        df_csv.columns = ["__".join([str(c) for c in col if c]).strip("_")
                          for col in df_csv.columns]
        out_csv = ABLATIONS_TRANSFER_DIR / "results_298k" / f"transfer_298k_summary_{metric}.csv"
        df_csv.to_csv(out_csv, index=False)
        print(f"[plots] wrote {out_csv}")
        # Multi-T comparison (only for RMSE)
        if metric == "RMSE":
            out = _multi_T_compare(metric, splits=splits)
            if out:
                print(f"[plots] wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default=None)
    p.add_argument("--metrics", nargs="+", default=["RMSE", "PS_RMSE"])
    p.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    args = p.parse_args()
    metrics = [args.metric] if args.metric else list(args.metrics)
    main(metrics, tuple(args.splits))
