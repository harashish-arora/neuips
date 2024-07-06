"""
Collect benchmark results into summary tables (CSV + console).

Reads results/*/summary.json and produces:
  - results/benchmark_table.csv
  - Console summary
"""

import csv
import json
from pathlib import Path

from .registry import METHOD_REGISTRY, METHOD_ORDER, EVAL_SPLITS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS = ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE", "f_aleatoric"]


def load_all_summaries() -> dict:
    summaries = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir():
            sp = d / "summary.json"
            if sp.exists():
                with open(sp) as f:
                    summaries[d.name] = json.load(f)
    return summaries


def print_status():
    """Print which methods have results and their headline metrics."""
    summaries = load_all_summaries()
    if not summaries:
        print("No results found.")
        return

    print(f"\n{'Method':<25s} {'Family':<12s} {'Seeds':>5s} {'eval RMSE':>10s} {'gold RMSE':>10s} {'gold Z':>8s}")
    print("-" * 75)

    ordered = [m for m in METHOD_ORDER if m in summaries]
    extra = [m for m in summaries if m not in METHOD_ORDER]

    for mk in ordered + extra:
        s = summaries[mk]
        info = METHOD_REGISTRY.get(mk, {})
        display = info.get("display", mk)
        family = info.get("family", "?")
        agg = s.get("aggregated", s)

        def g(split, metric):
            v = agg.get(split, {}).get(f"{metric}_mean")
            if v is None:
                v = s.get(f"{split}_{metric}_mean")
            return v

        n_seeds = len(s.get("seeds", [])) or s.get("n_seeds", "?")
        ev = g("eval", "RMSE")
        gold = g("sc3_gold", "RMSE")
        z = g("sc3_gold", "Z_RMSE")

        ev_s = f"{ev:.4f}" if ev else "---"
        gold_s = f"{gold:.4f}" if gold else "---"
        z_s = f"{z:.1f}" if z else "---"
        print(f"{display:<25s} {family:<12s} {str(n_seeds):>5s} {ev_s:>10s} {gold_s:>10s} {z_s:>8s}")

    print(f"\n{len(summaries)} methods with results.\n")


def write_csv():
    """Write benchmark_table.csv from all summary.json files."""
    summaries = load_all_summaries()
    if not summaries:
        print("No results to collect.")
        return

    ordered = [m for m in METHOD_ORDER if m in summaries]
    extra = [m for m in summaries if m not in METHOD_ORDER]

    fields = ["method", "display", "family", "n_seeds"]
    for sp in EVAL_SPLITS:
        for metric in METRICS:
            fields.append(f"{sp}_{metric}_mean")
            fields.append(f"{sp}_{metric}_std")

    out_path = RESULTS_DIR / "benchmark_table.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for mk in ordered + extra:
            s = summaries[mk]
            info = METHOD_REGISTRY.get(mk, {})
            agg = s.get("aggregated", s)
            row = {
                "method": mk,
                "display": info.get("display", mk),
                "family": info.get("family", "?"),
                "n_seeds": len(s.get("seeds", [])) or s.get("n_seeds", 0),
            }
            for sp in EVAL_SPLITS:
                sp_data = agg.get(sp, {})
                for metric in METRICS:
                    row[f"{sp}_{metric}_mean"] = sp_data.get(f"{metric}_mean", "")
                    row[f"{sp}_{metric}_std"]  = sp_data.get(f"{metric}_std", "")
                    if row[f"{sp}_{metric}_mean"] == "":
                        row[f"{sp}_{metric}_mean"] = s.get(f"{sp}_{metric}_mean", "")
                        row[f"{sp}_{metric}_std"]  = s.get(f"{sp}_{metric}_std", "")
            w.writerow(row)

    print(f"CSV written to {out_path} ({len(ordered)+len(extra)} methods)")


def list_methods():
    """Print all registered methods and whether results exist."""
    summaries = load_all_summaries()

    print(f"\n{'Key':<25s} {'Display':<25s} {'Family':<12s} {'Featurizer':<10s} {'Results':>8s}")
    print("-" * 85)
    for mk in METHOD_ORDER:
        info = METHOD_REGISTRY[mk]
        has = "done" if mk in summaries else "---"
        print(f"{mk:<25s} {info['display']:<25s} {info['family']:<12s} {info['featurizer']:<10s} {has:>8s}")
    print()
