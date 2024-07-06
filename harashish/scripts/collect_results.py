#!/usr/bin/env python3
"""
Collect all benchmark results into a single summary table.

Reads results/*/summary.json and produces:
  - results/benchmark_table.csv  (machine-readable)
  - results/benchmark_table.tex  (LaTeX-ready)
  - Prints a formatted console table

Usage:
    python scripts/collect_results.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Method display names and ordering
METHOD_ORDER = [
    # Analytical
    "gse", "esol", "abraham_lfer",
    # Physics
    "unifac",
    # Descriptor + ML (RDKit)
    "rf_rdkit", "xgb_rdkit", "lgb_rdkit", "catboost_rdkit",
    "knn_rdkit", "mlp_rdkit", "dt_rdkit",
    # Descriptor + ML (Morgan)
    "rf_morgan", "xgb_morgan", "lgb_morgan", "catboost_morgan",
    # Fingerprint + GP
    "gp_tanimoto",
    # GNNs
    "gnn_gcn", "gnn_gat", "gnn_gin", "solubnet",
    # Deep descriptor
    "fastprop", "fastsolv",
    # Domain-specific
    "dissolvr", "abraham_ml", "unifac_ml",
    "tayyebi_mordred",
    # Advanced
    "chemprop", "soltrannet", "rilood",
]

METHOD_DISPLAY = {
    "gse": "GSE",
    "esol": "ESOL",
    "abraham_lfer": "Abraham LFER",
    "unifac": "UNIFAC",
    "rf_rdkit": "RF (RDKit)",
    "xgb_rdkit": "XGBoost (RDKit)",
    "lgb_rdkit": "LightGBM (RDKit)",
    "catboost_rdkit": "CatBoost (RDKit)",
    "knn_rdkit": "kNN (RDKit)",
    "mlp_rdkit": "MLP (RDKit)",
    "dt_rdkit": "DT (RDKit)",
    "rf_morgan": "RF (Morgan)",
    "xgb_morgan": "XGBoost (Morgan)",
    "lgb_morgan": "LightGBM (Morgan)",
    "catboost_morgan": "CatBoost (Morgan)",
    "gp_tanimoto": "GP (Tanimoto)",
    "gnn_gcn": "GCN",
    "gnn_gat": "GAT",
    "gnn_gin": "GIN",
    "solubnet": "SolubNet (TAGConv)",
    "fastprop": "FastProp",
    "fastsolv": "FastSolv",
    "dissolvr": "Dissolvr",
    "abraham_ml": "Abraham ML",
    "unifac_ml": "UNIFAC+ML",
    "tayyebi_mordred": "Tayyebi (Mordred)",
    "chemprop": "Chemprop",
    "soltrannet": "SolTranNet",
    "rilood": "RILOOD",
}

BUCKET_LABELS = {
    "gse": "Analytical", "esol": "Analytical", "abraham_lfer": "Analytical",
    "unifac": "Physics",
    "rf_rdkit": "Desc+ML", "xgb_rdkit": "Desc+ML", "lgb_rdkit": "Desc+ML",
    "catboost_rdkit": "Desc+ML", "knn_rdkit": "Desc+ML", "mlp_rdkit": "Desc+ML",
    "dt_rdkit": "Desc+ML",
    "rf_morgan": "FP+ML", "xgb_morgan": "FP+ML", "lgb_morgan": "FP+ML",
    "catboost_morgan": "FP+ML",
    "gp_tanimoto": "FP+GP",
    "gnn_gcn": "GNN", "gnn_gat": "GNN", "gnn_gin": "GNN", "solubnet": "GNN",
    "fastprop": "Deep Desc", "fastsolv": "Deep Desc",
    "dissolvr": "Domain", "abraham_ml": "Domain", "unifac_ml": "Domain",
    "tayyebi_mordred": "Desc+ML",
    "chemprop": "D-MPNN", "soltrannet": "Transformer", "rilood": "GNN",
}


def load_all_summaries():
    """Load all summary.json files from results/."""
    summaries = {}
    for d in RESULTS_DIR.iterdir():
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summaries[d.name] = json.load(f)
    return summaries


def fmt_val(mean, std, decimals=4):
    """Format mean +/- std."""
    if mean is None:
        return "---"
    if std is not None and std > 1e-6:
        return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def fmt_val_plain(mean, std, decimals=4):
    """Format mean +/- std (plain text)."""
    if mean is None:
        return "---"
    if std is not None and std > 1e-6:
        return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def print_console_table(summaries):
    """Print formatted console table."""
    print(f"\n{'='*120}")
    print("SC3 BENCHMARK RESULTS (mean +/- std across seeds)")
    print(f"{'='*120}")

    header = (f"{'Method':<22} {'Seeds':>5} {'Eval RMSE':>14} {'Hard RMSE':>14} "
              f"{'Hard PS-RMSE':>14} {'Med RMSE':>14} {'Easy RMSE':>14} {'Time(s)':>8}")
    print(header)
    print("-" * 120)

    ordered = [m for m in METHOD_ORDER if m in summaries]
    extra = [m for m in summaries if m not in METHOD_ORDER]

    prev_bucket = None
    for method_name in ordered + extra:
        s = summaries[method_name]
        bucket = BUCKET_LABELS.get(method_name, "Other")
        if bucket != prev_bucket:
            if prev_bucket is not None:
                print("-" * 120)
            prev_bucket = bucket

        display = METHOD_DISPLAY.get(method_name, method_name)

        def g(split, metric):
            return s.get(f"{split}_{metric}_mean"), s.get(f"{split}_{metric}_std")

        print(f"{display:<22} {s.get('n_seeds', '?'):>5} "
              f"{fmt_val_plain(*g('eval','RMSE')):>14} "
              f"{fmt_val_plain(*g('test_hard','RMSE')):>14} "
              f"{fmt_val_plain(*g('test_hard','PS_RMSE')):>14} "
              f"{fmt_val_plain(*g('test_medium','RMSE')):>14} "
              f"{fmt_val_plain(*g('test_easy','RMSE')):>14} "
              f"{s.get('train_time_mean_s', 0):>7.1f}s")

    print(f"{'='*120}\n")


def write_csv(summaries):
    """Write CSV summary table."""
    import csv

    ordered = [m for m in METHOD_ORDER if m in summaries]
    extra = [m for m in summaries if m not in METHOD_ORDER]

    metrics = ["RMSE", "MAE", "R2", "PS_RMSE", "PS_R2", "Z_RMSE"]
    splits = ["eval", "test_ood", "test_hard", "test_medium", "test_easy"]

    fields = ["method", "bucket", "n_seeds", "train_time_s"]
    for split in splits:
        for metric in metrics:
            fields.append(f"{split}_{metric}_mean")
            fields.append(f"{split}_{metric}_std")

    out_path = RESULTS_DIR / "benchmark_table.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for method_name in ordered + extra:
            s = summaries[method_name]
            row = {
                "method": METHOD_DISPLAY.get(method_name, method_name),
                "bucket": BUCKET_LABELS.get(method_name, "Other"),
                "n_seeds": s.get("n_seeds", 0),
                "train_time_s": s.get("train_time_mean_s", 0),
            }
            for split in splits:
                for metric in metrics:
                    row[f"{split}_{metric}_mean"] = s.get(f"{split}_{metric}_mean", "")
                    row[f"{split}_{metric}_std"] = s.get(f"{split}_{metric}_std", "")
            writer.writerow(row)

    print(f"CSV written to {out_path}")


def write_latex(summaries):
    """Write LaTeX table to results/ and paper/tables/."""
    ordered = [m for m in METHOD_ORDER if m in summaries]
    extra = [m for m in summaries if m not in METHOD_ORDER]

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\scthree benchmark results. PS-RMSE (primary metric) and RMSE across all test tiers. Best result per column in \textbf{bold}.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{ll cccccc c}")
    lines.append(r"\toprule")
    lines.append(r" & & \multicolumn{2}{c}{\hard} & \multicolumn{2}{c}{\medium} & \multicolumn{2}{c}{\easy} & \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}")
    lines.append(r"Bucket & Method & PS-RMSE & RMSE & PS-RMSE & RMSE & PS-RMSE & RMSE & Time (s) \\")
    lines.append(r"\midrule")

    prev_bucket = None
    for method_name in ordered + extra:
        s = summaries[method_name]
        bucket = BUCKET_LABELS.get(method_name, "Other")
        display = METHOD_DISPLAY.get(method_name, method_name)

        if bucket != prev_bucket and prev_bucket is not None:
            lines.append(r"\midrule")
        prev_bucket = bucket

        def g(split, metric):
            return s.get(f"{split}_{metric}_mean"), s.get(f"{split}_{metric}_std")

        t = s.get("train_time_mean_s", 0)
        time_str = f"{t:.0f}" if t >= 1 else "$<$1"

        row = (f"{bucket} & {display} & "
               f"{fmt_val(*g('test_hard','PS_RMSE'))} & "
               f"{fmt_val(*g('test_hard','RMSE'))} & "
               f"{fmt_val(*g('test_medium','PS_RMSE'))} & "
               f"{fmt_val(*g('test_medium','RMSE'))} & "
               f"{fmt_val(*g('test_easy','PS_RMSE'))} & "
               f"{fmt_val(*g('test_easy','RMSE'))} & "
               f"{time_str} \\\\")
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex_content = "\n".join(lines)

    # Write to results/
    out_path = RESULTS_DIR / "benchmark_table.tex"
    with open(out_path, "w") as f:
        f.write(tex_content)
    print(f"LaTeX written to {out_path}")

    # Also write to paper/tables/ for \input
    paper_tables = RESULTS_DIR.parent / "paper" / "tables"
    paper_tables.mkdir(parents=True, exist_ok=True)
    paper_path = paper_tables / "main_results.tex"
    with open(paper_path, "w") as f:
        f.write(tex_content)
    print(f"LaTeX written to {paper_path}")


def main():
    summaries = load_all_summaries()

    if not summaries:
        print("No results found in", RESULTS_DIR)
        return

    print(f"Found {len(summaries)} method results: {list(summaries.keys())}")

    print_console_table(summaries)
    write_csv(summaries)
    write_latex(summaries)


if __name__ == "__main__":
    main()
