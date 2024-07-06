#!/usr/bin/env python
"""
Interpretability ablation - LightGBM SHAP analysis & figures.

Consumes the per-featurizer SHAP dumps written by ``run_shap.py``:

    results/<feat>/feature_names.json
    results/<feat>/shap_eval.npz
    results/<feat>/shap_ood.npz
    results/<feat>/metrics.json
    results/<feat>/model_seed_<seed>.pkl

and produces:

    A.  Global feature importance per featurizer
        - results/<feat>/global_importance.csv
        - results/<feat>/global_blockwise.json   (solute / solvent / T totals)
        - figures/global_top_features__<feat>__<split>.png   (bar)
        - figures/global_blockwise_panel.png                  (all featurizers, both splits)

    B.  Per-solvent feature importance  (top-25 ID solvents on `eval`)
        - results/<feat>/per_solvent__<split>.csv
        - results/<feat>/per_solvent_top5.json
        - figures/per_solvent_heatmap__<feat>__<split>.png   (top-K features across solvents)
        - figures/per_solvent_summary__<feat>__<solvent>.png  (one beeswarm per solvent: water/DMSO/ethanol/n-hexane)

    C.  Solvent-solvent clustering (per featurizer)
        - results/<feat>/solvent_similarity.npz
        - figures/solvent_dendrogram__<feat>.png
        - figures/solvent_corr_heatmap__<feat>.png

    D.  TreeSHAP interaction values for `dissolvr` and `rdkit`
        - results/<feat>/shap_interactions__<split>.npz   (sample of N rows)
        - figures/interaction_top__<feat>__<split>.png

    E.  Abraham/LSER axis ranking per solvent
        - results/abraham_only/abraham_axis_ranking.csv
        - figures/abraham_axis_per_solvent.png

For each panel we annotate which evaluation split it uses (eval vs ood) and
report the underlying RMSE/PS-RMSE so the reader can map importance back to
performance, exactly as the SC3 readme requires.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Cap CPU usage to ~60% of cores.
_N_CPUS_TOTAL = os.cpu_count() or 16
_N_JOBS = max(1, int(round(_N_CPUS_TOTAL * 0.60)))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_N_JOBS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

ALL_FEATURIZERS = [
    "rdkit", "morgan", "dissolvr", "mordred",
    "maccs", "atompair", "abraham_only",
]
SHAP_SPLITS = ["eval", "ood"]
TOP_K_FEATURES_PLOT = 20
TOP_K_PER_SOLVENT = 5
PER_SOLVENT_FOCUS = ["water", "DMSO", "ethanol", "n-hexane", "DMF", "acetonitrile"]
# Featurizers we run TreeSHAP interaction values on.  For featurizers with many
# features (rdkit, dissolvr, mordred, morgan, atompair, maccs) the
# interaction computation is O(N * T * F^2) and quickly becomes intractable
# on a single CPU.  We therefore:
#   - Always run on `abraham_only` (16 features, ~5s per 1000 rows) - exact and fast.
#   - Run on `rdkit` (320 features) with a small sample as the descriptor
#     reference; still ~10 min on CPU.
#   - Skip everything else (dissolvr, mordred, morgan, maccs, atompair) as
#     the block-level interaction story is already told by the block-wise
#     SHAP attribution panel.
INTERACTION_FEATURIZERS = ["abraham_only", "rdkit"]
INTERACTION_SAMPLE = 1500          # used when feature count is small
INTERACTION_SAMPLE_LARGE = 200     # used when feature count > 100
INTERACTION_TREE_LIMIT = 200       # cap iterations scanned
SOLV_CLUSTER_TOP_FEATS = 50  # top features (by overall mean|SHAP|) used to build solvent fingerprints

FEATURIZER_LABELS = {
    "rdkit":        "RDKit 2D desc.",
    "morgan":       "Morgan ECFP4",
    "dissolvr":     "Dissolvr (RDKit+MOSE+Joback+Abr.)",
    "mordred":      "Mordred 2D",
    "maccs":        "MACCS keys",
    "atompair":     "Atom-Pair FP",
    "abraham_only": "Abraham-only (5 LSER + Tm)",
}


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===========================================================================
# Loading helpers
# ===========================================================================

def _load_featurizer(feat: str) -> dict | None:
    """Load all SHAP dumps + names + metrics for one featurizer."""
    fdir = RESULTS_DIR / feat
    if not fdir.exists():
        return None
    fnames_p = fdir / "feature_names.json"
    metrics_p = fdir / "metrics.json"
    if not fnames_p.exists():
        return None
    with open(fnames_p) as f:
        feature_names = json.load(f)
    with open(metrics_p) as f:
        metrics = json.load(f)

    shap_data = {}
    for s in SHAP_SPLITS:
        sp = fdir / f"shap_{s}.npz"
        if not sp.exists():
            continue
        d = np.load(sp, allow_pickle=True)
        shap_data[s] = {
            "shap": d["shap"],
            "base_value": float(d["base_value"][0]) if "base_value" in d.files else 0.0,
            "y_true": d["y_true"],
            "solvent_names": d["solvent_names"],
            "solute_smiles": d["solute_smiles"],
            "solvent_smiles": d["solvent_smiles"],
            "temperature": d["temperature"],
        }
    if not shap_data:
        return None
    return {
        "featurizer": feat,
        "feature_names": feature_names,
        "metrics": metrics,
        "shap_data": shap_data,
    }


def _block_indices(feature_names: list[str]) -> dict[str, np.ndarray]:
    """Return dict {block: 1-D indices}.  Blocks: solute / solvent / T."""
    fn = np.array(feature_names)
    return {
        "solute":  np.flatnonzero(np.char.startswith(fn, "solute_")),
        "solvent": np.flatnonzero(np.char.startswith(fn, "solv_")),
        "T":       np.flatnonzero(np.isin(fn, ["T_norm", "T_inv", "T_sq", "T_log"])),
    }


# ===========================================================================
# A. Global feature importance
# ===========================================================================

def analyze_global(feat_dump: dict):
    feat = feat_dump["featurizer"]
    fnames = feat_dump["feature_names"]
    blk = _block_indices(fnames)
    out_dir = RESULTS_DIR / feat
    out_dir.mkdir(exist_ok=True)
    out_global = {}

    for sname, sd in feat_dump["shap_data"].items():
        sv = sd["shap"]
        mabs = np.abs(sv).mean(0)        # global feature importance (mean |SHAP|)
        mraw = sv.mean(0)                 # mean signed contribution
        ranked = np.argsort(-mabs)

        rows = []
        for r, idx in enumerate(ranked):
            rows.append({
                "rank": int(r + 1),
                "feature": fnames[idx],
                "block": ("T" if idx in blk["T"]
                          else "solute" if idx in blk["solute"]
                          else "solvent"),
                "mean_abs_shap": float(mabs[idx]),
                "mean_signed_shap": float(mraw[idx]),
            })
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"global_importance__{sname}.csv", index=False)

        # Block-wise totals
        block_totals = {}
        for blk_name, blk_idx in blk.items():
            block_totals[blk_name] = {
                "sum_mean_abs": float(mabs[blk_idx].sum()),
                "n_features":   int(len(blk_idx)),
                "share":        float(mabs[blk_idx].sum() / mabs.sum()) if mabs.sum() > 0 else 0.0,
            }
        out_global[sname] = block_totals

    with open(out_dir / "global_blockwise.json", "w") as f:
        json.dump(out_global, f, indent=2)


def plot_global_top_features(feat_dump: dict):
    feat = feat_dump["featurizer"]
    fnames = feat_dump["feature_names"]
    blk = _block_indices(fnames)
    block_color = {"solute": "#1f77b4", "solvent": "#2ca02c", "T": "#d62728"}

    for sname, sd in feat_dump["shap_data"].items():
        sv = sd["shap"]
        mabs = np.abs(sv).mean(0)
        ranked = np.argsort(-mabs)[:TOP_K_FEATURES_PLOT][::-1]   # top->bottom
        labels = [fnames[i] for i in ranked]
        vals = mabs[ranked]
        colors = []
        for i in ranked:
            if i in blk["T"]:
                colors.append(block_color["T"])
            elif i in blk["solute"]:
                colors.append(block_color["solute"])
            else:
                colors.append(block_color["solvent"])

        fig, ax = plt.subplots(figsize=(7.0, 0.32 * len(labels) + 1.5))
        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("global mean(|SHAP|) on logS prediction")
        rmse = feat_dump["metrics"].get("metrics", {}).get(sname, {}).get("RMSE", float("nan"))
        ps   = feat_dump["metrics"].get("metrics", {}).get(sname, {}).get("PS_RMSE", float("nan"))
        ax.set_title(
            f"{FEATURIZER_LABELS.get(feat,feat)}  ({sname})\n"
            f"top {TOP_K_FEATURES_PLOT} features by mean(|SHAP|)   "
            f"RMSE={rmse:.3f}  PS-RMSE={ps:.3f}",
            fontsize=10,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f"  {v:.3f}", va="center", fontsize=7)
        # Legend
        from matplotlib.patches import Patch
        ax.legend(
            handles=[Patch(facecolor=c, label=k) for k, c in block_color.items()],
            loc="lower right", fontsize=8, title="block",
        )
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        out = FIG_DIR / f"global_top_features__{feat}__{sname}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        _log(f"  wrote {out}")


def plot_global_blockwise_panel(all_dumps: dict):
    """One panel per split: stacked-bar share of solute/solvent/T per featurizer."""
    feats = [f for f in ALL_FEATURIZERS if f in all_dumps]
    if not feats:
        return
    block_color = {"solute": "#1f77b4", "solvent": "#2ca02c", "T": "#d62728"}

    fig, axes = plt.subplots(1, len(SHAP_SPLITS), figsize=(6.0 * len(SHAP_SPLITS), 5.0),
                             sharey=True)
    if len(SHAP_SPLITS) == 1:
        axes = [axes]
    for ax, sname in zip(axes, SHAP_SPLITS):
        x = np.arange(len(feats))
        bottoms = np.zeros(len(feats))
        for blk_name in ["solute", "solvent", "T"]:
            shares = []
            for f in feats:
                with open(RESULTS_DIR / f / "global_blockwise.json") as fh:
                    g = json.load(fh)
                shares.append(g.get(sname, {}).get(blk_name, {}).get("share", 0.0))
            shares = np.array(shares)
            ax.bar(x, shares, bottom=bottoms, label=blk_name,
                   color=block_color[blk_name], edgecolor="black", linewidth=0.4)
            for xi, (s, b) in enumerate(zip(shares, bottoms)):
                if s > 0.04:
                    ax.text(xi, b + s / 2, f"{s*100:.0f}%",
                            ha="center", va="center", fontsize=8, color="white"
                            if s > 0.10 else "black")
            bottoms = bottoms + shares
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURIZER_LABELS.get(f, f).split(" (")[0]
                            for f in feats], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("share of total mean(|SHAP|)")
        ax.set_title(f"Block-wise SHAP attribution  ({sname})")
        ax.grid(True, alpha=0.3, axis="y")
        if sname == SHAP_SPLITS[0]:
            ax.legend(title="block", loc="lower left", fontsize=8)
    fig.suptitle("Where does the model's signal come from? (LightGBM, fixed HPs)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "global_blockwise_panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")


# ===========================================================================
# B. Per-solvent feature importance
# ===========================================================================

def _top25_id_solvents(eval_solv: np.ndarray) -> list[str]:
    counts = pd.Series(eval_solv).value_counts()
    return counts.index[:25].tolist()


def analyze_per_solvent(feat_dump: dict):
    feat = feat_dump["featurizer"]
    fnames = feat_dump["feature_names"]
    out_dir = RESULTS_DIR / feat

    for sname, sd in feat_dump["shap_data"].items():
        sv = sd["shap"]
        solv = sd["solvent_names"]
        if len(solv) == 0:
            continue
        # Top solvents on this split (top-25 of `eval`, all available on `ood`).
        if sname == "eval":
            target_solvents = _top25_id_solvents(solv)
        else:
            counts = pd.Series(solv).value_counts()
            target_solvents = counts.index[:25].tolist()  # top-25 OOD solvents too

        rows = []
        per_solvent_top5 = {}
        for s in target_solvents:
            mask = solv == s
            n = int(mask.sum())
            if n < 5:
                continue
            sv_s = sv[mask]
            mabs = np.abs(sv_s).mean(0)
            order = np.argsort(-mabs)
            top5 = order[:TOP_K_PER_SOLVENT]
            per_solvent_top5[s] = [
                {"feature": fnames[i], "mean_abs_shap": float(mabs[i]),
                 "mean_signed_shap": float(sv_s.mean(0)[i])}
                for i in top5
            ]
            for r, idx in enumerate(order):
                rows.append({
                    "solvent": s,
                    "n_rows":  n,
                    "rank":    int(r + 1),
                    "feature": fnames[idx],
                    "mean_abs_shap":   float(mabs[idx]),
                    "mean_signed_shap": float(sv_s.mean(0)[idx]),
                })
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"per_solvent__{sname}.csv", index=False)
            with open(out_dir / f"per_solvent_top5__{sname}.json", "w") as f:
                json.dump(per_solvent_top5, f, indent=2)


def plot_per_solvent_heatmap(feat_dump: dict):
    """Heatmap: top-K (overall) features (rows) x top solvents (cols),
    z-scored mean(|SHAP|) per solvent so colours show relative importance."""
    feat = feat_dump["featurizer"]
    fnames = feat_dump["feature_names"]

    for sname, sd in feat_dump["shap_data"].items():
        sv = sd["shap"]
        solv = sd["solvent_names"]
        if len(solv) == 0:
            continue
        # Pick the global top-K features (so all solvents are scored on the same axis).
        global_mabs = np.abs(sv).mean(0)
        K = min(20, sv.shape[1])
        top_global = np.argsort(-global_mabs)[:K]
        top_global_names = [fnames[i] for i in top_global]

        if sname == "eval":
            target_solvents = _top25_id_solvents(solv)
        else:
            counts = pd.Series(solv).value_counts()
            target_solvents = counts.index[:25].tolist()

        # Build matrix (K, S)
        mat_rows = []
        kept = []
        for s in target_solvents:
            mask = solv == s
            if mask.sum() < 5:
                continue
            mabs_s = np.abs(sv[mask]).mean(0)[top_global]
            mat_rows.append(mabs_s)
            kept.append(s)
        if not mat_rows:
            continue
        M = np.array(mat_rows).T              # (K, S)
        # Per-solvent normalisation (column-wise) - shows relative importance
        # of each feature within that solvent column.
        M_norm = M / (M.sum(0, keepdims=True) + 1e-12)

        fig, ax = plt.subplots(figsize=(0.42 * len(kept) + 3, 0.28 * K + 2.5))
        im = ax.imshow(M_norm, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(kept)))
        ax.set_xticklabels(kept, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels(top_global_names, fontsize=8)
        ax.set_title(
            f"Per-solvent SHAP importance  ({FEATURIZER_LABELS.get(feat,feat)}, {sname})\n"
            f"column-normalised mean(|SHAP|) over top-{K} global features  -  "
            f"darker = relatively less important for that solvent",
            fontsize=10,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("col-normalised mean(|SHAP|)", fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"per_solvent_heatmap__{feat}__{sname}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        _log(f"  wrote {out}")


def plot_per_solvent_summary(feat_dump: dict):
    """Beeswarm-style per-solvent SHAP summary plots for a few key solvents.

    Uses ``shap.summary_plot`` (if shap is importable) for a quick chemistry-
    friendly visual.  Done on the ``eval`` split to keep things ID and
    well-populated.
    """
    feat = feat_dump["featurizer"]
    fnames = feat_dump["feature_names"]
    sd = feat_dump["shap_data"].get("eval")
    if sd is None:
        return
    sv = sd["shap"]
    solv = sd["solvent_names"]

    try:
        import shap
    except Exception as e:
        _log(f"  WARN: shap import failed for plot ({e}); skipping summary plots.")
        return

    for s in PER_SOLVENT_FOCUS:
        mask = solv == s
        if mask.sum() < 20:
            continue
        sv_s = sv[mask]
        # We don't have the X matrix saved; use rank-only summary (ignore feature
        # values).  Pass dummy zeros so summary_plot still draws bars sorted by
        # mean |SHAP|.
        plt.figure(figsize=(7, 5.5))
        shap.summary_plot(
            sv_s,
            feature_names=fnames,
            plot_type="bar",
            max_display=15,
            show=False,
        )
        plt.title(f"Per-solvent SHAP   featurizer={feat}   solvent={s}   N={int(mask.sum())}",
                  fontsize=10)
        plt.tight_layout()
        out = FIG_DIR / f"per_solvent_summary__{feat}__{s.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        _log(f"  wrote {out}")


# ===========================================================================
# C. Solvent-solvent clustering (per featurizer)
# ===========================================================================

def analyze_solvent_clustering(feat_dump: dict):
    """Build a per-solvent feature-importance fingerprint, compute solvent x
    solvent cosine similarity, and write a hierarchical-clustering dendrogram
    + heatmap.

    The fingerprint of solvent s on featurizer f is a vector of mean(|SHAP|)
    over the top-K (by global importance) features.  Two solvents whose
    fingerprints are similar are treated similarly by the model.
    """
    feat = feat_dump["featurizer"]
    fnames = feat_dump["feature_names"]
    sd = feat_dump["shap_data"].get("eval")
    if sd is None:
        return
    sv = sd["shap"]
    solv = sd["solvent_names"]
    if len(solv) == 0:
        return

    target_solvents = _top25_id_solvents(solv)
    global_mabs = np.abs(sv).mean(0)
    K = min(SOLV_CLUSTER_TOP_FEATS, sv.shape[1])
    top_idx = np.argsort(-global_mabs)[:K]

    rows, kept = [], []
    for s in target_solvents:
        mask = solv == s
        if mask.sum() < 5:
            continue
        mabs_s = np.abs(sv[mask]).mean(0)[top_idx]
        rows.append(mabs_s)
        kept.append(s)
    if len(kept) < 2:
        return
    F = np.array(rows)                       # (S, K)
    # L2-normalise so cosine sim is dot product
    F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    sim = F_norm @ F_norm.T

    out_dir = RESULTS_DIR / feat
    np.savez_compressed(
        out_dir / "solvent_similarity.npz",
        sim=sim, solvents=np.array(kept, dtype=object),
        feature_names=np.array([fnames[i] for i in top_idx], dtype=object),
        feature_indices=top_idx.astype(np.int32),
        fingerprints=F.astype(np.float32),
    )

    # Dendrogram + reordered heatmap
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    cond = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(cond, method="average")
    order = leaves_list(Z)
    sim_ord = sim[order][:, order]
    labels_ord = [kept[i] for i in order]

    # Heatmap
    fig, ax = plt.subplots(figsize=(0.36 * len(kept) + 3, 0.36 * len(kept) + 3))
    im = ax.imshow(sim_ord, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(kept))); ax.set_xticklabels(labels_ord, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(kept))); ax.set_yticklabels(labels_ord, fontsize=8)
    ax.set_title(f"Solvent x solvent cosine similarity of SHAP fingerprints\n"
                 f"({FEATURIZER_LABELS.get(feat,feat)}, eval, top-{K} features)",
                 fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("cosine similarity", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"solvent_corr_heatmap__{feat}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(max(8, 0.30 * len(kept) + 4), 4.5))
    dendrogram(Z, labels=kept, ax=ax, leaf_rotation=60, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_ylabel("1 - cosine similarity")
    ax.set_title(
        f"Hierarchical clustering of solvents by SHAP fingerprint  "
        f"({FEATURIZER_LABELS.get(feat,feat)}, eval)\n"
        "Solvents that cluster together are treated similarly by the model.",
        fontsize=10,
    )
    fig.tight_layout()
    out = FIG_DIR / f"solvent_dendrogram__{feat}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")


# ===========================================================================
# D. SHAP interaction values (dissolvr + rdkit)
# ===========================================================================

def compute_interactions(feat: str, sample_rows: int | None = None,
                         tree_limit: int | None = INTERACTION_TREE_LIMIT):
    """Exact Tree-SHAP interaction values on a small `eval` sample.

    Compute is O(N * T * L * F^2). For featurizers with many features this is
    very expensive; we fall back to a much smaller sample for F>100 and cap
    the trees scanned via ``tree_limit``.

    Outputs:
      results/<feat>/shap_interactions__eval.npz   (siv (N, F, F), sample_idx)
      results/<feat>/interactions_top50__eval.csv  (ranked pairs)
      figures/interaction_top__<feat>__eval.png
    """
    out_dir = RESULTS_DIR / feat
    fnames_p = out_dir / "feature_names.json"
    if not fnames_p.exists():
        _log(f"  skip interactions: no {fnames_p}")
        return
    with open(fnames_p) as f:
        fnames = json.load(f)

    model_path = sorted(out_dir.glob("model_seed_*.pkl"))
    if not model_path:
        _log(f"  skip interactions: no model in {out_dir}")
        return
    with open(model_path[0], "rb") as f:
        model = pickle.load(f)

    # Load X and align column count to feature names.
    sys.path.insert(0, str(HERE.parent.parent))  # for sc3_bench
    from sc3_bench.data import load_cached_features
    cached = load_cached_features(feat)
    if cached is None:
        return
    X_eval = cached["X_eval"]
    if X_eval.shape[1] != len(fnames):
        m = min(X_eval.shape[1], len(fnames))
        X_eval = X_eval[:, :m]; fnames = fnames[:m]

    F = X_eval.shape[1]
    if sample_rows is None:
        sample_rows = INTERACTION_SAMPLE if F <= 50 else INTERACTION_SAMPLE_LARGE
    rng = np.random.RandomState(0)
    if sample_rows < len(X_eval):
        idx = rng.choice(len(X_eval), size=sample_rows, replace=False)
    else:
        idx = np.arange(len(X_eval))
    Xs = X_eval[idx]
    mem_gb = (sample_rows * F * F * 4) / 1e9
    if mem_gb > 8.0:
        _log(f"  skip interactions for {feat}: would need ~{mem_gb:.1f}GB; too large.")
        return

    import shap
    _log(f"  TreeSHAP interactions for {feat}: {len(idx)} rows x {F} feats  "
         f"(~{mem_gb:.2f} GB, tree_limit={tree_limit})...")
    t0 = time.time()
    expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    siv = expl.shap_interaction_values(Xs, tree_limit=tree_limit)
    if isinstance(siv, list):
        siv = siv[0]
    siv = siv.astype(np.float32)
    _log(f"    siv.shape={siv.shape}  ({time.time()-t0:.1f}s)")

    np.savez_compressed(out_dir / "shap_interactions__eval.npz",
                        siv=siv, sample_idx=idx.astype(np.int32))

    # Rank interactions (off-diagonal absolute mean across sample)
    abs_mean = np.abs(siv).mean(0)
    np.fill_diagonal(abs_mean, 0.0)
    iu = np.triu_indices_from(abs_mean, k=1)
    flat = abs_mean[iu]
    order = np.argsort(-flat)[:50]
    rows = []
    blk = _block_indices(fnames)
    def _blk(i):
        return ("T" if i in blk["T"]
                else "solute" if i in blk["solute"]
                else "solvent")
    for r, k in enumerate(order):
        i, j = iu[0][k], iu[1][k]
        rows.append({
            "rank": int(r + 1),
            "feature_a": fnames[i], "block_a": _blk(i),
            "feature_b": fnames[j], "block_b": _blk(j),
            "abs_mean_interaction": float(abs_mean[i, j]),
            "block_pair": tuple(sorted([_blk(i), _blk(j)])),
        })
    df = pd.DataFrame(rows)
    df["block_pair"] = df["block_pair"].astype(str)
    df.to_csv(out_dir / "interactions_top50__eval.csv", index=False)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8.5, 7))
    top = df.head(20)[::-1]
    labels = [f"{a} <-> {b}" for a, b in zip(top["feature_a"], top["feature_b"])]
    bp_color = {
        "('solute', 'solvent')": "#9467bd",
        "('solute', 'T')":       "#e377c2",
        "('solvent', 'T')":      "#bcbd22",
        "('solute', 'solute')":  "#1f77b4",
        "('solvent', 'solvent')":"#2ca02c",
        "('T', 'T')":            "#d62728",
    }
    colors = [bp_color.get(bp, "#333") for bp in top["block_pair"]]
    ax.barh(np.arange(len(top)), top["abs_mean_interaction"], color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("mean |interaction SHAP|")
    rmse = json.load(open(out_dir / "metrics.json"))["metrics"].get("eval", {}).get("RMSE", float("nan"))
    ax.set_title(f"Top-20 SHAP interaction features  ({FEATURIZER_LABELS.get(feat,feat)}, eval)\n"
                 f"sample={len(idx)}   RMSE={rmse:.3f}",
                 fontsize=10)
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=k.replace("'", "")) for k, c in bp_color.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / f"interaction_top__{feat}__eval.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")


# ===========================================================================
# E. Abraham/LSER axis ranking per solvent
# ===========================================================================

def analyze_abraham_axes():
    """Use the abraham_only featurizer SHAP to rank, for each solvent, which
    of A / B / S / E / V / Tm dominates the model's signal.

    Two views:
      (1) which axis on the *solvent* dominates - 'what type of solvent
          chemistry am I?'
      (2) which axis on the *solute*  dominates - 'what type of solute
          chemistry matters most when this solvent is the medium?'
    """
    dump = _load_featurizer("abraham_only")
    if dump is None:
        _log("  abraham_only not available, skipping LSER analysis.")
        return
    fnames = dump["feature_names"]
    sd = dump["shap_data"].get("eval")
    if sd is None:
        return
    sv = sd["shap"]
    solv = sd["solvent_names"]

    blk = _block_indices(fnames)
    # Axis names (drop the prefix to plot cleanly)
    axes = ["pred_Tm", "abraham_A", "abraham_B", "abraham_S", "abraham_E", "abraham_V"]

    target_solvents = _top25_id_solvents(solv)
    rows = []
    for s in target_solvents:
        mask = solv == s
        if mask.sum() < 5:
            continue
        sv_s = sv[mask]
        for prefix, side in [("solute_", "solute"), ("solv_", "solvent")]:
            for a in axes:
                full = f"{prefix}{a}"
                if full not in fnames:
                    continue
                idx = fnames.index(full)
                rows.append({
                    "solvent": s,
                    "side":    side,
                    "axis":    a,
                    "n_rows":  int(mask.sum()),
                    "mean_abs_shap":   float(np.abs(sv_s[:, idx]).mean()),
                    "mean_signed_shap": float(sv_s[:, idx].mean()),
                })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_dir = RESULTS_DIR / "abraham_only"
    df.to_csv(out_dir / "abraham_axis_ranking.csv", index=False)

    # Heatmap: rows = solvents, cols = axes, value = mean(|SHAP|).  Two panels
    # side by side: solute side, solvent side.
    for side in ["solvent", "solute"]:
        sub = df[df["side"] == side]
        if sub.empty:
            continue
        pivot = sub.pivot(index="solvent", columns="axis", values="mean_abs_shap")
        # Sort rows by total importance descending
        pivot = pivot.reindex(pivot.sum(axis=1).sort_values(ascending=False).index)
        # Sort cols in canonical Abraham order
        col_order = [a for a in axes if a in pivot.columns]
        pivot = pivot[col_order]

        fig, ax = plt.subplots(figsize=(0.65 * len(col_order) + 3,
                                        0.32 * len(pivot) + 2))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels([c.replace("abraham_", "").replace("pred_", "")
                            for c in col_order], fontsize=9)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(
            f"Abraham/LSER axis importance per solvent  -  {side} side\n"
            f"(mean(|SHAP|) on logS, abraham_only featurizer, eval)",
            fontsize=10,
        )
        # Annotate
        vmax = pivot.values.max()
        for i in range(len(pivot)):
            for j in range(len(col_order)):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.45 * vmax else "black")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        fig.tight_layout()
        out = FIG_DIR / f"abraham_axis_per_solvent__{side}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        _log(f"  wrote {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--featurizers", nargs="+", default=ALL_FEATURIZERS,
                        help="Featurizers to analyse (default: all 7).")
    parser.add_argument("--skip-interactions", action="store_true",
                        help="Skip TreeSHAP interaction values (rdkit + dissolvr).")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Only compute tables/JSONs; no PNGs.")
    parser.add_argument("--only", choices=["abc", "interactions", "abraham"],
                        default=None,
                        help="Run only one analysis stage: abc=A+B+C, "
                             "interactions=only D, abraham=only E.")
    parser.add_argument("--interaction-sample", type=int, default=INTERACTION_SAMPLE,
                        help=f"Sample size for interaction values (default: {INTERACTION_SAMPLE}).")
    args = parser.parse_args()

    _log(f"CPU cap: using {_N_JOBS}/{_N_CPUS_TOTAL} cores (60%)")

    do_abc          = args.only in (None, "abc")
    do_interactions = args.only in (None, "interactions") and not args.skip_interactions
    do_abraham      = args.only in (None, "abraham")

    all_dumps = {}
    for f in args.featurizers:
        d = _load_featurizer(f)
        if d is None:
            _log(f"  skip {f}: no SHAP dump found in {RESULTS_DIR/f}")
            continue
        all_dumps[f] = d

    if do_abc:
        for f, d in all_dumps.items():
            _log(f"\n=== {f} ===")
            _log("  A. global importance...")
            analyze_global(d)
            if not args.skip_figures:
                plot_global_top_features(d)

            _log("  B. per-solvent importance...")
            analyze_per_solvent(d)
            if not args.skip_figures:
                plot_per_solvent_heatmap(d)
                plot_per_solvent_summary(d)

            _log("  C. solvent-solvent clustering...")
            analyze_solvent_clustering(d)

        if all_dumps and not args.skip_figures:
            _log("\n=== panel: blockwise across featurizers ===")
            plot_global_blockwise_panel(all_dumps)

    if do_interactions:
        for f in INTERACTION_FEATURIZERS:
            if f not in all_dumps:
                continue
            _log(f"\n=== D. interactions for {f} ===")
            compute_interactions(f, sample_rows=args.interaction_sample)

    if do_abraham and "abraham_only" in all_dumps:
        _log("\n=== E. Abraham/LSER axis ranking ===")
        analyze_abraham_axes()

    _log("\nDone.")


if __name__ == "__main__":
    main()
