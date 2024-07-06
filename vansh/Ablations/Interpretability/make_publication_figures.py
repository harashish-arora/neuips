#!/usr/bin/env python
"""
Publication-grade figure builder for the Interpretability ablation.

Reads the precomputed SHAP / GCN dumps under ``results/<feat>/`` and
produces a small, deliberately curated set of figures that go straight
into the paper.  Each figure is saved as both ``.pdf`` (for LaTeX) and
``.png`` (for previews).

Figures emitted (under ``paper_figures/``):

    fig1_blockwise.{pdf,png}            Block-wise SHAP across 7 featurizers
                                         (eval + ood, stacked-bar panel)
    fig2_global_top_features.{pdf,png}  Top-15 global features for the three
                                         chemistry-readable featurizers
                                         (rdkit, dissolvr, abraham_only)
    fig3_per_solvent_heatmap.{pdf,png}  Per-solvent feature importance
                                         heatmap on dissolvr (eval, top-15
                                         features × 25 ID solvents)
    fig4_solvent_dendrograms.{pdf,png}  2x2 panel of solvent clustering:
                                         dissolvr / abraham_only (clean) vs
                                         morgan / atompair (chemistry-blind)
    fig5_abraham_lser.{pdf,png}         Abraham/LSER axis importance per
                                         solvent (solvent + solute side)
    fig6_interactions.{pdf,png}         Top SHAP interaction values
                                         (abraham_only + rdkit side-by-side)
    fig7_gcn.{pdf,png}                  GCN: atom-type ranking (top) +
                                         BRICS fragment ranking (bottom)

Run once after ``run_shap.py``, ``analyze_shap.py``, and ``run_gcn_explain.py``
have populated ``results/``.  ~10 s to regenerate everything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUT_DIR = HERE / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "savefig.bbox":      "tight",
    "savefig.dpi":       200,
    "pdf.fonttype":      42,        # editable text in Illustrator
    "ps.fonttype":       42,
})

# Colour palette (block colours used everywhere)
BLOCK_COLOR = {
    "solute":  "#1f77b4",
    "solvent": "#2ca02c",
    "T":       "#d62728",
}
PAIR_COLOR = {
    "('solute', 'solute')":   BLOCK_COLOR["solute"],
    "('solvent', 'solvent')": BLOCK_COLOR["solvent"],
    "('T', 'T')":              BLOCK_COLOR["T"],
    "('solute', 'solvent')":  "#9467bd",
    "('solute', 'T')":         "#e377c2",
    "('solvent', 'T')":        "#bcbd22",
}

FEATURIZERS_ALL = ["rdkit", "morgan", "dissolvr", "mordred",
                   "maccs", "atompair", "abraham_only"]

FEATURIZER_LABEL = {
    "rdkit":        "RDKit (158)",
    "morgan":       "Morgan ECFP4 (1024)",
    "dissolvr":     "Dissolvr (RDKit + MOSE + Joback + Abr.)",
    "mordred":      "Mordred (1600)",
    "maccs":        "MACCS keys (167)",
    "atompair":     "Atom-Pair FP (1024)",
    "abraham_only": "Abraham-only (5 LSER + Tm)",
}
FEATURIZER_SHORT = {
    "rdkit":        "RDKit",
    "morgan":       "Morgan",
    "dissolvr":     "Dissolvr",
    "mordred":      "Mordred",
    "maccs":        "MACCS",
    "atompair":     "AtomPair",
    "abraham_only": "Abraham",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_metrics(feat: str):
    p = RESULTS_DIR / feat / "metrics.json"
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def _load_blockwise(feat: str) -> dict:
    p = RESULTS_DIR / feat / "global_blockwise.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def _load_global_imp(feat: str, split: str) -> pd.DataFrame:
    p = RESULTS_DIR / feat / f"global_importance__{split}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _load_solvent_sim(feat: str):
    p = RESULTS_DIR / feat / "solvent_similarity.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def _load_shap(feat: str, split: str):
    p = RESULTS_DIR / feat / f"shap_{split}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def _load_feature_names(feat: str):
    p = RESULTS_DIR / feat / "feature_names.json"
    with open(p) as fh:
        return json.load(fh)


def _save(fig, name: str):
    """Save .pdf and .png side-by-side."""
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf.name}  +  {png.name}")


# ===========================================================================
# Figure 1 - Block-wise SHAP attribution
# ===========================================================================

def fig1_blockwise():
    """Stacked bar panel: solute / solvent / T share per featurizer, eval+ood."""
    feats = [f for f in FEATURIZERS_ALL if (RESULTS_DIR / f / "global_blockwise.json").exists()]
    if not feats:
        return

    blocks = ["solute", "solvent", "T"]
    splits = ["eval", "ood"]

    # Sort featurizers by descending solute share on eval (puts the
    # descriptor methods on the left, fingerprints on the right).
    eval_solute = []
    for f in feats:
        b = _load_blockwise(f).get("eval", {})
        eval_solute.append(b.get("solute", {}).get("share", 0.0))
    feats = [f for _, f in sorted(zip(eval_solute, feats), reverse=True)]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, sname in zip(axes, splits):
        x = np.arange(len(feats))
        bottom = np.zeros(len(feats))
        for blk in blocks:
            shares = []
            for f in feats:
                shares.append(_load_blockwise(f).get(sname, {}).get(blk, {}).get("share", 0.0))
            shares = np.array(shares)
            ax.bar(x, shares, bottom=bottom,
                   color=BLOCK_COLOR[blk],
                   edgecolor="black", linewidth=0.5,
                   label=blk if sname == splits[0] else None)
            for xi, (s, b) in enumerate(zip(shares, bottom)):
                if s > 0.04:
                    txt_color = "white" if s > 0.10 else "black"
                    ax.text(xi, b + s / 2, f"{int(round(s*100))}",
                            ha="center", va="center",
                            color=txt_color, fontsize=8.5,
                            fontweight="bold" if s > 0.5 else "normal")
            bottom = bottom + shares

        ax.set_xticks(x)
        ax.set_xticklabels([FEATURIZER_SHORT.get(f, f) for f in feats],
                           rotation=25, ha="right")
        ax.set_ylim(0, 1.005)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
        ax.set_title(f"{sname} split", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Share of total mean(|SHAP|)")
    axes[0].legend(title="block", loc="upper left", bbox_to_anchor=(0.0, -0.18),
                   ncol=3, frameon=False)
    fig.suptitle("Block-wise attribution: solute features dominate, "
                 "solvent share grows on OOD",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_blockwise")


# ===========================================================================
# Figure 2 - Global top features for the chemistry-readable featurizers
# ===========================================================================

def fig2_global_top_features():
    feats = ["rdkit", "dissolvr", "abraham_only"]
    K = 12

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.8))

    for ax, feat in zip(axes, feats):
        df = _load_global_imp(feat, "eval")
        if df.empty:
            continue
        df = df.head(K).iloc[::-1]
        labels = [_pretty_feature(n) for n in df["feature"].tolist()]
        vals = df["mean_abs_shap"].values
        colors = [BLOCK_COLOR[b] for b in df["block"].values]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.012, yi, f"{v:.3f}",
                    va="center", fontsize=7.5, color="0.2")
        m = (_load_metrics(feat) or {}).get("metrics", {}).get("eval", {})
        rmse = m.get("RMSE", float("nan"))
        ps   = m.get("PS_RMSE", float("nan"))
        ax.set_title(
            f"{FEATURIZER_LABEL.get(feat, feat)}\n"
            rf"eval RMSE = {rmse:.3f}, PS-RMSE = {ps:.3f}",
            fontweight="bold", fontsize=9.5,
        )
        ax.set_xlabel(r"global $\mathrm{mean}(|\mathrm{SHAP}|)$ on $\log S$")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlim(0, max(vals) * 1.18)

    # Single legend at the bottom
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=BLOCK_COLOR[k], label=k) for k in ("solute", "solvent", "T")]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Top-12 global features per featurizer (LightGBM, fixed HPs, eval)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig2_global_top_features")


def _pretty_feature(name: str) -> str:
    """Strip the solute_/solv_ prefix and add a small inline tag."""
    if name.startswith("solute_"):
        return name[len("solute_"):] + "  (solute)"
    if name.startswith("solv_"):
        return name[len("solv_"):]   + "  (solv)"
    return name


# ===========================================================================
# Figure 3 - Per-solvent feature importance heatmap (dissolvr, eval)
# ===========================================================================

def fig3_per_solvent_heatmap(feat: str = "dissolvr",
                             split: str = "eval", K: int = 15):
    sd = _load_shap(feat, split)
    if sd is None:
        return
    sv = sd["shap"]
    solv = sd["solvent_names"]
    if len(solv) == 0:
        return
    fnames = _load_feature_names(feat)

    # Top-25 ID solvents on eval
    counts = pd.Series(solv).value_counts()
    target = counts.index[:25].tolist()

    # Top-K features by global mean|SHAP|
    global_mabs = np.abs(sv).mean(0)
    top_idx = np.argsort(-global_mabs)[:K]
    top_names = [_pretty_feature(fnames[i]) for i in top_idx]

    # Per-solvent column-normalised mean|SHAP|
    cols, kept_solvs = [], []
    for s in target:
        mask = solv == s
        if mask.sum() < 5:
            continue
        m = np.abs(sv[mask]).mean(0)[top_idx]
        m_n = m / (m.sum() + 1e-12)
        cols.append(m_n)
        kept_solvs.append(s)
    M = np.array(cols).T   # (K, S)

    # Order solvents by hierarchical linkage on the columns - same look as the
    # dendrogram figure
    from scipy.cluster.hierarchy import linkage, leaves_list
    if len(kept_solvs) > 2:
        Z = linkage(M.T, method="average", metric="cosine")
        order = leaves_list(Z)
        M = M[:, order]
        kept_solvs = [kept_solvs[i] for i in order]

    fig, ax = plt.subplots(figsize=(0.42 * len(kept_solvs) + 2.5,
                                    0.36 * K + 1.8))
    vmax = float(M.max())
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(kept_solvs)))
    ax.set_xticklabels(kept_solvs, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels(top_names, fontsize=8.5)
    ax.set_title(
        f"Per-solvent SHAP profile (Dissolvr, eval) - "
        "top-15 features, column-normalised so each solvent column sums to 1",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("solvent (ordered by hierarchical-cluster leaf order)")
    ax.set_ylabel("feature (descending global importance)")
    ax.tick_params(top=False, right=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012, shrink=0.85)
    cb.set_label("col-normalised mean(|SHAP|)")
    cb.outline.set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, "fig3_per_solvent_heatmap")


# ===========================================================================
# Figure 4 - Solvent clustering (descriptor vs fingerprint)
# ===========================================================================

def fig4b_solvent_clusters_2d():
    """2-D MDS embedding of per-solvent SHAP fingerprints, coloured by
    K-means cluster (K=4) and shape-coded by chemist's classical family.

    Distance metric is exactly the same one the dendrogram uses
    (1 - cos(SHAP fingerprint)) so the scatter plot and the dendrogram are
    just two visualisations of the same underlying matrix.

    The four classical solvent families (used as marker shapes):

      water      *   (singleton, the prototype protic-extreme)
      apolar     s   (alkanes)
      aprotic    ^   (DMF, THF, chloroform, toluene, esters, ketones, dioxane,
                       acetonitrile)
      protic     o   (alcohols, glycols, n-octanol)

    Goodness-of-fit between model clustering and chemist family is reported
    in each panel as Adjusted Rand Index (ARI in [-1, 1]; 1 = perfect agreement,
    0 = random).
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.manifold import MDS
    from sklearn.metrics import adjusted_rand_score

    # Hand-curated chemist family labels for the 25 ID solvents.
    # Source: Snyder solvent classification + standard organic-chem texts.
    SOLVENT_FAMILY = {
        "water":          "water",
        "n-hexane":       "apolar",
        "cyclohexane":    "apolar",
        "toluene":        "aprotic",
        "chloroform":     "aprotic",
        "DMF":            "aprotic",
        "THF":            "aprotic",
        "1,4-dioxane":    "aprotic",
        "ethyl acetate":  "aprotic",
        "methyl acetate": "aprotic",
        "n-propyl acetate": "aprotic",
        "n-butyl acetate":  "aprotic",
        "acetone":        "aprotic",
        "2-butanone":     "aprotic",
        "acetonitrile":   "aprotic",
        "methanol":       "protic",
        "ethanol":        "protic",
        "n-propanol":     "protic",
        "n-butanol":      "protic",
        "n-pentanol":     "protic",
        "isopropanol":    "protic",
        "isobutanol":     "protic",
        "sec-butanol":    "protic",
        "n-octanol":      "protic",
        "ethylene glycol": "protic",
    }

    FAMILY_MARKER = {
        "water":   "*",
        "apolar":  "s",
        "aprotic": "^",
        "protic":  "o",
    }
    FAMILY_SIZE = {
        "water":   320,
        "apolar":  220,
        "aprotic": 150,
        "protic":  150,
    }

    # Cluster colours - distinct, colourblind-friendly
    CLUSTER_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                      "#ff7f0e", "#17becf", "#bcbd22"]
    K_CLUSTERS = 4

    panels = [
        ("dissolvr",     "Dissolvr (descriptors + Joback + Abr.)"),
        ("abraham_only", "Abraham-only (5 LSER + Tm)"),
        ("morgan",       "Morgan ECFP4 (fingerprint)"),
        ("atompair",     "Atom-Pair fingerprint"),
    ]

    def _plot_one_panel(ax, feat: str, label: str, *,
                        title_fontsize: float = 11.5,
                        label_fontsize: float = 9):
        sim = _load_solvent_sim(feat)
        if sim is None:
            ax.set_visible(False)
            return None

        S = sim["sim"]
        names = list(sim["solvents"])
        dist = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        dist = 0.5 * (dist + dist.T)

        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=0, n_init=8, normalized_stress="auto")
        emb = mds.fit_transform(dist)

        cond = dist[np.triu_indices_from(dist, k=1)]
        Z = linkage(cond, method="average")
        cluster_labels = fcluster(Z, t=K_CLUSTERS, criterion="maxclust")

        family_str = [SOLVENT_FAMILY.get(n, "unknown") for n in names]
        fam_to_int = {f: i for i, f in enumerate(sorted(set(family_str)))}
        family_int = [fam_to_int[f] for f in family_str]
        ari = adjusted_rand_score(family_int, cluster_labels)

        for i, name in enumerate(names):
            fam = SOLVENT_FAMILY.get(name, "aprotic")
            marker = FAMILY_MARKER[fam]
            size = FAMILY_SIZE[fam]
            color = CLUSTER_COLORS[(cluster_labels[i] - 1) % len(CLUSTER_COLORS)]
            ax.scatter(emb[i, 0], emb[i, 1],
                       marker=marker, s=size,
                       facecolor=color, edgecolor="black", linewidth=1.3,
                       alpha=1.0, zorder=6)

        from scipy.spatial import ConvexHull
        for c in np.unique(cluster_labels):
            pts = emb[cluster_labels == c]
            if len(pts) < 3:
                continue
            try:
                hull = ConvexHull(pts)
                hpts = pts[hull.vertices]
                ax.fill(hpts[:, 0], hpts[:, 1],
                        color=CLUSTER_COLORS[(c - 1) % len(CLUSTER_COLORS)],
                        alpha=0.12, zorder=1, lw=0)
            except Exception:
                pass

        x_range = emb[:, 0].max() - emb[:, 0].min()
        y_range = emb[:, 1].max() - emb[:, 1].min()
        label_xy, anchor_xy = _spread_labels(
            emb, [len(n) for n in names], x_range, y_range,
            n_iter=220,
        )
        for i, name in enumerate(names):
            lx, ly = label_xy[i]
            ax_, ay_ = anchor_xy[i]
            d = np.hypot(lx - ax_, ly - ay_)
            if d > 0.04 * max(x_range, y_range):
                ax.plot([ax_, lx], [ay_, ly],
                        color="0.55", lw=0.6, alpha=0.65, zorder=2)
            ax.annotate(
                name, xy=(lx, ly),
                fontsize=label_fontsize, ha="center", va="center",
                color="0.08", fontweight="normal",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="white",
                          edgecolor="none", alpha=0.85),
                zorder=5,
            )

        ax.set_title(
            f"{label}\nARI vs chemist family = {ari:.2f}",
            fontweight="bold", fontsize=title_fontsize, pad=8,
        )
        ax.set_xlabel("MDS-1", fontsize=9)
        ax.set_ylabel("MDS-2", fontsize=9)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(0.22)
        return ari

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 13.5))
    for ax, (feat, label) in zip(axes.flat, panels):
        _plot_one_panel(ax, feat, label)

    # ---- Single legend (marker shapes = families) ----
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=FAMILY_MARKER[f], color="w",
               markerfacecolor="0.75", markeredgecolor="black",
               markersize=np.sqrt(FAMILY_SIZE[f]),
               label=f"chemist family: {f}")
        for f in ("water", "apolar", "aprotic", "protic")
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005),
        fontsize=9.5,
    )
    fig.suptitle(
        "Solvent clustering by SHAP-fingerprint similarity (2-D MDS)\n"
        "Distance = $1 - \\cos(\\mathrm{SHAP\\ fingerprint})$.   "
        "Marker shape = chemist family (textbook Snyder classification).   "
        "Marker colour = automatic cluster (K=4 hierarchical, average linkage on the same distance).   "
        "ARI ($\\in [-1,1]$) measures how well the two groupings agree.",
        fontsize=10.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    _save(fig, "fig4b_solvent_clusters_2d")

    # ------------------------------------------------------------------
    # Hero panel: just abraham_only at large size (cleanest example).
    # ------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(11, 9))
    ari_hero = _plot_one_panel(
        ax2, "abraham_only",
        "Abraham-only featurizer  (5 LSER + Tm)  -  the model's solvent map",
        title_fontsize=12.5, label_fontsize=10,
    )

    legend_handles_2 = [
        Line2D([0], [0], marker=FAMILY_MARKER[f], color="w",
               markerfacecolor="0.75", markeredgecolor="black",
               markersize=np.sqrt(FAMILY_SIZE[f]),
               label=f"chemist family: {f}")
        for f in ("water", "apolar", "aprotic", "protic")
    ]
    fig2.legend(
        handles=legend_handles_2, loc="lower center",
        ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005),
        fontsize=10,
    )
    fig2.suptitle(
        "Solvent map: 2-D MDS embedding of per-solvent SHAP fingerprints\n"
        "Marker shape = chemist's classical family.   "
        "Marker colour = automatic K=4 cluster on the SHAP fingerprint distance.",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig2.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig2, "fig4c_solvent_map_hero")


def _spread_labels(points: np.ndarray, label_lengths: list[int],
                   data_xrange: float, data_yrange: float,
                   n_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Iterative repulsion to spread out crowded text labels.

    Each label is initialised at its anchor point + a small radial offset.
    On each iteration:
      1. Pull each label towards its anchor (spring).
      2. Push pairs of labels that overlap apart (repulsion).
      3. Push labels away from any non-anchor point they're sitting on top of.

    Returns:
        label_xy : (N, 2) final label positions in data coordinates
        anchor_xy: (N, 2) the original anchor points (unchanged)
    """
    n = len(points)
    if n == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # Approximate label half-width / half-height in data units, given the
    # visible character count and a font size of ~9 pt with bbox padding.
    char_w = 0.014 * data_xrange       # per character (incl. bbox padding)
    line_h = 0.038 * data_yrange       # one line of text
    half_w = np.array([0.5 * char_w * max(L, 4) for L in label_lengths])
    half_h = np.full(n, 0.5 * line_h)

    # Initialise label positions: small radial offset from the centre of mass.
    centre = points.mean(axis=0)
    init_dir = points - centre
    init_norm = np.linalg.norm(init_dir, axis=1, keepdims=True).clip(min=1e-9)
    init_offset = init_dir / init_norm * 0.05 * max(data_xrange, data_yrange)
    labels = points + init_offset

    spring_k = 0.06
    repel_k = 0.18
    point_repel_k = 0.06

    rng = np.random.RandomState(0)

    for it in range(n_iter):
        forces = np.zeros_like(labels)

        # Spring back to anchor
        forces += spring_k * (points - labels)

        # Pairwise repulsion between labels (rectangular overlap test)
        for i in range(n):
            for j in range(i + 1, n):
                dx = labels[i, 0] - labels[j, 0]
                dy = labels[i, 1] - labels[j, 1]
                min_dx = half_w[i] + half_w[j] + 0.1 * char_w
                min_dy = half_h[i] + half_h[j] + 0.05 * line_h
                if abs(dx) < min_dx and abs(dy) < min_dy:
                    overlap_x = (min_dx - abs(dx)) * np.sign(dx if dx != 0 else 1.0)
                    overlap_y = (min_dy - abs(dy)) * np.sign(dy if dy != 0 else 1.0)
                    f = repel_k * np.array([overlap_x, overlap_y])
                    forces[i] += f
                    forces[j] -= f

        # Repel labels away from non-anchor points (avoid sitting on a marker)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dx = labels[i, 0] - points[j, 0]
                dy = labels[i, 1] - points[j, 1]
                # marker radius ~ 0.025 of the smaller axis range
                r = 0.025 * min(data_xrange, data_yrange)
                d = np.hypot(dx, dy)
                if d < r and d > 1e-9:
                    forces[i] += point_repel_k * np.array([dx, dy]) / d * (r - d)

        labels = labels + forces
        # Tiny noise to break symmetric ties (only first 30 iters)
        if it < 30:
            labels += rng.randn(*labels.shape) * 0.002 * data_xrange

    return labels, points


def fig4_solvent_dendrograms():
    panels = [
        ("dissolvr",     "Dissolvr (descriptors + Joback + Abr.)"),
        ("abraham_only", "Abraham-only (5 LSER + Tm)"),
        ("morgan",       "Morgan ECFP4 (fingerprint)"),
        ("atompair",     "Atom-Pair fingerprint"),
    ]
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Wider canvas + 2x1 layout (instead of 2x2) so each dendrogram has the
    # full width to spread its 25 leaf labels.
    fig, axes = plt.subplots(4, 1, figsize=(13, 13))
    for ax, (feat, label) in zip(axes, panels):
        sim = _load_solvent_sim(feat)
        if sim is None:
            ax.set_visible(False)
            continue
        S = sim["sim"]
        names = list(sim["solvents"])
        dist = 1.0 - S
        np.fill_diagonal(dist, 0.0)
        cond = dist[np.triu_indices_from(dist, k=1)]
        Z = linkage(cond, method="average")

        # Choose a colour threshold that gives ~2-4 clusters for clarity
        max_h = float(Z[:, 2].max())
        ct = 0.62 * max_h
        dendrogram(Z, labels=names, ax=ax,
                   leaf_rotation=42, leaf_font_size=8.5,
                   color_threshold=ct,
                   above_threshold_color="0.55")
        ax.set_ylabel(r"$1 - \cos(\mathrm{SHAP\ fingerprint})$",
                      fontsize=9)
        ax.set_title(label, fontweight="bold", fontsize=10.5,
                     loc="left", pad=4)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Add a bit of space below x labels
        ax.tick_params(axis="x", pad=2)

    fig.suptitle(
        "Solvent clustering by SHAP-fingerprint similarity\n"
        "Descriptor / Abraham representations (top) recover classical "
        "solvent families.  Circular fingerprints (bottom) group by shared "
        "bits, not by chemistry.",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig.tight_layout(h_pad=2.5)
    _save(fig, "fig4_solvent_dendrograms")


# ===========================================================================
# Figure 5 - Abraham/LSER axis importance per solvent
# ===========================================================================

def fig5_abraham_lser():
    p = RESULTS_DIR / "abraham_only" / "abraham_axis_ranking.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    axes_order = ["A", "B", "S", "E", "V", "Tm"]
    df["axis"] = df["axis"].str.replace("abraham_", "", regex=False).str.replace("pred_", "", regex=False)
    df = df[df["axis"].isin(axes_order)]

    fig, axarr = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
    # Order solvents alphabetically grouped by family for readability:
    family_order = [
        # apolar
        "n-hexane", "cyclohexane",
        # weakly polar / aromatic / aprotic
        "toluene", "chloroform", "1,4-dioxane", "THF", "DMF", "ethyl acetate",
        "methyl acetate", "n-propyl acetate", "n-butyl acetate",
        "acetone", "2-butanone", "acetonitrile",
        # protic
        "methanol", "ethanol", "n-propanol", "n-butanol", "n-pentanol",
        "isopropanol", "isobutanol", "sec-butanol", "n-octanol",
        "ethylene glycol",
        # special
        "water",
    ]

    vmax = float(df["mean_abs_shap"].max())
    for ax, side, title in [
        (axarr[0], "solvent", "(a) solvent-side axes"),
        (axarr[1], "solute",  "(b) solute-side axes"),
    ]:
        sub = df[df["side"] == side]
        if sub.empty:
            continue
        pivot = sub.pivot(index="solvent", columns="axis",
                          values="mean_abs_shap")
        pivot = pivot.reindex([s for s in family_order if s in pivot.index])
        pivot = pivot[[a for a in axes_order if a in pivot.columns]]

        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index), fontsize=8.5)
        for i in range(len(pivot)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isnan(v):
                    continue
                txt_color = "white" if v < 0.55 * vmax else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color)
        ax.set_xlabel("Abraham/LSER axis")
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.tick_params(top=False, right=False)
        # group separators (apolar | aprotic | protic | water)
        for sep in (1.5, 13.5, 23.5):
            if sep < len(pivot) - 0.5:
                ax.axhline(sep, color="white", lw=0.8)

    cb = fig.colorbar(im, ax=axarr.ravel().tolist(), fraction=0.025, pad=0.02,
                      shrink=0.85)
    cb.set_label(r"per-solvent $\mathrm{mean}(|\mathrm{SHAP}|)$ on $\log S$")
    cb.outline.set_linewidth(0.5)

    fig.suptitle(
        "Abraham/LSER axis importance per solvent\n"
        "Solvents grouped (top-bottom): apolar | aprotic-aromatic & ester & ketone | protic alcohols | water",
        fontsize=10.5, fontweight="bold", y=1.0,
    )
    _save(fig, "fig5_abraham_lser")


# ===========================================================================
# Figure 6 - SHAP interaction values
# ===========================================================================

def fig6_interactions():
    feats = [
        ("abraham_only", "Abraham-only (16 features, exact, sample = 1500)"),
        ("rdkit",        "RDKit (320 features, exact, sample = 200)"),
    ]
    K = 15
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))

    for ax, (feat, label) in zip(axes, feats):
        p = RESULTS_DIR / feat / "interactions_top50__eval.csv"
        if not p.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(p).head(K).iloc[::-1]
        labels = [
            f"{_pretty_feature(a)}  ↔  {_pretty_feature(b)}"
            for a, b in zip(df["feature_a"], df["feature_b"])
        ]
        vals = df["abs_mean_interaction"].values
        colors = [PAIR_COLOR.get(bp, "#444") for bp in df["block_pair"]]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.012, yi, f"{v:.3f}",
                    va="center", fontsize=7, color="0.2")

        m = (_load_metrics(feat) or {}).get("metrics", {}).get("eval", {})
        rmse = m.get("RMSE", float("nan"))
        ax.set_title(f"{label}\nRMSE = {rmse:.3f}",
                     fontweight="bold", fontsize=9.5)
        ax.set_xlabel(r"$\mathrm{mean}(|\Phi_{ij}|)$ - Tree-SHAP interaction")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xlim(0, max(vals) * 1.22)

    # Legend (only show pair-types that actually appear in either panel)
    from matplotlib.patches import Patch
    used = set()
    for feat, _ in feats:
        p = RESULTS_DIR / feat / "interactions_top50__eval.csv"
        if not p.exists():
            continue
        used.update(pd.read_csv(p).head(K)["block_pair"].unique())
    legend_handles = []
    pretty = {
        "('solute', 'solute')":   "solute × solute",
        "('solvent', 'solvent')": "solvent × solvent",
        "('T', 'T')":              "T × T",
        "('solute', 'solvent')":  "solute × solvent",
        "('solute', 'T')":         "solute × T",
        "('solvent', 'T')":        "solvent × T",
    }
    for k in [
        "('solute', 'solute')",
        "('solute', 'solvent')",
        "('solvent', 'solvent')",
        "('solute', 'T')",
        "('solvent', 'T')",
    ]:
        if k in used:
            legend_handles.append(Patch(facecolor=PAIR_COLOR[k], label=pretty[k]))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Top-15 SHAP interaction features - "
        "Abraham's solvation cross-terms emerge spontaneously",
        fontsize=10.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig6_interactions")


# ===========================================================================
# Figure 7 - GCN: atom-type ranking + BRICS fragment ranking
# ===========================================================================

def fig7_gcn():
    p_atom = RESULTS_DIR / "gcn" / "atom_occlusion_by_atom_type.csv"
    p_brics = RESULTS_DIR / "gcn" / "brics_fragment_importance.csv"
    if not p_atom.exists() or not p_brics.exists():
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 6.2),
                                     gridspec_kw={"width_ratios": [0.45, 0.55]})

    # ------- Left: atom-type ranking -------
    by_kind = pd.read_csv(p_atom)
    # Pretty labels: split kind into element + (aromatic/ring) tags
    def _pretty_kind(k: str):
        # k = "S_ar0_R0"
        elem, ar, ring = k.split("_")
        ar = "aromatic" if ar == "ar1" else ""
        ring = "ring" if ring == "R1" else ""
        tags = ", ".join([t for t in [ar, ring] if t]) or "acyclic"
        return f"{elem}  ({tags})"

    K_A = 14
    by_kind = by_kind.head(K_A).iloc[::-1]
    labels = [_pretty_kind(k) for k in by_kind["atom_kind"]]
    means = by_kind["mean_abs_delta"].values
    err = by_kind["std_abs_delta"].values / np.sqrt(by_kind["n"].values.clip(min=1))
    y = np.arange(len(labels))
    ax_a.barh(y, means, xerr=err,
              color="#2ca02c", edgecolor="black", linewidth=0.4,
              error_kw={"elinewidth": 0.7, "ecolor": "0.3", "capsize": 1.6})
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=8)
    for yi, m, n in zip(y, means, by_kind["n"]):
        ax_a.text(m + 0.02, yi, f"  {m:.2f}  (n={int(n)})",
                  va="center", fontsize=7, color="0.2")
    ax_a.set_xlabel(r"mean $|\Delta \log S|$  per atom-occlusion")
    ax_a.set_title("(a) Atom-type importance via occlusion",
                   fontweight="bold", fontsize=10)
    ax_a.grid(axis="x", alpha=0.3)
    ax_a.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_a.spines[spine].set_visible(False)
    ax_a.set_xlim(0, means.max() * 1.45)

    # ------- Right: BRICS fragments -------
    df = pd.read_csv(p_brics)
    df = df[df["fragment"].apply(lambda s: isinstance(s, str) and s and not s.startswith("<"))]
    df = df[df["n"] >= 30]

    K_B = 15
    df = df.sort_values("sum_abs_delta", ascending=False).head(K_B).iloc[::-1]
    labels = [_pretty_brics(f) for f in df["fragment"]]
    vals = df["sum_abs_delta"].values
    n_arr = df["n"].values

    y = np.arange(len(labels))
    ax_b.barh(y, vals, color="#9467bd", edgecolor="black", linewidth=0.4)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=7.8)
    for yi, v, n in zip(y, vals, n_arr):
        ax_b.text(v + max(vals) * 0.012, yi, f"  n={int(n)}",
                  va="center", fontsize=7, color="0.2")
    ax_b.set_xlabel(r"$\sum |\Delta \log S|$  over the eval sample")
    ax_b.set_title("(b) Top BRICS fragments (≥30 atom occurrences)",
                   fontweight="bold", fontsize=10)
    ax_b.grid(axis="x", alpha=0.3)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)
    ax_b.set_xlim(0, max(vals) * 1.18)

    fig.suptitle(
        "GCN dual-encoder interpretability: atoms and BRICS substructures driving the prediction\n"
        "(occlusion attribution, 5 000 eval rows; sanity-check eval RMSE = 0.608, ood RMSE = 0.786)",
        fontsize=10.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig7_gcn")


# ===========================================================================
# Figure 8 - Signed SHAP directionality
# ===========================================================================

def fig8_signed_shap():
    """Signed/directional SHAP view.

    Most figures in the interpretation section use mean(|SHAP|), which answers
    "what matters?".  This figure answers the complementary question
    "which direction does it push logS?"  For a feature f we compute Spearman's
    rho between its raw feature value X_f and its SHAP contribution phi_f on
    the eval split:

        rho_f = corr_rank(X_f, phi_f)

    rho > 0 means high values of the feature tend to *increase* the predicted
    logS; rho < 0 means high values tend to *decrease* it.  Bar length remains
    mean(|SHAP|), so the panel simultaneously shows magnitude and direction.
    """
    sys.path.insert(0, str(HERE.parent.parent))
    from sc3_bench.data import load_cached_features
    from scipy.stats import spearmanr

    panels = [
        ("rdkit", "RDKit descriptors", 16),
        ("abraham_only", "Abraham-only LSER axes", 16),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4))
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    for ax, (feat, title, K) in zip(axes, panels):
        sd = _load_shap(feat, "eval")
        if sd is None:
            ax.set_visible(False)
            continue
        shap_vals = sd["shap"]
        fnames = _load_feature_names(feat)
        cached = load_cached_features(feat)
        if cached is None:
            ax.set_visible(False)
            continue
        X = cached["X_eval"]
        if X.shape[1] != len(fnames):
            m = min(X.shape[1], len(fnames))
            X = X[:, :m]
            shap_vals = shap_vals[:, :m]
            fnames = fnames[:m]

        mean_abs = np.abs(shap_vals).mean(axis=0)
        mean_signed = shap_vals.mean(axis=0)
        rhos = []
        for j in range(X.shape[1]):
            xj = X[:, j]
            sj = shap_vals[:, j]
            if np.nanstd(xj) < 1e-12 or np.nanstd(sj) < 1e-12:
                rhos.append(np.nan)
                continue
            try:
                rhos.append(float(spearmanr(xj, sj, nan_policy="omit").statistic))
            except Exception:
                rhos.append(np.nan)
        rhos = np.array(rhos, dtype=float)

        out = pd.DataFrame({
            "feature": fnames,
            "block": [
                "T" if n in ("T_norm", "T_inv", "T_sq", "T_log")
                else "solvent" if n.startswith("solv_")
                else "solute"
                for n in fnames
            ],
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "spearman_feature_vs_shap": rhos,
        }).sort_values("mean_abs_shap", ascending=False)
        out.to_csv(RESULTS_DIR / feat / "signed_shap__eval.csv", index=False)

        top = out.head(K).iloc[::-1]
        labels = [_pretty_feature(n) for n in top["feature"]]
        vals = top["mean_abs_shap"].values
        rho_vals = top["spearman_feature_vs_shap"].values
        colors = [cmap(norm(0.0 if np.isnan(r) else r)) for r in rho_vals]

        y = np.arange(len(top))
        ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.45)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for yi, v, r in zip(y, vals, rho_vals):
            r_txt = "n/a" if np.isnan(r) else f"{r:+.2f}"
            ax.text(v + max(vals) * 0.015, yi, r"$\rho$=" + r_txt,
                    va="center", fontsize=7.5, color="0.15")
        ax.set_xlim(0, max(vals) * 1.32)
        ax.set_xlabel(r"global $\mathrm{mean}(|\mathrm{SHAP}|)$ on eval")
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Compact legend instead of a full colorbar: the actual direction score
    # is printed next to every bar as rho, so the legend only needs to explain
    # the sign of the colour.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=cmap(norm(-0.85)), edgecolor="black",
              label=r"high feature value lowers predicted $\log S$"),
        Patch(facecolor=cmap(norm(0.0)), edgecolor="black",
              label=r"weak / mixed direction"),
        Patch(facecolor=cmap(norm(+0.85)), edgecolor="black",
              label=r"high feature value raises predicted $\log S$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=8.5)

    fig.suptitle(
        "Signed SHAP directionality: what pushes predicted solubility up or down?",
        fontsize=11.2, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    _save(fig, "fig8_signed_shap")


# ===========================================================================
# Figure 9 - Example GCN molecule attributions
# ===========================================================================

def fig9_gcn_molecule_examples():
    """Draw representative molecules with GCN atom occlusion weights.

    The aggregate BRICS panel (Fig. 7) says which fragments matter globally.
    This figure makes the graph explanation concrete: individual molecules are
    drawn with atoms painted by |delta logS| when that atom is occluded.
    """
    occ_path = RESULTS_DIR / "gcn" / "atom_occlusion__eval.csv.gz"
    if not occ_path.exists():
        return

    import pickle
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
    from PIL import Image

    sys.path.insert(0, str(HERE.parent.parent))
    from sc3_bench.data import load_all_splits
    from sc3_bench.registry import get_hp
    from sc3_bench.models.gnn_models import (
        DualGNNSolubility, smiles_to_graph, batch_graph_list,
    )
    splits = load_all_splits(verbose=False)
    eval_df = splits["eval"].reset_index(drop=True)

    occ = pd.read_csv(occ_path)
    # Add row-level y_true and error context (not needed for drawing, useful in captions).
    row_meta = eval_df[["LogS", "Solute", "Solvent_Name"]].copy()
    row_meta["row_idx"] = np.arange(len(row_meta))
    mol_rows = (
        occ.groupby(
            ["row_idx", "solute_smiles", "solvent_smiles", "solvent_name", "T", "base_pred"],
            as_index=False,
        )
        .agg(total_abs_delta=("abs_delta", "sum"),
             max_abs_delta=("abs_delta", "max"),
             n_atoms=("atom_idx", "count"))
        .merge(row_meta[["row_idx", "LogS"]], on="row_idx", how="left")
    )

    motif_specs = [
        ("sulfur-rich / sulfone", Chem.MolFromSmarts("[S]")),
        ("carboxylic acid", Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]")),
        ("phenol / aromatic OH", Chem.MolFromSmarts("c[OX2H]")),
        ("piperazine / diamine", Chem.MolFromSmarts("N1CCNCC1")),
        ("nitro-aromatic", Chem.MolFromSmarts("[N+](=O)[O-]")),
    ]

    selected = []
    used_smiles = set()
    for label, patt in motif_specs:
        if patt is None:
            continue
        candidates = []
        for _, r in mol_rows.iterrows():
            smi = r["solute_smiles"]
            if smi in used_smiles:
                continue
            mol = Chem.MolFromSmiles(smi)
            if (
                mol is not None
                and 8 <= mol.GetNumAtoms() <= 34
                and mol.HasSubstructMatch(patt)
            ):
                candidates.append((float(r["total_abs_delta"]), label, r))
        if not candidates:
            continue
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, label, row = candidates[0]
        selected.append((label, row))
        used_smiles.add(row["solute_smiles"])
        if len(selected) == 4:
            break

    # Fallback: highest attribution molecules if one motif was missing.
    if len(selected) < 4:
        for _, row in mol_rows.sort_values("total_abs_delta", ascending=False).iterrows():
            mol = Chem.MolFromSmiles(row["solute_smiles"])
            if mol is None or not (8 <= mol.GetNumAtoms() <= 34):
                continue
            if row["solute_smiles"] not in used_smiles:
                selected.append(("high-attribution molecule", row))
                used_smiles.add(row["solute_smiles"])
            if len(selected) == 4:
                break

    if not selected:
        return

    # Reconstruct the already-trained seed-42 GCN on CPU.  This is only used
    # for four molecules, so no GPU is needed (and the hulk GPU workaround is
    # irrelevant here).
    hp = get_hp("gcn")
    model = DualGNNSolubility(
        node_dim=7, hidden_dim=hp.get("hidden_dim", 96),
        num_layers=hp.get("num_layers", 4), gnn_type="GCN",
    )
    model_path = HERE.parent.parent / "results" / "gcn" / "model_seed_42.pkl"
    # The state dict was pickled after GPU training, so tensor storages may
    # carry CUDA device tags.  Force all storages onto CPU while unpickling so
    # this figure remains CPU-only even on hulk while /dev/nvidia0 is broken.
    import torch.storage
    _orig_load_from_bytes = torch.storage._load_from_bytes
    try:
        torch.storage._load_from_bytes = (
            lambda b: torch.load(BytesIO(b), map_location=torch.device("cpu"), weights_only=False)
        )
        state = pickle.load(open(model_path, "rb"))
    finally:
        torch.storage._load_from_bytes = _orig_load_from_bytes
    model.load_state_dict(state, strict=True)
    model.eval()

    @torch.no_grad()
    def _predict_one(sol_g, solv_g, T):
        sol_b = batch_graph_list([sol_g])
        solv_b = batch_graph_list([solv_g])
        tf = torch.tensor(
            [[T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)]],
            dtype=torch.float32,
        )
        return float(model(sol_b, solv_b, tf).cpu().numpy()[0])

    @torch.no_grad()
    def _signed_atom_weights(sol_smi, solv_smi, T):
        sol_g = smiles_to_graph(sol_smi)
        solv_g = smiles_to_graph(solv_smi)
        if sol_g is None or solv_g is None:
            return None
        base = _predict_one(sol_g, solv_g, T)
        weights = []
        for ai in range(sol_g["num_nodes"]):
            nf = sol_g["node_feats"].clone()
            nf[ai] = 0.0
            sol_occ = {
                "node_feats": nf,
                "edge_index": sol_g["edge_index"],
                "num_nodes": sol_g["num_nodes"],
            }
            occluded = _predict_one(sol_occ, solv_g, T)
            # Positive means this atom raises the predicted logS relative to
            # the same graph with the atom features zeroed.  Negative means
            # the atom lowers predicted solubility.  This gives red/blue
            # directionality analogous to signed SHAP maps.
            weights.append(base - occluded)
        return np.array(weights, dtype=float), base

    # Global signed colour scale over selected atoms, clipped to the 95th
    # percentile of absolute contribution.
    signed_cache = {}
    vals_all = []
    for _, row in selected:
        result = _signed_atom_weights(
            row["solute_smiles"], row["solvent_smiles"], float(row.get("T", 298.15))
        )
        if result is None:
            continue
        signed_cache[int(row["row_idx"])] = result
        vals_all.extend(np.abs(result[0]).tolist())
    vmax = max(np.percentile(vals_all, 95), 1e-6)
    cmap = plt.get_cmap("YlOrRd")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4))
    axes = axes.ravel()

    for panel_i, (ax, (motif_label, row)) in enumerate(zip(axes, selected)):
        row_idx = int(row["row_idx"])
        smi = row["solute_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ax.set_visible(False)
            continue
        AllChem.Compute2DCoords(mol)

        weights, base = signed_cache.get(row_idx, (None, None))
        if weights is None:
            ax.set_visible(False)
            continue

        # SimilarityMaps produces the red/blue smooth halos shown in the
        # reference image: red atoms increase predicted logS; blue atoms lower
        # predicted logS.
        drawer = rdMolDraw2D.MolDraw2DCairo(560, 380)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.padding = 0.06
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, list(weights), drawer,
            colorMap="bwr", scale=vmax,
            contourLines=9, alpha=0.45,
            size=(560, 380),
        )
        drawer.FinishDrawing()
        img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB")

        ax.imshow(img)
        ax.axis("off")

        err = float(row["base_pred"] - row["LogS"]) if pd.notna(row.get("LogS")) else np.nan
        panel_letter = "abcd"[panel_i]
        ax.text(
            0.01, 0.98, panel_letter,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=16, fontweight="bold",
        )
        # Short, clean in-panel label.  Details belong in the caption.
        ax.text(
            0.5, 0.02,
            f"{motif_label}\nsolvent={row['solvent_name']};  "
            f"pred={base:+.2f}, y={float(row['LogS']):+.2f}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.0, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.75),
        )

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap="bwr", norm=mcolors.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.01, shrink=0.78)
    cb.set_label(
        r"signed atom occlusion contribution to predicted $\log S$"
        r"  (red raises, blue lowers; clipped at selected 95th percentile)"
    )
    cb.outline.set_linewidth(0.5)

    fig.suptitle(
        "Example GCN graph attributions: red/blue halos show signed atom effects",
        fontsize=11.2, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 0.96, 0.97), h_pad=1.6, w_pad=1.4)
    _save(fig, "fig9_gcn_molecule_examples")


# Friendly chemistry names for the SMILES of common BRICS fragments,
# falls back to the SMILES string itself.
_BRICS_NAMES = {
    "c1ccccc1":               "benzene  (c1ccccc1)",
    "O=CO":                   "carboxylic acid -COOH  (O=CO)",
    "CO":                     "alcohol -OH  (CO)",
    "C=O":                    "carbonyl C=O",
    "CC=O":                   "acetaldehyde-like CC=O",
    "Oc1ccccc1":              "phenol  (Oc1ccccc1)",
    "CC(=O)O":                "acetic-acid motif  CC(=O)O",
    "N":                      "amine N",
    "C1CNCCN1":               "piperazine  (C1CNCCN1)",
    "Clc1ccccc1":             "chlorobenzene",
    "CCC(=O)O":               "propionic-acid motif",
    "c1ccncc1":               "pyridine",
    "c1ccc2ccccc2c1":         "naphthalene",
    "CC":                     "ethyl  CC",
    "O=[N+]([O-])c1ccccc1":   "nitrobenzene",
    "O":                      "water-like O",
    "c1ccc2[nH]ccc2c1":       "indole-like",
    "Cc1ccccc1":              "toluene-like",
    "FC(F)F":                 "trifluoromethyl",
    "c1ccc2cccnc2c1":         "quinoline-like",
}


def _pretty_brics(s: str) -> str:
    if s in _BRICS_NAMES:
        return _BRICS_NAMES[s]
    if len(s) > 30:
        return s[:27] + "..."
    return s


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"Writing publication figures to: {OUT_DIR}")
    fig1_blockwise()
    fig2_global_top_features()
    fig3_per_solvent_heatmap()
    fig4_solvent_dendrograms()
    fig4b_solvent_clusters_2d()
    fig5_abraham_lser()
    fig6_interactions()
    fig7_gcn()
    fig8_signed_shap()
    fig9_gcn_molecule_examples()
    print("\nDone.  PDF + PNG side-by-side under paper_figures/.")


if __name__ == "__main__":
    main()
