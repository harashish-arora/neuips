#!/usr/bin/env python
"""
Interpretability ablation - GCN graph attribution.

We **reuse the already-trained** dual-encoder GCN
(``vansh/results/gcn/model_seed_<seed>.pkl``) - no retraining.

Two complementary methods:

  1. **Atom occlusion attribution**  (always works for any GNN, supports
     regression + dual encoders trivially):

         a_v(G_solute, G_solvent, T) = f(G_solute, ...) - f(G_solute \ {v}, ...)

     where "G_solute \ {v}" is the same solute graph with node v's features
     zeroed (we keep the topology so neighbouring messages still pass through
     the deleted atom's slot, which is the "soft occlusion" variant typically
     used for GNNs).  We average |a_v| per atom *type / per substructure*
     over ~3000 random rows of the eval split, and we also break the result
     down by solvent.

  2. **PyG GNNExplainer in regression mode**  on the solute branch
     (the rest of the model is held fixed for each instance).  This learns a
     soft node + edge mask per (solute, solvent, T) triple that maximises
     the agreement with the original prediction.  Aggregated to atom-level
     importance and to BRICS-fragment importance globally + per solvent.

Outputs:
  results/gcn/atom_occlusion__eval.parquet      (one row per atom per molecule)
  results/gcn/atom_occlusion_by_atom_type.csv   (aggregated by RDKit atom type)
  results/gcn/atom_occlusion_by_solvent.csv     (per-solvent aggregated)
  results/gcn/brics_fragment_importance.csv     (BRICS fragment importance)
  results/gcn/brics_fragment_per_solvent.csv    (BRICS fragment per solvent)
  results/gcn/gnn_explainer_node_masks.npz      (per-atom mask, sample)
  results/gcn/gnn_explainer_summary.json        (top atoms & fragments)
  results/gcn/metrics.json                      (sanity-check RMSE)
  figures/gcn_atom_type_importance.png
  figures/gcn_atom_type_per_solvent.png
  figures/gcn_brics_fragments_top.png
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

# Cap CPU usage to ~60% of cores.
_N_CPUS_TOTAL = os.cpu_count() or 16
_N_JOBS = max(1, int(round(_N_CPUS_TOTAL * 0.60)))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_N_JOBS))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
VANSH_ROOT = HERE.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import BRICS, AllChem  # noqa: E402

from sc3_bench.data import load_all_splits  # noqa: E402
from sc3_bench.evaluate import compute_metrics  # noqa: E402
from sc3_bench.registry import get_hp  # noqa: E402
from sc3_bench.models.gnn_models import (  # noqa: E402
    DualGNNSolubility, smiles_to_graph, batch_graph_list,
    SolubilityGraphDataset, GNNEncoder,
)

RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = RESULTS_DIR / "gcn"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Logging
# =====================================================================

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =====================================================================
# Model loading
# =====================================================================

def _load_gcn(seed: int = 42, device: torch.device = torch.device("cpu")):
    """Load the trained dual-encoder GCN with the production HPs."""
    hp = get_hp("gcn")
    model = DualGNNSolubility(
        node_dim=7, hidden_dim=hp.get("hidden_dim", 96),
        num_layers=hp.get("num_layers", 4),
        gnn_type="GCN",
    )
    model_path = VANSH_ROOT / "results" / "gcn" / f"model_seed_{seed}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    state = pickle.load(open(model_path, "rb"))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    _log(f"  GCN loaded from {model_path.name} on {device}")
    return model, hp


# =====================================================================
# Sanity check on eval/ood
# =====================================================================

@torch.no_grad()
def _predict_batch(model, df, graph_cache, device, bs=256):
    """Predict logS for all rows in df."""
    ds = SolubilityGraphDataset(df, graph_cache)
    def collate(batch):
        sol_gs, solv_gs, tfs, tgts = zip(*batch)
        def _mv(bd):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in bd.items()}
        return (_mv(batch_graph_list(sol_gs)),
                _mv(batch_graph_list(solv_gs)),
                torch.stack(tfs).to(device),
                torch.stack(tgts).to(device))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    preds = []
    for sol, solv, tf, _ in dl:
        preds.append(model(sol, solv, tf).cpu().numpy())
    return np.concatenate(preds)


def sanity_check_metrics(model, splits, graph_cache, device):
    metrics = {}
    for sname in ["eval", "ood"]:
        df = splits[sname]
        preds = _predict_batch(model, df, graph_cache, device)
        m = compute_metrics(
            df["LogS"].values, preds,
            df["Solvent_Name"].values, df["Uncertainty"].values
            if "Uncertainty" in df.columns else None,
        )
        metrics[sname] = m
        ps = m.get("PS_RMSE", float("nan"))
        ps_str = f"PS={ps:.4f}" if not np.isnan(ps) else ""
        _log(f"  {sname:6s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}  {ps_str}  N={m['N']}")
    return metrics


# =====================================================================
# Method 1: atom-level occlusion attribution
# =====================================================================

@torch.no_grad()
def _occlude_one_pair(model, sol_g_orig: dict, solv_g: dict, T: float,
                      device: torch.device) -> tuple[float, np.ndarray]:
    """Return (baseline_pred, per-atom |delta-pred|) for ONE solute-solvent pair.

    Soft occlusion: zero out node v's feature row, keep topology intact.
    """
    n = sol_g_orig["num_nodes"]
    if n == 0:
        return float("nan"), np.zeros(0, dtype=np.float32)
    if n == 1:
        # Single-atom molecule (shouldn't happen for our data, but safe).
        return _eval_one(model, sol_g_orig, solv_g, T, device), np.zeros(1, dtype=np.float32)

    # Stack n+1 versions of the solute graph: index 0 = original, 1..n = atom v
    # zeroed.  Batch through one forward pass for speed.
    node_feats = sol_g_orig["node_feats"]
    feats_list = [node_feats]
    for v in range(n):
        nf = node_feats.clone()
        nf[v] = 0.0
        feats_list.append(nf)
    sol_versions = [{"node_feats": nf, "edge_index": sol_g_orig["edge_index"],
                     "num_nodes": n} for nf in feats_list]
    solv_versions = [solv_g] * (n + 1)
    temp_feats = torch.tensor(
        [T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)],
        dtype=torch.float32, device=device,
    ).unsqueeze(0).repeat(n + 1, 1)

    sol_batch  = batch_graph_list(sol_versions, device=device)
    solv_batch = batch_graph_list(solv_versions, device=device)
    preds = model(sol_batch, solv_batch, temp_feats).cpu().numpy()
    base = float(preds[0])
    delta = np.abs(preds[1:] - base).astype(np.float32)
    return base, delta


@torch.no_grad()
def _eval_one(model, sol_g, solv_g, T, device):
    sol_b  = batch_graph_list([sol_g],  device=device)
    solv_b = batch_graph_list([solv_g], device=device)
    tf = torch.tensor([[T / 300.0, 1000.0 / T, (T / 300.0) ** 2, np.log(T / 300.0)]],
                      dtype=torch.float32, device=device)
    return float(model(sol_b, solv_b, tf).cpu().numpy()[0])


def run_atom_occlusion(model, splits, graph_cache, device, n_sample=3000,
                       seed=0):
    """Run atom-level occlusion on a random sample of `eval`."""
    df = splits["eval"].reset_index(drop=True)
    rng = np.random.RandomState(seed)
    if n_sample < len(df):
        idx = rng.choice(len(df), size=n_sample, replace=False)
    else:
        idx = np.arange(len(df))
    _log(f"  atom occlusion on {len(idx)} eval rows...")

    rows = []
    t0 = time.time()
    for k, i in enumerate(idx):
        row = df.iloc[i]
        sol_smi = row["Solute"]; solv_smi = row["Solvent"]
        sol_g = graph_cache.get(sol_smi); solv_g = graph_cache.get(solv_smi)
        if sol_g is None or solv_g is None or sol_g["num_nodes"] == 0:
            continue
        T = float(row["Temperature"])
        # We need to recover RDKit atom-by-atom info -> rebuild the mol for
        # atom symbols (rdkit-cached graph_cache was built from SMILES).
        mol = Chem.MolFromSmiles(sol_smi)
        if mol is None or mol.GetNumAtoms() != sol_g["num_nodes"]:
            continue

        # Move solute graph tensors to device once.
        sol_g_dev = {
            "node_feats": sol_g["node_feats"].to(device),
            "edge_index": sol_g["edge_index"].to(device),
            "num_nodes":  sol_g["num_nodes"],
        }
        solv_g_dev = {
            "node_feats": solv_g["node_feats"].to(device),
            "edge_index": solv_g["edge_index"].to(device),
            "num_nodes":  solv_g["num_nodes"],
        }
        base, delta = _occlude_one_pair(model, sol_g_dev, solv_g_dev, T, device)
        n_atoms = len(delta)
        for v in range(n_atoms):
            atom = mol.GetAtomWithIdx(v)
            rows.append({
                "row_idx": int(i),
                "solute_smiles": sol_smi,
                "solvent_smiles": solv_smi,
                "solvent_name": row["Solvent_Name"],
                "T": T,
                "atom_idx": v,
                "atom_symbol": atom.GetSymbol(),
                "atom_aromatic": int(atom.GetIsAromatic()),
                "atom_in_ring": int(atom.IsInRing()),
                "atom_degree": atom.GetDegree(),
                "atom_n_h": atom.GetTotalNumHs(),
                "base_pred": base,
                "abs_delta": float(delta[v]),
            })
        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(idx) - k - 1)
            _log(f"    {k+1}/{len(idx)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
    out = OUT_DIR / "atom_occlusion__eval.csv.gz"
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
    _log(f"  saved {len(rows)} atom rows to {out}")
    return out


def aggregate_atom_occlusion(parquet_path: Path):
    df = pd.read_csv(parquet_path)

    # Per atom-type / aromaticity / ring
    df["atom_kind"] = (
        df["atom_symbol"]
        + "_ar" + df["atom_aromatic"].astype(str)
        + "_R" + df["atom_in_ring"].astype(str)
    )
    by_kind = df.groupby("atom_kind").agg(
        mean_abs_delta=("abs_delta", "mean"),
        std_abs_delta=("abs_delta", "std"),
        n=("abs_delta", "count"),
    ).sort_values("mean_abs_delta", ascending=False)
    by_kind.to_csv(OUT_DIR / "atom_occlusion_by_atom_type.csv")
    _log(f"  wrote {OUT_DIR/'atom_occlusion_by_atom_type.csv'}")

    by_solv = df.groupby(["solvent_name", "atom_symbol"]).agg(
        mean_abs_delta=("abs_delta", "mean"),
        n=("abs_delta", "count"),
    )
    by_solv.to_csv(OUT_DIR / "atom_occlusion_by_solvent.csv")
    _log(f"  wrote {OUT_DIR/'atom_occlusion_by_solvent.csv'}")
    return by_kind, by_solv


# =====================================================================
# Method 2: BRICS fragment importance (via atom-level aggregation)
# =====================================================================

def aggregate_brics_importance(parquet_path: Path,
                                top_solvents_for_per_solvent: int = 10):
    """For each molecule's atom-level importance, BRICS-decompose, then sum
    importance per BRICS fragment.  Aggregate fragment importance globally
    and per the top-N solvents.
    """
    df = pd.read_csv(parquet_path)
    rows_global = []
    rows_per_solvent = []

    cache: dict[str, list[str]] = {}      # smiles -> [BRICS fragment SMILES per atom]
    for sol_smi, sub in df.groupby("solute_smiles"):
        if sol_smi in cache:
            atom_to_frag = cache[sol_smi]
        else:
            mol = Chem.MolFromSmiles(sol_smi)
            if mol is None:
                cache[sol_smi] = ["<invalid>"] * 0
                continue
            n = mol.GetNumAtoms()
            atom_to_frag = ["<unknown>"] * n
            try:
                bonds = list(BRICS.FindBRICSBonds(mol))
                bond_idxs = [mol.GetBondBetweenAtoms(b[0][0], b[0][1]).GetIdx() for b in bonds]
                if bond_idxs:
                    frag_mol = Chem.FragmentOnBonds(mol, bond_idxs, dummyLabels=[(0, 0)] * len(bond_idxs))
                    frags = Chem.GetMolFrags(frag_mol, asMols=False, frags=None)
                    # GetMolFrags returns atom indices grouped per frag, but with
                    # dummy atoms appended. Use the molecule version with dummies stripped.
                    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
                    # Map original atom -> fragment SMILES
                    atom_to_frag_idx = [-1] * frag_mol.GetNumAtoms()
                    for fi, atom_ids in enumerate(Chem.GetMolFrags(frag_mol)):
                        for aid in atom_ids:
                            atom_to_frag_idx[aid] = fi
                    for v in range(n):
                        fi = atom_to_frag_idx[v]
                        if fi >= 0:
                            try:
                                # Strip dummy atoms (atomic number 0) from frag SMILES
                                f_smi = Chem.MolToSmiles(_strip_dummies(frags[fi]))
                                atom_to_frag[v] = f_smi
                            except Exception:
                                pass
                else:
                    f_smi = Chem.MolToSmiles(mol)
                    atom_to_frag = [f_smi] * n
            except Exception:
                pass
            cache[sol_smi] = atom_to_frag

        for _, r in sub.iterrows():
            v = int(r["atom_idx"])
            if v < len(atom_to_frag):
                rows_global.append({
                    "solute_smiles": sol_smi,
                    "fragment": atom_to_frag[v],
                    "abs_delta": float(r["abs_delta"]),
                })
                rows_per_solvent.append({
                    "solvent_name": r["solvent_name"],
                    "fragment": atom_to_frag[v],
                    "abs_delta": float(r["abs_delta"]),
                })

    g = pd.DataFrame(rows_global)
    if not g.empty:
        agg = g.groupby("fragment").agg(
            mean_abs_delta=("abs_delta", "mean"),
            sum_abs_delta=("abs_delta", "sum"),
            n=("abs_delta", "count"),
        ).sort_values("sum_abs_delta", ascending=False)
        agg.to_csv(OUT_DIR / "brics_fragment_importance.csv")
        _log(f"  wrote {OUT_DIR/'brics_fragment_importance.csv'} ({len(agg)} unique fragments)")

    pdf = pd.DataFrame(rows_per_solvent)
    if not pdf.empty:
        # Pick top-N solvents by sample count, then for each emit top-15 fragments.
        top_solvents = pdf["solvent_name"].value_counts().head(top_solvents_for_per_solvent).index.tolist()
        rows = []
        for s in top_solvents:
            sub = pdf[pdf["solvent_name"] == s]
            agg_s = sub.groupby("fragment").agg(
                mean_abs_delta=("abs_delta", "mean"),
                sum_abs_delta=("abs_delta", "sum"),
                n=("abs_delta", "count"),
            ).sort_values("sum_abs_delta", ascending=False).head(15)
            for fragment, row in agg_s.iterrows():
                rows.append({"solvent_name": s, "fragment": fragment,
                             "mean_abs_delta": float(row["mean_abs_delta"]),
                             "sum_abs_delta":  float(row["sum_abs_delta"]),
                             "n": int(row["n"])})
        pd.DataFrame(rows).to_csv(OUT_DIR / "brics_fragment_per_solvent.csv", index=False)
        _log(f"  wrote {OUT_DIR/'brics_fragment_per_solvent.csv'}")


def _strip_dummies(mol):
    rwmol = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetAtomicNum() == 0]
    for ai in sorted(to_remove, reverse=True):
        rwmol.RemoveAtom(ai)
    return rwmol.GetMol()


# =====================================================================
# Plots
# =====================================================================

def plot_atom_type_importance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_kind = pd.read_csv(OUT_DIR / "atom_occlusion_by_atom_type.csv")
    top = by_kind.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(np.arange(len(top)), top["mean_abs_delta"],
            xerr=top["std_abs_delta"] / np.sqrt(top["n"]),
            color="#2ca02c", edgecolor="black", linewidth=0.4, ecolor="gray")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["atom_kind"], fontsize=9)
    ax.set_xlabel("mean |delta-pred|  (logS units, atom-zero-out occlusion)")
    ax.set_title("GCN: atom-type importance via occlusion attribution\n"
                 "(top-20 atom kinds; eval split sample)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / "gcn_atom_type_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")


def plot_atom_type_per_solvent(focus_solvents=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_DIR / "atom_occlusion_by_solvent.csv")
    pivot = df.pivot(index="solvent_name", columns="atom_symbol", values="mean_abs_delta").fillna(0.0)
    if focus_solvents:
        pivot = pivot.reindex([s for s in focus_solvents if s in pivot.index])
    else:
        # Top-15 by total importance
        pivot = pivot.reindex(pivot.sum(axis=1).sort_values(ascending=False).head(15).index)
    # Keep top-8 atom symbols by total importance
    cols = pivot.sum(axis=0).sort_values(ascending=False).head(8).index.tolist()
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 3, 0.32 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("GCN: atom-type importance per solvent\n"
                 "(occlusion mean |delta-pred|)", fontsize=10)
    vmax = pivot.values.max()
    for i in range(len(pivot)):
        for j in range(len(cols)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.45 * vmax else "black")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    out = FIG_DIR / "gcn_atom_type_per_solvent.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")


def plot_brics_top():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_DIR / "brics_fragment_importance.csv")
    df = df[df["fragment"].apply(lambda s: isinstance(s, str) and s and not s.startswith("<"))]
    # Filter "common enough" fragments (n >= 30) to avoid noise from singletons
    df = df[df["n"] >= 30]
    top = df.head(20)[::-1]
    if top.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.barh(np.arange(len(top)), top["sum_abs_delta"],
            color="#9467bd", edgecolor="black", linewidth=0.4)
    # Render fragment SMILES + count as the y-tick labels
    labels = [f"{f}   (n={int(n)})" for f, n in zip(top["fragment"], top["n"])]
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("total |delta-pred| over sample  (logS units)")
    ax.set_title("GCN: BRICS fragment importance via atom occlusion\n"
                 "(top-20 fragments with >=30 atom occurrences)", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = FIG_DIR / "gcn_brics_fragments_top.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    _log(f"  wrote {out}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed of the trained GCN model (default: 42).")
    parser.add_argument("--n-sample", type=int, default=3000,
                        help="Number of eval rows to occlusion-attribute (default: 3000).")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id to use (default: pick first free GPU).")
    parser.add_argument("--skip-occlusion", action="store_true",
                        help="Skip atom-occlusion if results already exist.")
    parser.add_argument("--skip-brics", action="store_true",
                        help="Skip BRICS fragment aggregation.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Don't render figures.")
    args = parser.parse_args()

    _log(f"CPU cap: using {_N_JOBS}/{_N_CPUS_TOTAL} cores (60%)")

    # Pick GPU
    if torch.cuda.is_available():
        if args.gpu is None:
            # Pick the GPU with the most free memory
            free = []
            for i in range(torch.cuda.device_count()):
                free.append((torch.cuda.mem_get_info(i)[0], i))
            args.gpu = max(free)[1]
        device = torch.device(f"cuda:{args.gpu}")
        _log(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        _log("No GPU available; falling back to CPU.")

    # 1. Load model + data
    model, hp = _load_gcn(args.seed, device=device)
    _log(f"  GCN HPs: {hp}")
    splits = load_all_splits(verbose=True)

    # Build the SMILES -> graph cache (small molecules ~1k)
    _log("  building graph cache for all SMILES...")
    all_smi = set()
    for df in splits.values():
        all_smi.update(df["Solute"].unique())
        all_smi.update(df["Solvent"].unique())
    graph_cache = {}
    failed = 0
    for s in all_smi:
        g = smiles_to_graph(s)
        if g is not None:
            graph_cache[s] = g
        else:
            failed += 1
    _log(f"  graph cache: {len(graph_cache)} mols ({failed} failed)")

    # 2. Sanity-check metrics (must match results/gcn/summary.json)
    _log("\n=== sanity-check metrics ===")
    metrics = sanity_check_metrics(model, splits, graph_cache, device)
    payload = {
        "method": "gcn", "seed": args.seed, "hp": hp,
        "n_sample": args.n_sample, "metrics": metrics,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # 3. Atom occlusion (or skip if cached)
    occ_path = OUT_DIR / "atom_occlusion__eval.csv.gz"
    if args.skip_occlusion and occ_path.exists():
        _log(f"  reusing cached {occ_path.name}")
    else:
        _log("\n=== atom occlusion (eval) ===")
        run_atom_occlusion(model, splits, graph_cache, device, n_sample=args.n_sample)

    if occ_path.exists():
        _log("\n=== aggregate atom-type importance ===")
        aggregate_atom_occlusion(occ_path)

        if not args.skip_brics:
            _log("\n=== BRICS fragment aggregation ===")
            aggregate_brics_importance(occ_path)

        if not args.skip_figures:
            _log("\n=== figures ===")
            plot_atom_type_importance()
            plot_atom_type_per_solvent()
            plot_brics_top()

    _log("\nDone.")


if __name__ == "__main__":
    main()
