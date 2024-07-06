"""
Per-method trainers for the data-scaling ablation.

Each trainer takes a *subset* of the training rows (selected by
`subsample_indices`) and returns metrics on eval / ood / sc3_gold.

Three methods are supported:

  - lgb_rdkit : LightGBM on RDKit-2D features (from feature_cache/rdkit.npz)
  - fastprop  : Deep MLP on RDKit-2D features (PyTorch)
  - molmerger : AttentiveFP on Gasteiger-merged solute-solvent graphs

The shared featurization caches are built once per process (graph cache
for molmerger, RDKit feature cache from disk for the descriptor models).

All three trainers print live tqdm progress bars and per-epoch best-RMSE
lines so you can monitor a long sweep from `tail -f`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Make the existing sc3_bench package importable.
VANSH_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.data import load_all_splits, load_cached_features  # noqa: E402
from sc3_bench.evaluate import compute_metrics  # noqa: E402

EVAL_SPLITS = ["eval", "ood", "sc3_gold"]

# Cap CPU usage to 60% of available cores so we don't starve other jobs
# (e.g. the GNN training already running in tmux on this host).
N_CPUS_TOTAL = os.cpu_count() or 16
N_JOBS = max(1, int(round(N_CPUS_TOTAL * 0.60)))

# On-disk cache for graph featurization (so we never rebuild skeletons
# across separate runs of the script).  Lives next to the existing
# feature_cache/*.npz files used by the descriptor models.
GRAPH_CACHE_DIR = VANSH_ROOT / "feature_cache"
GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MOLMERGER_CACHE_FILE = GRAPH_CACHE_DIR / "molmerger_skeletons.pt"
GCN_CACHE_FILE = GRAPH_CACHE_DIR / "gcn_graphs.pt"


# ---------------------------------------------------------------------------
# Reproducible subsampling
# ---------------------------------------------------------------------------

def subsample_indices(n: int, fraction: float, seed: int) -> np.ndarray:
    """Return `int(round(n*fraction))` row indices drawn without replacement.

    Reproducible across calls for the same (n, fraction, seed).
    Always returns at least 32 indices to keep small models trainable.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.RandomState(seed)
    k = max(32, int(round(n * fraction)))
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


# ---------------------------------------------------------------------------
# Eval-metadata helpers (y_true / solvent_names / uncertainty per split)
# ---------------------------------------------------------------------------

def _split_meta(splits: dict) -> dict:
    meta = {}
    for sname in EVAL_SPLITS:
        df = splits[sname]
        meta[sname] = {
            "y_true": df["LogS"].values,
            "solvent_names": df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
            "uncertainties": df["Uncertainty"].values if "Uncertainty" in df.columns else None,
        }
    return meta


def _log(msg: str) -> None:
    """Single-line stamped log to stdout that flushes immediately."""
    ts = time.strftime("%H:%M:%S")
    print(f"      [{ts}] {msg}", flush=True)


# =====================================================================
# 1) LightGBM on RDKit features
# =====================================================================

def train_lgb_rdkit(
    fraction: float,
    seed: int,
    splits: dict,
    params: dict,
    cached: dict,
) -> dict:
    """Train LightGBM on a fraction of training rows; return per-split metrics."""
    from lightgbm import LGBMRegressor
    import lightgbm as lgb

    X_full, y_full = cached["X_train"], cached["y_train"]
    X_eval, y_eval = cached["X_eval"], cached["y_eval"]

    idx = subsample_indices(len(y_full), fraction, seed)
    X_tr, y_tr = X_full[idx], y_full[idx]
    _log(f"lgb: train_rows={len(idx):,}  (fraction={fraction:.3f}, seed={seed})  n_jobs={N_JOBS}")

    model = LGBMRegressor(random_state=seed, n_jobs=N_JOBS, verbose=-1, **params)
    t0 = time.time()

    n_estimators = int(params.get("n_estimators", 3000))
    pbar = tqdm(total=n_estimators, desc=f"lgb f={fraction:.2f} s={seed}",
                ncols=100, leave=False)

    def _tqdm_cb(env):
        pbar.update(1)
        if env.evaluation_result_list:
            name, _, val, _ = env.evaluation_result_list[0]
            pbar.set_postfix(val=f"{val:.4f}")

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_eval, y_eval)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            _tqdm_cb,
        ],
    )
    pbar.close()
    best_iter = getattr(model, "best_iteration_", None)
    _log(f"lgb: trained in {time.time()-t0:.1f}s  best_iter={best_iter}")

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        preds = model.predict(cached[f"X_{sname}"])
        metrics[sname] = compute_metrics(
            meta[sname]["y_true"], preds,
            meta[sname]["solvent_names"], meta[sname]["uncertainties"],
        )
        _log(f"lgb: eval[{sname:9s}]  RMSE={metrics[sname]['RMSE']:.4f}  N={metrics[sname]['N']}")
    metrics["_n_train"] = int(len(idx))
    metrics["_best_iter"] = int(best_iter) if best_iter else None
    return metrics


# =====================================================================
# 2) FastProp (deep MLP on RDKit features)
# =====================================================================

def train_fastprop(
    fraction: float,
    seed: int,
    splits: dict,
    params: dict,
    cached: dict,
    device,
) -> dict:
    """Train FastProp deep MLP on a subsample of train rows."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sc3_bench.models.descriptor_models import FastPropNet

    X_full, y_full = cached["X_train"], cached["y_train"]
    X_eval, y_eval = cached["X_eval"], cached["y_eval"]

    idx = subsample_indices(len(y_full), fraction, seed)
    X_tr, y_tr = X_full[idx], y_full[idx]
    _log(f"fastprop: train_rows={len(idx):,}  (fraction={fraction:.3f}, seed={seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)

    f_mu = X_tr.mean(0)
    f_sd = X_tr.std(0) + 1e-8
    X_tr_n = ((X_tr - f_mu) / f_sd).astype(np.float32)
    X_ev_n = ((X_eval - f_mu) / f_sd).astype(np.float32)

    in_dim = X_tr_n.shape[1]
    hidden = tuple(params.get("hidden_dims", (512, 256, 128)))

    model = FastPropNet(in_dim=in_dim, hidden_dims=hidden,
                        dropout=params.get("dropout", 0.1)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"fastprop: model has {n_params:,} parameters  hidden={hidden}")

    weight_decay = params.get("weight_decay", 1e-5)
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3),
                           weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=params.get("lr_patience", 10), factor=0.5)

    X_t = torch.tensor(X_tr_n, device=device)
    y_t = torch.tensor(y_tr, device=device)
    X_v = torch.tensor(X_ev_n, device=device)

    n = len(y_tr)
    bs = min(params.get("batch_size", 256), max(32, n // 4))
    epochs = params.get("epochs", 100)
    patience = params.get("patience", 20)
    _log(f"fastprop: epochs={epochs}  batch_size={bs}  lr={params.get('lr', 1e-3)}  "
         f"patience={patience}  weight_decay={weight_decay}  dropout={params.get('dropout', 0.1)}")

    best_vl, best_st, best_ep, wait = float("inf"), None, -1, 0
    train_rmse_at_best = float("nan")
    t0 = time.time()
    pbar = tqdm(range(epochs), desc=f"fastprop f={fraction:.2f} s={seed}",
                ncols=100, leave=False)
    for ep in pbar:
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss_sum, n_batches = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            ib = perm[i:i + bs]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(X_t[ib]), y_t[ib])
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()); n_batches += 1
        train_rmse = float(np.sqrt(train_loss_sum / max(n_batches, 1)))

        model.eval()
        with torch.no_grad():
            vp = model(X_v).cpu().numpy()
        vl = float(np.mean((vp - y_eval) ** 2))
        vrmse = float(np.sqrt(vl))
        scheduler.step(vl)

        improved = vl < best_vl
        if improved:
            best_vl, best_ep, wait = vl, ep, 0
            train_rmse_at_best = train_rmse
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        pbar.set_postfix(train=f"{train_rmse:.4f}", val=f"{vrmse:.4f}",
                         best=f"{np.sqrt(best_vl):.4f}", wait=wait)
        if (ep + 1) % 10 == 0 or improved:
            _log(f"fastprop ep {ep+1:3d}  train_RMSE={train_rmse:.4f}  "
                 f"val_RMSE={vrmse:.4f}  best={np.sqrt(best_vl):.4f}  "
                 f"({'*best' if improved else f'wait={wait}'})")
        if wait >= patience:
            _log(f"fastprop early stop at epoch {ep+1}  (best_ep={best_ep+1})")
            break
    pbar.close()
    _log(f"fastprop: trained in {time.time()-t0:.1f}s  best_val_RMSE={np.sqrt(best_vl):.4f}")

    if best_st:
        model.load_state_dict(best_st)
    model.eval()

    def predict(X_raw):
        Xn = ((X_raw - f_mu) / f_sd).astype(np.float32)
        with torch.no_grad():
            return model(torch.tensor(Xn, device=device)).cpu().numpy()

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        preds = predict(cached[f"X_{sname}"])
        metrics[sname] = compute_metrics(
            meta[sname]["y_true"], preds,
            meta[sname]["solvent_names"], meta[sname]["uncertainties"],
        )
        _log(f"fastprop: eval[{sname:9s}]  RMSE={metrics[sname]['RMSE']:.4f}  N={metrics[sname]['N']}")
    metrics["_n_train"] = int(len(idx))
    metrics["_best_epoch"] = int(best_ep + 1)
    metrics["_train_RMSE_at_best"] = float(train_rmse_at_best)
    metrics["_val_RMSE_best"] = float(np.sqrt(best_vl))
    metrics["_n_params"] = int(n_params)
    return metrics


# =====================================================================
# 3) MolMerger (AttentiveFP on Gasteiger-merged graphs)
# =====================================================================

# Process-level skeleton cache: built once for all (solute, solvent) pairs.
_molmerger_cache: Optional[dict] = None
_gcn_graph_cache: Optional[dict] = None


def build_gcn_graph_cache(splits: dict, verbose: bool = True,
                          cache_path: Path = GCN_CACHE_FILE) -> dict:
    """Build SMILES -> graph dict for every unique solute/solvent in splits.

    Used by GCN / GAT / GIN trainers (each row picks two graphs from the
    cache: one solute, one solvent).  The dict is pickled to
    ``feature_cache/gcn_graphs.pt`` so subsequent runs only build new
    graphs (none, in practice, since the splits are fixed).

    Returns a dict ``{smiles: {"node_feats": Tensor, "edge_index": Tensor,
    "num_nodes": int}}``.
    """
    global _gcn_graph_cache
    if _gcn_graph_cache is not None:
        return _gcn_graph_cache

    import torch
    from sc3_bench.models.gnn_models import smiles_to_graph

    smiles_set: set[str] = set()
    for df in splits.values():
        smiles_set.update(df["Solute"].unique())
        if "Solvent" in df.columns:
            smiles_set.update(df["Solvent"].unique())

    cache: dict = {}
    if cache_path.exists():
        if verbose:
            _log(f"gcn: loading graph cache from {cache_path.name}...")
        t0 = time.time()
        cache = torch.load(cache_path, weights_only=False)
        if verbose:
            _log(f"gcn: loaded {len(cache):,} graphs in {time.time()-t0:.1f}s")

    missing = [s for s in smiles_set if s not in cache]
    if missing:
        if verbose:
            _log(f"gcn: building graphs for {len(missing):,} new SMILES "
                 f"({len(smiles_set):,} total, {len(cache):,} already cached)...")
        t0 = time.time()
        bad = 0
        pbar = tqdm(sorted(missing), desc="gcn graphs", ncols=100, leave=True)
        for s in pbar:
            g = smiles_to_graph(s)
            if g is None:
                bad += 1
                cache[s] = {
                    "node_feats": torch.zeros((1, 7)),
                    "edge_index": torch.zeros((2, 0), dtype=torch.long),
                    "num_nodes": 1,
                }
            else:
                cache[s] = g
        pbar.close()
        if verbose:
            _log(f"gcn: built {len(missing):,} new graphs ({bad} unparseable) "
                 f"in {time.time()-t0:.1f}s")
            _log(f"gcn: saving cache to {cache_path}")
        torch.save(cache, cache_path)
        if verbose:
            _log(f"gcn: saved cache ({cache_path.stat().st_size/1e6:.1f} MB)")
    else:
        if verbose:
            _log(f"gcn: cache already covers all {len(smiles_set):,} SMILES (no rebuild needed)")

    _gcn_graph_cache = cache
    return cache


def build_molmerger_cache(splits: dict, verbose: bool = True,
                          cache_path: Path = MOLMERGER_CACHE_FILE) -> dict:
    """Build (solute, solvent) -> skeleton Data once for all pairs in splits.

    Skeletons are temperature-free; the runner stamps T at batch time.

    On-disk cache: the dict is pickled to ``feature_cache/molmerger_skeletons.pt``
    so subsequent runs only rebuild skeletons for *new* pairs (none, in
    practice, since the splits are fixed).
    """
    global _molmerger_cache
    if _molmerger_cache is not None:
        return _molmerger_cache

    import torch
    from sc3_bench.models.molmerger import molmerger_skeleton

    pairs: set[tuple[str, str]] = set()
    for df in splits.values():
        for sol, solv in zip(df["Solute"].values, df["Solvent"].values):
            pairs.add((sol, solv))

    cache: dict = {}
    if cache_path.exists():
        if verbose:
            _log(f"molmerger: loading skeleton cache from {cache_path.name}...")
        t0 = time.time()
        cache = torch.load(cache_path, weights_only=False)
        if verbose:
            _log(f"molmerger: loaded {len(cache):,} skeletons in {time.time()-t0:.1f}s")

    missing = [p for p in pairs if p not in cache]
    if missing:
        if verbose:
            _log(f"molmerger: building skeletons for {len(missing):,} new pairs "
                 f"({len(pairs):,} total, {len(cache):,} already cached)...")
        t0 = time.time()
        bad = 0
        pbar = tqdm(sorted(missing), desc="molmerger skeletons", ncols=100, leave=True)
        for (sol, solv) in pbar:
            g = molmerger_skeleton(sol, solv)
            if g is None:
                bad += 1
            cache[(sol, solv)] = g
        pbar.close()
        if verbose:
            _log(f"molmerger: built {len(missing):,} new skeletons "
                 f"({bad} unparseable) in {time.time() - t0:.1f}s")
        if verbose:
            _log(f"molmerger: saving cache to {cache_path}")
        t0 = time.time()
        torch.save(cache, cache_path)
        if verbose:
            _log(f"molmerger: saved cache ({cache_path.stat().st_size/1e6:.1f} MB) "
                 f"in {time.time()-t0:.1f}s")
    else:
        if verbose:
            _log(f"molmerger: cache already covers all {len(pairs):,} pairs (no rebuild needed)")

    _molmerger_cache = cache
    return cache


def _make_molmerger_dataset(df, cache):
    """Yield (Data_with_T, target) for every row in df."""
    import torch
    from sc3_bench.models.molmerger import stamp_temperature

    samples = []
    skipped = 0
    for sol, solv, T, y in zip(df["Solute"].values, df["Solvent"].values,
                                df["Temperature"].values, df["LogS"].values):
        skel = cache.get((sol, solv))
        if skel is None:
            skipped += 1
            continue
        data = stamp_temperature(skel, float(T))
        data.y = torch.tensor([float(y)], dtype=torch.float32)
        samples.append(data)
    return samples, skipped


def train_molmerger(
    fraction: float,
    seed: int,
    splits: dict,
    params: dict,
    cache: dict,
    device,
) -> dict:
    """Train MolMerger / AttentiveFP on a subsample of train rows."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sc3_bench.models.molmerger import MolMergerNet

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_df = splits["train"]
    n = len(train_df)
    idx = subsample_indices(n, fraction, seed)
    train_sub = train_df.iloc[idx].reset_index(drop=True)
    _log(f"molmerger: train_rows={len(idx):,}  (fraction={fraction:.3f}, seed={seed})")

    t0 = time.time()
    train_data, n_skipped_tr = _make_molmerger_dataset(train_sub, cache)
    eval_data, _ = _make_molmerger_dataset(splits["eval"], cache)
    _log(f"molmerger: built {len(train_data):,} train graphs "
         f"({n_skipped_tr} skipped) and {len(eval_data):,} eval graphs in {time.time()-t0:.1f}s")

    bs = min(params.get("batch_size", 256), max(32, len(train_data) // 4))
    train_loader = PyGDataLoader(train_data, batch_size=bs, shuffle=True)
    eval_loader = PyGDataLoader(eval_data, batch_size=bs, shuffle=False)

    model = MolMergerNet(
        hidden_dim=params.get("hidden_dim", 200),
        num_layers=params.get("num_layers", 3),
        num_timesteps=params.get("num_timesteps", 2),
        dropout=params.get("dropout", 0.2),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"molmerger: model has {n_params:,} parameters  "
         f"hidden_dim={params.get('hidden_dim', 200)}  layers={params.get('num_layers', 3)}")

    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3),
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5)

    epochs = params.get("epochs", 200)
    patience = params.get("patience", 25)
    _log(f"molmerger: epochs={epochs}  batch_size={bs}  lr={params.get('lr', 1e-3)}  patience={patience}")

    n_train_batches = max(1, (len(train_data) + bs - 1) // bs)

    best_vl, best_st, best_ep, wait = float("inf"), None, -1, 0
    t0 = time.time()
    epoch_pbar = tqdm(range(epochs), desc=f"molmerger f={fraction:.2f} s={seed}",
                      ncols=100, leave=False)
    for ep in epoch_pbar:
        model.train()
        train_loss_sum, n_batches = 0.0, 0
        batch_pbar = tqdm(train_loader, total=n_train_batches,
                          desc=f"  ep {ep+1:3d}/{epochs}", ncols=100, leave=False)
        for batch in batch_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = nn.functional.mse_loss(pred, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.item()); n_batches += 1
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")
        batch_pbar.close()
        train_rmse = float(np.sqrt(train_loss_sum / max(n_batches, 1)))

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                vp.append(model(batch).cpu().numpy())
                vt.append(batch.y.cpu().numpy())
        vp = np.concatenate(vp)
        vt = np.concatenate(vt)
        vl = float(np.mean((vp - vt) ** 2))
        vrmse = float(np.sqrt(vl))
        scheduler.step(vl)

        improved = vl < best_vl
        if improved:
            best_vl, best_ep, wait = vl, ep, 0
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        epoch_pbar.set_postfix(train=f"{train_rmse:.4f}", val=f"{vrmse:.4f}",
                               best=f"{np.sqrt(best_vl):.4f}", wait=wait)
        if (ep + 1) % 5 == 0 or improved:
            _log(f"molmerger ep {ep+1:3d}  train_RMSE={train_rmse:.4f}  "
                 f"val_RMSE={vrmse:.4f}  best={np.sqrt(best_vl):.4f}  "
                 f"({'*best' if improved else f'wait={wait}'})")
        if wait >= patience:
            _log(f"molmerger early stop at epoch {ep+1}  (best_ep={best_ep+1})")
            break
    epoch_pbar.close()
    _log(f"molmerger: trained in {time.time()-t0:.1f}s  best_val_RMSE={np.sqrt(best_vl):.4f}")

    if best_st:
        model.load_state_dict(best_st)
    model.eval()

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        data, n_skipped = _make_molmerger_dataset(splits[sname], cache)
        loader = PyGDataLoader(data, batch_size=bs, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                preds.append(model(batch).cpu().numpy())
        preds = np.concatenate(preds)

        # Align truth with preds (skip rows whose skeleton failed to parse).
        df = splits[sname]
        keep_mask = np.array(
            [(s, sv) in cache and cache[(s, sv)] is not None
             for s, sv in zip(df["Solute"].values, df["Solvent"].values)]
        )
        y_true = meta[sname]["y_true"][keep_mask]
        sn_arr = meta[sname]["solvent_names"]
        sn_arr = sn_arr[keep_mask] if sn_arr is not None else None
        un = meta[sname]["uncertainties"]
        un = un[keep_mask] if un is not None else None
        metrics[sname] = compute_metrics(y_true, preds, sn_arr, un)
        metrics[sname]["n_skipped_unparseable"] = int(n_skipped)
        _log(f"molmerger: eval[{sname:9s}]  RMSE={metrics[sname]['RMSE']:.4f}  "
             f"N={metrics[sname]['N']}  skipped={n_skipped}")

    metrics["_n_train"] = int(len(train_data))
    metrics["_n_train_skipped_unparseable"] = int(n_skipped_tr)
    metrics["_best_epoch"] = int(best_ep + 1)
    return metrics
