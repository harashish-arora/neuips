"""
Unified training dispatcher for SC3 benchmark.

Trains any registered method on multiple seeds, evaluates on all splits,
saves per-seed metrics + models + aggregated summary.json.
"""

import json
import pickle
import time
import os
from pathlib import Path

import numpy as np

from .registry import METHOD_REGISTRY, EVAL_SPLITS, DEFAULT_SEEDS, get_hp
from .data import load_all_splits, load_cached_features, CACHE_DIR
from .evaluate import compute_metrics, print_metrics_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _split_meta(splits):
    """Extract y_true / solvent_names / uncertainties per eval split."""
    meta = {}
    for sname in EVAL_SPLITS:
        df = splits[sname]
        meta[sname] = {
            "y_true": df["LogS"].values,
            "solvent_names": df["Solvent_Name"].values if "Solvent_Name" in df.columns else None,
            "uncertainties": df["Uncertainty"].values if "Uncertainty" in df.columns else None,
        }
    return meta


# =====================================================================
# Tree-based methods
# =====================================================================

def _train_tree_seed(method_key, params, seed, splits):
    from .models.tree_models import TREE_BUILDERS

    info = METHOD_REGISTRY[method_key]
    feat_name = info["featurizer"]
    tree_method = info["tree_method"]

    cached = load_cached_features(feat_name)
    if cached is None:
        raise FileNotFoundError(
            f"Feature cache not found for '{feat_name}'. Run: python sc3 cache"
        )

    X_tr, y_tr = cached["X_train"], cached["y_train"]
    X_ev, y_ev = cached["X_eval"],  cached["y_eval"]

    builder = TREE_BUILDERS[tree_method]
    model = builder(params, seed, X_tr, y_tr, X_ev, y_ev)

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        X = cached[f"X_{sname}"]
        preds = model.predict(X)
        metrics[sname] = compute_metrics(
            meta[sname]["y_true"], preds,
            meta[sname]["solvent_names"], meta[sname]["uncertainties"],
        )
    return model, metrics


# =====================================================================
# Descriptor NN methods (FastProp, FastSolv, MLP)
# =====================================================================

def _train_descriptor_nn_seed(method_key, params, seed, splits, device):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from .models.descriptor_models import (
        FastPropNet, FastSolvNet, SimpleMLP,
        compute_sobolev_targets, SOBOLEV_SCALE,
    )

    info = METHOD_REGISTRY[method_key]
    arch = info["nn_arch"]

    cached = load_cached_features("rdkit")
    X_tr, y_tr = cached["X_train"], cached["y_train"]
    X_ev, y_ev = cached["X_eval"],  cached["y_eval"]

    torch.manual_seed(seed); np.random.seed(seed)

    cls_map = {"fastprop": FastPropNet, "fastsolv": FastSolvNet, "mlp": SimpleMLP}
    is_fastsolv = arch == "fastsolv"

    if is_fastsolv:
        n_dc = X_tr.shape[1] - 4
        desc_tr = X_tr[:, :n_dc]; t_raw = X_tr[:, n_dc].copy()
        d_mu, d_sd = desc_tr.mean(0), desc_tr.std(0) + 1e-8
        desc_tr_n = ((desc_tr - d_mu) / d_sd).astype(np.float32)
        t_mu, t_sd = float(t_raw.mean()), float(t_raw.std()) + 1e-8
        t_std_arr = ((t_raw - t_mu) / t_sd).astype(np.float32)
        y_mu, y_sd = float(y_tr.mean()), float(y_tr.std()) + 1e-8
        y_tr_std = ((y_tr - y_mu) / y_sd).astype(np.float32)
        tf_raw = np.column_stack([t_raw, (10./3.)/np.clip(t_raw,1e-6,None), t_raw**2, np.log(np.clip(t_raw,1e-6,None))])
        tf_mu, tf_sd = tf_raw.mean(0).astype(np.float32), (tf_raw.std(0)+1e-8).astype(np.float32)
        grads = compute_sobolev_targets(desc_tr, t_std_arr, y_tr_std)
        in_dim = n_dc + 4
        stats = dict(d_mu=d_mu, d_sd=d_sd, t_mu=t_mu, t_sd=t_sd, y_mu=y_mu, y_sd=y_sd,
                     tf_mu=tf_mu, tf_sd=tf_sd, n_dc=n_dc)
    else:
        f_mu = X_tr.mean(0); f_sd = X_tr.std(0) + 1e-8
        X_tr_n = ((X_tr - f_mu) / f_sd).astype(np.float32)
        X_ev_n = ((X_ev - f_mu) / f_sd).astype(np.float32)
        in_dim = X_tr_n.shape[1]
        stats = dict(f_mu=f_mu, f_sd=f_sd)

    hidden = tuple(params.get("hidden_dims", (512, 256, 128)))
    model = cls_map[arch](in_dim=in_dim, hidden_dims=hidden,
                          dropout=params.get("dropout", 0.1)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    if is_fastsolv:
        desc_t = torch.tensor(desc_tr_n, device=device)
        t_t = torch.tensor(t_std_arr, device=device).unsqueeze(1)
        y_t = torch.tensor(y_tr_std, device=device)
        g_t = torch.tensor(grads, device=device)
        tm = torch.tensor([t_mu], device=device); ts = torch.tensor([t_sd], device=device)
        tfm = torch.tensor(tf_mu, device=device); tfs = torch.tensor(tf_sd, device=device)
        def _feat(d, t):
            tr = t * ts + tm
            tf = torch.cat([tr, (10./3.)/tr.clamp(1e-6), tr**2, tr.clamp(1e-6).log()], -1)
            return torch.cat([d, (tf - tfm) / tfs], -1)
        desc_ev = ((X_ev[:, :stats["n_dc"]] - d_mu) / d_sd).astype(np.float32)
        t_ev_std = ((X_ev[:, stats["n_dc"]] - t_mu) / t_sd).astype(np.float32)
        desc_v = torch.tensor(desc_ev, device=device)
        t_v = torch.tensor(t_ev_std, device=device).unsqueeze(1)
    else:
        X_t = torch.tensor(X_tr_n, device=device); y_t = torch.tensor(y_tr, device=device)
        X_v = torch.tensor(X_ev_n, device=device)

    n = len(y_tr); bs = params.get("batch_size", 256)
    best_vl, best_st, wait = float("inf"), None, 0

    for ep in range(params.get("epochs", 100)):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i+bs]; optimizer.zero_grad()
            if is_fastsolv:
                tb = t_t[idx].detach().requires_grad_(True)
                yh = model(_feat(desc_t[idx], tb))
                mse = nn.functional.mse_loss(yh, y_t[idx])
                dy = torch.autograd.grad(yh, tb, torch.ones_like(yh), create_graph=True, retain_graph=True)[0].squeeze(-1)
                vm = ~torch.isnan(g_t[idx])
                sob = SOBOLEV_SCALE * ((dy[vm] - g_t[idx][vm])**2).mean() if vm.any() else torch.zeros(1, device=device)[0]
                (mse + sob).backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            else:
                nn.functional.mse_loss(model(X_t[idx]), y_t[idx]).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = (model(_feat(desc_v, t_v)).cpu().numpy() * y_sd + y_mu) if is_fastsolv else model(X_v).cpu().numpy()
        vl = float(np.mean((vp - y_ev)**2)); scheduler.step(vl)
        if vl < best_vl:
            best_vl = vl; best_st = {k: v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if wait >= params.get("patience", 20):
            break

    if best_st: model.load_state_dict(best_st)
    model.eval()

    def predict_fn(X_raw):
        with torch.no_grad():
            if is_fastsolv:
                dc = stats["n_dc"]
                dn = ((X_raw[:, :dc] - stats["d_mu"]) / stats["d_sd"]).astype(np.float32)
                tn = ((X_raw[:, dc] - stats["t_mu"]) / stats["t_sd"]).astype(np.float32)
                return model(_feat(torch.tensor(dn, device=device),
                                   torch.tensor(tn, device=device).unsqueeze(1))).cpu().numpy() * stats["y_sd"] + stats["y_mu"]
            else:
                Xn = ((X_raw - stats["f_mu"]) / stats["f_sd"]).astype(np.float32)
                return model(torch.tensor(Xn, device=device)).cpu().numpy()

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        preds = predict_fn(cached[f"X_{sname}"])
        metrics[sname] = compute_metrics(
            meta[sname]["y_true"], preds,
            meta[sname]["solvent_names"], meta[sname]["uncertainties"],
        )
    return model, metrics


# =====================================================================
# GNN methods (GCN, GAT, GIN)
# =====================================================================

_graph_cache = None

def _get_graph_cache(splits):
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache
    import torch
    from .models.gnn_models import smiles_to_graph
    all_smi = set()
    for df in splits.values():
        all_smi.update(df["Solute"].unique()); all_smi.update(df["Solvent"].unique())
    _graph_cache = {}
    for s in all_smi:
        g = smiles_to_graph(s)
        _graph_cache[s] = g if g is not None else {
            "node_feats": torch.zeros((1,7)), "edge_index": torch.zeros((2,0), dtype=torch.long), "num_nodes": 1}
    print(f"  Graph cache: {len(_graph_cache)} molecules")
    return _graph_cache


def _train_gnn_seed(method_key, params, seed, splits, device):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from .models.gnn_models import DualGNNSolubility, SolubilityGraphDataset, batch_graph_list

    info = METHOD_REGISTRY[method_key]
    gnn_type = info["gnn_type"]
    torch.manual_seed(seed); np.random.seed(seed)
    graph_cache = _get_graph_cache(splits)
    BS = 256

    def collate(batch):
        sol_gs, solv_gs, tfs, tgts = zip(*batch)
        def _mv(bd): return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in bd.items()}
        return (_mv(batch_graph_list(sol_gs)), _mv(batch_graph_list(solv_gs)),
                torch.stack(tfs).to(device), torch.stack(tgts).to(device))

    train_dl = torch.utils.data.DataLoader(
        SolubilityGraphDataset(splits["train"], graph_cache), batch_size=BS, shuffle=True, collate_fn=collate, num_workers=0)
    eval_dl = torch.utils.data.DataLoader(
        SolubilityGraphDataset(splits["eval"], graph_cache), batch_size=BS, shuffle=False, collate_fn=collate, num_workers=0)

    model = DualGNNSolubility(
        node_dim=7, hidden_dim=params.get("hidden_dim", 64),
        num_layers=params.get("num_layers", 3), gnn_type=gnn_type,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(params.get("epochs", 100)):
        model.train()
        for sol, solv, tf, tgt in train_dl:
            optimizer.zero_grad()
            nn.functional.mse_loss(model(sol, solv, tf), tgt).backward()
            optimizer.step()

        model.eval()
        vp, vt_list = [], []
        with torch.no_grad():
            for sol, solv, tf, tgt in eval_dl:
                vp.append(model(sol, solv, tf).cpu().numpy()); vt_list.append(tgt.cpu().numpy())
        vp = np.concatenate(vp); vt = np.concatenate(vt_list)
        vl = float(np.mean((vp - vt)**2)); vrmse = float(np.sqrt(vl))
        scheduler.step(vl)
        if vl < best_vl:
            best_vl = vl; best_st = {k: v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if (ep+1) % 10 == 0 or wait == 0:
            print(f"      ep {ep+1:3d}: val_RMSE={vrmse:.4f}  ({'*best' if wait==0 else f'wait={wait}'})")
        if wait >= params.get("patience", 20):
            print(f"      Early stop at epoch {ep+1}")
            break

    if best_st: model.load_state_dict(best_st)
    model.to(device).eval()

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        ds = SolubilityGraphDataset(splits[sname], graph_cache)
        dl = torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=False, collate_fn=collate, num_workers=0)
        preds = []
        with torch.no_grad():
            for sol, solv, tf, _ in dl:
                preds.append(model(sol, solv, tf).cpu().numpy())
        preds = np.concatenate(preds)
        metrics[sname] = compute_metrics(
            meta[sname]["y_true"], preds,
            meta[sname]["solvent_names"], meta[sname]["uncertainties"],
        )
    return model, metrics


# =====================================================================
# MolMerger (merged-graph GNN with PyG AttentiveFP)
# =====================================================================

_skeleton_cache = None

def _get_skeleton_cache(splits):
    global _skeleton_cache
    if _skeleton_cache is not None:
        return _skeleton_cache
    from .models.molmerger import molmerger_skeleton
    pairs = set()
    for df in splits.values():
        for sol, solv in zip(df["Solute"].values, df["Solvent"].values):
            pairs.add((sol, solv))
    print(f"  Building MolMerger skeleton cache for {len(pairs)} pairs...")
    _skeleton_cache = {}
    failed = 0
    for sol, solv in pairs:
        g = molmerger_skeleton(sol, solv)
        if g is not None:
            _skeleton_cache[(sol, solv)] = g
        else:
            failed += 1
    print(f"  Cached {len(_skeleton_cache)} skeletons ({failed} failed)")
    return _skeleton_cache


def _train_molmerger_seed(method_key, params, seed, splits, device):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch_geometric.data import Batch
    from .models.molmerger import stamp_temperature, MolMergerNet, NODE_DIM, EDGE_DIM

    torch.manual_seed(seed); np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    skeleton_cache = _get_skeleton_cache(splits)

    class _Dataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.skeletons, self.temps, self.targets = [], [], []
            for i in range(len(df)):
                key = (df.iloc[i]["Solute"], df.iloc[i]["Solvent"])
                skel = skeleton_cache.get(key)
                if skel is not None:
                    self.skeletons.append(skel)
                    self.temps.append(float(df.iloc[i]["Temperature"]))
                    self.targets.append(float(df.iloc[i]["LogS"]))
        def __len__(self): return len(self.skeletons)
        def __getitem__(self, idx):
            return stamp_temperature(self.skeletons[idx], self.temps[idx]), self.targets[idx]

    def _collate(batch):
        graphs, targets = zip(*batch)
        return Batch.from_data_list(list(graphs)), torch.tensor(targets, dtype=torch.float32)

    bs = params.get("batch_size", 256)
    train_dl = torch.utils.data.DataLoader(
        _Dataset(splits["train"]), batch_size=bs, shuffle=True, collate_fn=_collate, num_workers=0)
    eval_dl = torch.utils.data.DataLoader(
        _Dataset(splits["eval"]), batch_size=bs, shuffle=False, collate_fn=_collate, num_workers=0)

    model = MolMergerNet(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        hidden_dim=params.get("hidden_dim", 200),
        num_layers=params.get("num_layers", 3),
        num_timesteps=params.get("num_timesteps", 2),
        dropout=params.get("dropout", 0.2),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5, min_lr=1e-6)

    best_vl, best_st, wait = float("inf"), None, 0
    for ep in range(params.get("epochs", 200)):
        model.train()
        for batch_data, targets in train_dl:
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch_data), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        vp, vt_list = [], []
        with torch.no_grad():
            for batch_data, targets in eval_dl:
                batch_data = batch_data.to(device)
                vp.append(model(batch_data).cpu().numpy())
                vt_list.append(targets.numpy())
        vp = np.concatenate(vp); vt = np.concatenate(vt_list)
        vl = float(np.mean((vp - vt)**2)); vrmse = float(np.sqrt(vl))
        scheduler.step(vl)

        if vl < best_vl:
            best_vl = vl; best_st = {k: v.cpu().clone() for k,v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
        if (ep+1) % 10 == 0 or wait == 0:
            print(f"      ep {ep+1:3d}: val_RMSE={vrmse:.4f}  ({'*best' if wait==0 else f'wait={wait}'})")
        if wait >= params.get("patience", 25):
            print(f"      Early stop at epoch {ep+1}")
            break

    if best_st: model.load_state_dict(best_st)
    model.to(device).eval()

    meta = _split_meta(splits)
    metrics = {}
    for sname in EVAL_SPLITS:
        ds = _Dataset(splits[sname])
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=_collate, num_workers=0)
        preds = []
        with torch.no_grad():
            for batch_data, _ in dl:
                batch_data = batch_data.to(device)
                preds.append(model(batch_data).cpu().numpy())
        preds = np.concatenate(preds)
        y_true = np.array(ds.targets)
        sv = meta[sname]["solvent_names"]
        unc = meta[sname]["uncertainties"]
        if sv is not None and len(sv) != len(preds):
            df = splits[sname]
            mask = [skeleton_cache.get((r["Solute"], r["Solvent"])) is not None for _, r in df.iterrows()]
            sv = sv[mask] if sv is not None else None
            unc = unc[mask] if unc is not None else None
        metrics[sname] = compute_metrics(y_true, preds, sv, unc)

    return model, metrics


# =====================================================================
# Public API
# =====================================================================

def train_method(method_key: str, seeds: list[int] | None = None,
                 gpu: int | None = None, params: dict | None = None):
    """Train a method on multiple seeds, evaluate, and save results."""
    if method_key not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method_key}'. Available: {list(METHOD_REGISTRY.keys())}")

    if seeds is None:
        seeds = DEFAULT_SEEDS
    info = METHOD_REGISTRY[method_key]
    if params is None:
        params = get_hp(method_key)

    device_str = "cpu"
    if info["model_type"] in ("descriptor_nn", "gnn", "molmerger"):
        import torch
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = None
    if info["model_type"] in ("descriptor_nn", "gnn", "molmerger"):
        import torch
        device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  {method_key}  ({info['display']}, {info['family']})")
    print(f"  seeds={seeds}  device={device_str}")
    print(f"{'='*60}")

    splits = load_all_splits(verbose=False)
    out_dir = RESULTS_DIR / method_key
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seed_metrics = {}
    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        t0 = time.time()

        if info["model_type"] == "tree":
            model, metrics = _train_tree_seed(method_key, params, seed, splits)
        elif info["model_type"] == "descriptor_nn":
            model, metrics = _train_descriptor_nn_seed(method_key, params, seed, splits, device)
        elif info["model_type"] == "gnn":
            model, metrics = _train_gnn_seed(method_key, params, seed, splits, device)
        elif info["model_type"] == "molmerger":
            model, metrics = _train_molmerger_seed(method_key, params, seed, splits, device)

        dt = time.time() - t0
        for sname in EVAL_SPLITS:
            metrics[sname]["train_time_s"] = dt

        all_seed_metrics[seed] = metrics
        print_metrics_table(metrics)
        print(f"    ({dt:.1f}s)")

        with open(out_dir / f"seed_{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)
        try:
            obj = model.state_dict() if hasattr(model, "state_dict") else model
            with open(out_dir / f"model_seed_{seed}.pkl", "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    # Aggregate
    agg = {}
    for sname in EVAL_SPLITS:
        agg[sname] = {}
        for mk in all_seed_metrics[seeds[0]][sname]:
            vals = [all_seed_metrics[s][sname][mk] for s in seeds
                    if isinstance(all_seed_metrics[s][sname].get(mk), (int, float))
                    and not np.isnan(all_seed_metrics[s][sname][mk])]
            if vals:
                agg[sname][f"{mk}_mean"] = float(np.mean(vals))
                agg[sname][f"{mk}_std"]  = float(np.std(vals))

    summary = {
        "method": method_key, "display": info["display"], "family": info["family"],
        "seeds": seeds, "params": params, "aggregated": agg,
        "per_seed": {str(s): v for s, v in all_seed_metrics.items()},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Aggregated ({method_key}):")
    for sn in EVAL_SPLITS:
        a = agg[sn]
        rm = a.get("RMSE_mean"); rs = a.get("RMSE_std", 0)
        pm = a.get("PS_RMSE_mean"); ps = a.get("PS_RMSE_std", 0)
        zm = a.get("Z_RMSE_mean")
        parts = [f"RMSE={rm:.4f}+/-{rs:.4f}"]
        if pm: parts.append(f"PS={pm:.4f}+/-{ps:.4f}")
        if zm: parts.append(f"Z={zm:.1f}")
        print(f"    {sn:15s} {'  '.join(parts)}")

    print(f"  Saved to {out_dir}")
    return summary
