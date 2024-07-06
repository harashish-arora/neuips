"""
Evaluation metrics for SC3 benchmark.

Computes: RMSE, MAE, R2, PS-RMSE, PS-R2, Z-RMSE, f_aleatoric.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, solvent_names=None, uncertainties=None) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
        "N":    len(y_true),
    }

    if solvent_names is not None:
        solvent_names = np.asarray(solvent_names)
        per_solvent_rmses, per_solvent_r2s = [], []
        for s in np.unique(solvent_names):
            mask = solvent_names == s
            if mask.sum() < 2:
                continue
            rmse_s = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            per_solvent_rmses.append(rmse_s)
            var_s = np.var(y_true[mask])
            if var_s > 0:
                r2_s = 1.0 - np.mean((y_true[mask] - y_pred[mask]) ** 2) / var_s
                per_solvent_r2s.append(float(r2_s))
        metrics["PS_RMSE"]    = float(np.mean(per_solvent_rmses)) if per_solvent_rmses else np.nan
        metrics["PS_R2"]      = float(np.mean(per_solvent_r2s))   if per_solvent_r2s else np.nan
        metrics["N_solvents"] = len(per_solvent_rmses)

    if uncertainties is not None:
        unc = np.asarray(uncertainties, dtype=np.float64)
        valid = ~np.isnan(unc) & (unc > 0)
        if valid.sum() > 0:
            unc_clip = np.clip(unc[valid], 0.01, None)
            z = (y_true[valid] - y_pred[valid]) / unc_clip
            metrics["Z_RMSE"]             = float(np.sqrt(np.mean(z ** 2)))
            within = np.abs(y_true[valid] - y_pred[valid]) <= 2 * unc[valid]
            metrics["f_aleatoric"]        = float(np.mean(within))
            metrics["N_with_uncertainty"] = int(valid.sum())

    return metrics


def print_metrics_table(all_metrics: dict):
    header = f"{'Split':15s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s} {'PS_RMSE':>8s} {'Z_RMSE':>8s}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        ps = m.get("PS_RMSE", float("nan"))
        ps_str = f"{ps:8.4f}" if not np.isnan(ps) else "     n/a"
        zr = m.get("Z_RMSE", float("nan"))
        zr_str = f"{zr:8.2f}" if not np.isnan(zr) else "     n/a"
        print(f"{name:15s} {m['RMSE']:7.4f} {m['MAE']:7.4f} {m['R2']:7.4f} {ps_str} {zr_str}")
