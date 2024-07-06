"""
Evaluation metrics for SC3 benchmark.

Reports: RMSE, MAE, R^2, PS-RMSE, Z-RMSE, f_aleatoric.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, solvent_names=None, uncertainties=None):
    """
    Compute all SC3 evaluation metrics.

    Args:
        y_true: ground truth LogS values
        y_pred: predicted LogS values
        solvent_names: per-point solvent name (for PS-RMSE)
        uncertainties: per-point uncertainty sigma (for Z-RMSE)

    Returns:
        dict of metric_name -> value
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    metrics = {}

    # Standard metrics
    metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))
    metrics["R2"] = float(r2_score(y_true, y_pred))
    metrics["N"] = len(y_true)

    # Per-solvent RMSE (PS-RMSE)
    if solvent_names is not None:
        solvent_names = np.asarray(solvent_names)
        unique_solvents = np.unique(solvent_names)
        per_solvent_rmses = []
        per_solvent_r2s = []
        for s in unique_solvents:
            mask = solvent_names == s
            if mask.sum() < 2:
                continue
            rmse_s = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
            per_solvent_rmses.append(rmse_s)
            var_s = np.var(y_true[mask])
            if var_s > 0:
                r2_s = 1.0 - np.mean((y_true[mask] - y_pred[mask]) ** 2) / var_s
                per_solvent_r2s.append(float(r2_s))
        metrics["PS_RMSE"] = float(np.mean(per_solvent_rmses)) if per_solvent_rmses else np.nan
        metrics["PS_R2"] = float(np.mean(per_solvent_r2s)) if per_solvent_r2s else np.nan
        metrics["N_solvents"] = len(per_solvent_rmses)

    # Aleatoric-aware metrics (Z-RMSE)
    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties, dtype=np.float64)
        valid = ~np.isnan(uncertainties) & (uncertainties > 0)
        if valid.sum() > 0:
            # Clip uncertainties to avoid near-zero values inflating Z-RMSE
            unc_clipped = np.clip(uncertainties[valid], a_min=0.01, a_max=None)
            z = (y_true[valid] - y_pred[valid]) / unc_clipped
            metrics["Z_RMSE"] = float(np.sqrt(np.mean(z ** 2)))
            # Fraction within 2*sigma (use original uncertainties)
            within = np.abs(y_true[valid] - y_pred[valid]) <= 2 * uncertainties[valid]
            metrics["f_aleatoric"] = float(np.mean(within))
            metrics["N_with_uncertainty"] = int(valid.sum())

    return metrics


def format_metrics(metrics: dict, name: str = "") -> str:
    """Format metrics dict as a readable string."""
    parts = [f"{name}" if name else ""]
    parts.append(f"  RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.4f}")
    if "PS_RMSE" in metrics:
        parts.append(f"  PS-RMSE={metrics['PS_RMSE']:.4f}  PS-R2={metrics['PS_R2']:.4f}  ({metrics['N_solvents']} solvents)")
    if "Z_RMSE" in metrics:
        parts.append(f"  Z-RMSE={metrics['Z_RMSE']:.4f}  f_aleatoric={metrics['f_aleatoric']:.3f}")
    return "\n".join(parts)


def evaluate_predictions(df, pred_col="pred_LogS"):
    """
    Evaluate predictions stored in a DataFrame with columns:
      LogS (true), pred_LogS (predicted), Solvent_Name, Uncertainty (optional)
    """
    solvent_col = "Solvent_Name" if "Solvent_Name" in df.columns else None
    unc_col = "Uncertainty" if "Uncertainty" in df.columns else None

    return compute_metrics(
        y_true=df["LogS"].values,
        y_pred=df[pred_col].values,
        solvent_names=df[solvent_col].values if solvent_col else None,
        uncertainties=df[unc_col].values if unc_col else None,
    )
