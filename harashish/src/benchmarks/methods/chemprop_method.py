"""
Chemprop D-MPNN baseline for SC3 benchmark.

Uses chemprop v1.x (compatible with Python 3.10) in multicomponent mode:
  - Two SMILES inputs: Solute (D-MPNN encoder 1) + Solvent (D-MPNN encoder 2)
  - Temperature as extra features fed into the FFN after message passing
  - Default chemprop hyperparameters, 60 epochs, warmup 2 epochs

The Directed Message Passing Neural Network (D-MPNN) from Yang et al. (2019)
"Analyzing Learned Molecular Representations for Property Prediction" passes
messages along directed bonds rather than atoms. For bond u→v, the update
aggregates all incoming bonds w→u except v→u, preventing information shortcuts
that plague standard MPNN architectures.

In multicomponent mode (number_of_molecules=2), separate D-MPNN encoders
process solute and solvent graphs independently. The learned representations
are concatenated with temperature features and fed through a feed-forward
network (FFN) to predict LogS.
"""

import os
import shutil
import tempfile
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CHEMPROP_TRAIN_CMD = "chemprop_train"
CHEMPROP_PREDICT_CMD = "chemprop_predict"

DEFAULT_EPOCHS = 60
DEFAULT_DEPTH = 3
DEFAULT_HIDDEN_SIZE = 300
DEFAULT_FFN_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.0
DEFAULT_BATCH_SIZE = 50
DEFAULT_WARMUP_EPOCHS = 2.0


def _temp_features(T: np.ndarray) -> np.ndarray:
    """
    Compute temperature features for the FFN.

    Following the benchmark convention, we encode temperature as:
      T_norm = T / 300        (dimensionless, ~1.0 at room temp)
      T_inv  = 1000 / T       (captures Arrhenius-like dependence)
      T_sq   = (T/300)^2      (quadratic term)
      T_log  = log(T/300)     (log-linear term)
    """
    T = np.asarray(T, dtype=np.float64)
    return np.column_stack([
        T / 300.0,
        1000.0 / T,
        (T / 300.0) ** 2,
        np.log(T / 300.0),
    ])


def _write_data_csv(df: pd.DataFrame, path: str, include_targets: bool = True):
    """Write a chemprop-compatible data CSV (SMILES + optional targets)."""
    cols = ["Solute", "Solvent"]
    if include_targets:
        cols.append("LogS")
    df[cols].to_csv(path, index=False)


def _write_features_csv(df: pd.DataFrame, path: str):
    """Write a features CSV with temperature-derived features."""
    feats = _temp_features(df["Temperature"].values)
    pd.DataFrame(feats, columns=["T_norm", "T_inv", "T_sq", "T_log"]).to_csv(path, index=False)


def train_chemprop(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    save_dir: str,
    seed: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    depth: int = DEFAULT_DEPTH,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ffn_num_layers: int = DEFAULT_FFN_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_epochs: float = DEFAULT_WARMUP_EPOCHS,
    quiet: bool = True,
) -> str:
    """
    Train a chemprop D-MPNN model on solute+solvent with temperature features.

    Args:
        train_df: Training DataFrame with Solute, Solvent, Temperature, LogS columns.
        val_df: Validation DataFrame (same columns).
        save_dir: Directory to save the trained model.
        seed: Random seed for data ordering.
        epochs: Number of training epochs.
        depth: Message passing depth (number of D-MPNN iterations).
        hidden_size: Hidden dimension of the D-MPNN.
        ffn_num_layers: Number of FFN layers after message passing.
        dropout: Dropout probability.
        batch_size: Training batch size.
        warmup_epochs: Number of warmup epochs for learning rate schedule.
        quiet: Suppress chemprop training output.

    Returns:
        Path to the saved model directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="chemprop_sc3_")
    try:
        train_csv = os.path.join(tmp_dir, "train.csv")
        train_feat = os.path.join(tmp_dir, "train_features.csv")
        val_csv = os.path.join(tmp_dir, "val.csv")
        val_feat = os.path.join(tmp_dir, "val_features.csv")

        _write_data_csv(train_df, train_csv, include_targets=True)
        _write_features_csv(train_df, train_feat)
        _write_data_csv(val_df, val_csv, include_targets=True)
        _write_features_csv(val_df, val_feat)

        cmd = [
            CHEMPROP_TRAIN_CMD,
            "--data_path", train_csv,
            "--dataset_type", "regression",
            "--number_of_molecules", "2",
            "--smiles_columns", "Solute", "Solvent",
            "--target_columns", "LogS",
            "--features_path", train_feat,
            "--separate_val_path", val_csv,
            "--separate_val_features_path", val_feat,
            "--split_type", "random",
            "--split_sizes", "1.0", "0.0", "0.0",
            "--save_dir", str(save_dir),
            "--epochs", str(epochs),
            "--depth", str(depth),
            "--hidden_size", str(hidden_size),
            "--ffn_num_layers", str(ffn_num_layers),
            "--dropout", str(dropout),
            "--batch_size", str(batch_size),
            "--warmup_epochs", str(warmup_epochs),
            "--seed", str(seed),
            "--pytorch_seed", str(seed),
            "--metric", "rmse",
            "--num_folds", "1",
            "--num_workers", "0",
            "--cache_cutoff", "200000",
        ]

        if quiet:
            cmd.append("--quiet")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"chemprop_train failed (exit {result.returncode}):\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        return str(save_dir)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def predict_chemprop(
    df: pd.DataFrame,
    model_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Predict LogS using a trained chemprop model.

    Args:
        df: DataFrame with Solute, Solvent, Temperature columns.
        model_dir: Path to the trained model directory (containing fold_0/).
        batch_size: Prediction batch size.

    Returns:
        Array of predicted LogS values (shape: [N,]).
    """
    tmp_dir = tempfile.mkdtemp(prefix="chemprop_pred_")
    try:
        test_csv = os.path.join(tmp_dir, "test.csv")
        test_feat = os.path.join(tmp_dir, "test_features.csv")
        preds_csv = os.path.join(tmp_dir, "preds.csv")

        _write_data_csv(df, test_csv, include_targets=False)
        _write_features_csv(df, test_feat)

        model_dir = Path(model_dir)
        checkpoint_dir = model_dir / "fold_0"
        if not checkpoint_dir.exists():
            candidates = list(model_dir.glob("fold_*"))
            if candidates:
                checkpoint_dir = candidates[0]
            else:
                checkpoint_dir = model_dir

        cmd = [
            CHEMPROP_PREDICT_CMD,
            "--test_path", test_csv,
            "--preds_path", preds_csv,
            "--checkpoint_dir", str(checkpoint_dir),
            "--number_of_molecules", "2",
            "--smiles_columns", "Solute", "Solvent",
            "--features_path", test_feat,
            "--batch_size", str(batch_size),
            "--num_workers", "0",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"chemprop_predict failed (exit {result.returncode}):\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        pred_df = pd.read_csv(preds_csv)
        preds = pred_df["LogS"].values
        return preds.astype(np.float64)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def filter_valid_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with SMILES that RDKit cannot parse."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    valid = (
        df["Solute"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        & df["Solvent"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    )
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        warnings.warn(f"Filtered {n_invalid} rows with invalid SMILES")
    return df[valid].reset_index(drop=True)
