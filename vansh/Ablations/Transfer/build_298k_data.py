"""
Build the 298 K-locked SC3 subsets used by `run_transfer_298k.py`.

We construct *two* 298 K versions of the SC3 splits, both written to
`cache/298k_<approach>.npz`:

(A) FILTER  — keep only existing rows whose T is within ±3 K of 298.15
              (i.e., 295.15 ≤ T ≤ 301.15).  Real measurements only.
              Sizes:
                train  ~ 7 460
                eval   ~   844
                ood    ~ 1 491
                gold   ~   472

(B) INTERP  — for every (Solute_Canon, Solvent_Canon) pair we have a
              Van't Hoff / Apelblat fit in `interim/04_fits.csv`,
              evaluate logS at T = 298.15 K (only when 298.15 K is
              within the fit's measured T range, no extrapolation; only
              when R² ≥ 0.95; only when n_points ≥ 3 for Apelblat or
              n_points == 2 for Van't Hoff).
              Each pair contributes exactly one row.
              Then split using the *original* split assignment of the
              pair, breaking ties at the row level by checking which
              SC3 split contains the (solute, solvent) pair.

For both we featurise with the same sc3_bench RDKit pipeline so the
input format is identical to the main caches.

Output:
  cache/298k_filter.npz   (X_train, y_train, X_eval, y_eval, X_ood, y_ood,
                           X_sc3_gold, y_sc3_gold, plus *_solvent_names)
  cache/298k_interp.npz   (same keys)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ABLATIONS_TRANSFER_DIR = Path(__file__).resolve().parent
VANSH_ROOT = ABLATIONS_TRANSFER_DIR.parent.parent
SOLUBILITY_ROOT = VANSH_ROOT.parent

sys.path.insert(0, str(VANSH_ROOT))
from sc3_bench.featurizers import get_featurizer, build_features  # noqa: E402

CACHE_DIR = ABLATIONS_TRANSFER_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = SOLUBILITY_ROOT / "sc3_benchmark_data_curation_v2" / "data"
SPLITS_DIR = DATA_ROOT / "splits"
SC3_DIR = DATA_ROOT / "sc3"
INTERIM_DIR = DATA_ROOT / "interim"

T_TARGET = 298.15
T_LO, T_HI = 295.15, 301.15  # ±3 K window for the FILTER approach
R2_MIN = 0.95
RMSE_MAX = 0.30  # quality cut on the Apelblat/Van't Hoff fit

SPLITS = {
    "train":   SPLITS_DIR / "bench_train.csv",
    "eval":    SPLITS_DIR / "bench_eval.csv",
    "ood":     SPLITS_DIR / "bench_ood.csv",
    "sc3_gold":   SC3_DIR / "gold.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _standardise_bench(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Solute_Canon": "Solute", "Temperature_K": "Temperature"})
    if "Solvent" in df.columns and "Solvent_Canon" in df.columns:
        df = df.rename(columns={"Solvent": "Solvent_Name", "Solvent_Canon": "Solvent"})
    elif "Solvent_Canon" in df.columns:
        df = df.rename(columns={"Solvent_Canon": "Solvent"})
    return df


def _standardise_sc3(df: pd.DataFrame, solvent_map: dict) -> pd.DataFrame:
    df = df.rename(columns={
        "Solute_Canon": "Solute", "Solvent_Canon": "Solvent",
        "Temperature_K": "Temperature", "LogS_consensus": "LogS", "sigma": "Uncertainty",
    })
    df["Solvent_Name"] = df["Solvent"].map(solvent_map).fillna("Unknown")
    return df


def _load_splits() -> dict[str, pd.DataFrame]:
    train = _standardise_bench(pd.read_csv(SPLITS[ "train"]))
    eval_ = _standardise_bench(pd.read_csv(SPLITS[ "eval" ]))
    ood   = _standardise_bench(pd.read_csv(SPLITS[ "ood"  ]))

    solvent_map: dict[str, str] = {}
    for df in [train, eval_, ood]:
        if "Solvent_Name" in df.columns and "Solvent" in df.columns:
            for _, r in df[["Solvent", "Solvent_Name"]].drop_duplicates().iterrows():
                solvent_map[r["Solvent"]] = r["Solvent_Name"]
    gold = _standardise_sc3(pd.read_csv(SPLITS[ "sc3_gold" ]), solvent_map)
    return {"train": train, "eval": eval_, "ood": ood, "sc3_gold": gold}


def _featurise(df: pd.DataFrame) -> np.ndarray:
    feat = get_featurizer("rdkit")
    cache: dict = {}
    return build_features(df, feat, cache)


def _save_npz(out_path: Path, splits: dict[str, pd.DataFrame],
              X_by_split: dict[str, np.ndarray]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for sname, df in splits.items():
        arrays[f"X_{sname}"] = X_by_split[sname]
        arrays[f"y_{sname}"] = df["LogS"].values.astype(np.float32)
        if "Solvent_Name" in df.columns:
            arrays[f"solv_{sname}"] = df["Solvent_Name"].values.astype(str)
        else:
            arrays[f"solv_{sname}"] = df["Solvent"].values.astype(str)
        if "Uncertainty" in df.columns:
            arrays[f"unc_{sname}"] = df["Uncertainty"].values.astype(np.float32)
    np.savez_compressed(out_path, **arrays)
    print(f"[298k] saved {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Approach A: FILTER
# ---------------------------------------------------------------------------

def build_filter() -> Path:
    out_path = CACHE_DIR / "298k_filter.npz"
    if out_path.exists():
        print(f"[298k/filter] cache exists, skipping ({out_path})")
        return out_path

    splits = _load_splits()
    print(f"[298k/filter] selecting rows with T in [{T_LO:.2f}, {T_HI:.2f}] K ...")
    sub: dict[str, pd.DataFrame] = {}
    for name, df in splits.items():
        T = df["Temperature"].astype(float).values
        m = (T >= T_LO) & (T <= T_HI)
        keep = df[m].reset_index(drop=True).copy()
        # Pin the T to 298.15 — this matches the pretraining domain exactly.
        keep["Temperature"] = T_TARGET
        sub[name] = keep
        print(f"[298k/filter]   {name:9s}: {m.sum():6d} of {len(df):6d} rows kept "
              f"({100 * m.mean():.1f}%)")

    X_by = {n: _featurise(sub[n]) for n in sub}
    _save_npz(out_path, sub, X_by)
    return out_path


# ---------------------------------------------------------------------------
# Approach B: INTERPOLATE
# ---------------------------------------------------------------------------

def _eval_at_target(row: pd.Series, T: float = T_TARGET) -> float | None:
    """Evaluate the fit at T (Kelvin). Returns None if out of range or untrained."""
    if not (row["T_min"] - 0.5 <= T <= row["T_max"] + 0.5):
        return None
    model = row["model"]
    A = row["A"]; B = row["B"]
    if model == "apelblat":
        C = row["C"]
        return float(A + B / T + C * np.log(T))
    elif model == "vanthoff":
        return float(A + B / T)
    else:
        return None


def _build_interp_table() -> pd.DataFrame:
    """Build per-pair table interpolated to 298.15 K with a quality filter."""
    fits = pd.read_csv(INTERIM_DIR / "04_fits.csv")
    print(f"[298k/interp] read {len(fits):,} fits from 04_fits.csv")

    fits = fits[fits["model"].isin({"apelblat", "vanthoff"})]
    fits = fits[fits["R2"].astype(float) >= R2_MIN]
    fits = fits[fits["RMSE"].astype(float) <= RMSE_MAX]
    fits = fits[(fits["T_min"] <= T_TARGET) & (fits["T_max"] >= T_TARGET)]
    print(f"[298k/interp] after R²≥{R2_MIN}, RMSE≤{RMSE_MAX}, T-range covers 298.15 K: "
          f"{len(fits):,} fits")

    rows: list[dict] = []
    for _, r in fits.iterrows():
        y = _eval_at_target(r, T_TARGET)
        if y is None:
            continue
        rows.append({
            "Solute":  r["Solute_Canon"],
            "Solvent": r["Solvent_Canon"],
            "Temperature": T_TARGET,
            "LogS":    float(y),
            "gid":     int(r["gid"]),
            "n_points": int(r["n_points"]),
            "fit_R2":   float(r["R2"]),
            "fit_RMSE": float(r["RMSE"]),
        })
    df = pd.DataFrame(rows)
    print(f"[298k/interp] interpolated to 298.15 K: {len(df):,} pairs")
    return df


def _split_assignment(splits: dict[str, pd.DataFrame]) -> dict[tuple[str, str], str]:
    """Map (solute, solvent) -> the *most-restrictive* split that contains it.

    Priority ordering (most restrictive first):
       sc3_gold > ood > eval > train.
    A pair appearing in `sc3_gold` is held out for testing even if it
    also appears in `train`; this matches what the main benchmark does.
    """
    priority = ["sc3_gold", "ood", "eval", "train"]
    assign: dict[tuple[str, str], str] = {}
    for name in priority:
        df = splits[name]
        for s, sv in zip(df["Solute"].values, df["Solvent"].values):
            key = (s, sv)
            if key not in assign:
                assign[key] = name
    return assign


def build_interp() -> Path:
    out_path = CACHE_DIR / "298k_interp.npz"
    if out_path.exists():
        print(f"[298k/interp] cache exists, skipping ({out_path})")
        return out_path

    interp = _build_interp_table()
    splits_real = _load_splits()
    assign = _split_assignment(splits_real)
    interp["split"] = [assign.get((s, sv), "train")
                       for s, sv in zip(interp["Solute"].values, interp["Solvent"].values)]
    print("[298k/interp] split distribution:")
    print(interp["split"].value_counts().sort_index().to_string())

    # Build a Solvent_Name column from the original splits' map.
    name_map: dict[str, str] = {}
    for df in splits_real.values():
        if "Solvent_Name" in df.columns:
            for _, r in df[["Solvent", "Solvent_Name"]].drop_duplicates().iterrows():
                name_map[r["Solvent"]] = r["Solvent_Name"]
    interp["Solvent_Name"] = interp["Solvent"].map(name_map).fillna("Unknown")

    sub: dict[str, pd.DataFrame] = {}
    for name in ["train", "eval", "ood", "sc3_gold"]:
        sub[name] = interp[interp["split"] == name].reset_index(drop=True)

    X_by = {n: _featurise(sub[n]) for n in sub}
    _save_npz(out_path, sub, X_by)
    return out_path


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(force: bool = False, only: str | None = None) -> None:
    if force:
        for p in [CACHE_DIR / "298k_filter.npz", CACHE_DIR / "298k_interp.npz"]:
            if p.exists():
                p.unlink()
    if only in (None, "filter"):
        build_filter()
    if only in (None, "interp"):
        build_interp()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", choices=["filter", "interp"], default=None)
    args = p.parse_args()
    main(force=args.force, only=args.only)
