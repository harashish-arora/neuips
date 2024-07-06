"""
Sample selection: loads models/data, generates features, selects diverse
solute-solvent pairs, and compiles per-sample evidence bundles.
"""

import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from catboost import Pool
from typing import List, Dict, Tuple

from .config import PipelineConfig


class SampleSelector:
    """Selects samples and compiles model evidence for each."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device

    def load_models_and_data(self):
        print("Loading models and data...")

        self.df_test = pd.read_csv(self.config.test_file)

        self.sol_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_raw.parquet")
        ).set_index("SMILES_KEY")
        self.solv_raw = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_raw.parquet")
        ).set_index("SMILES_KEY")
        self.sol_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solute_council.parquet")
        ).set_index("SMILES_KEY")
        self.solv_council = pd.read_parquet(
            os.path.join(self.config.store_dir, "solvent_council.parquet")
        ).set_index("SMILES_KEY")

        from train_transformer import InteractionTransformer

        self.transformer = InteractionTransformer().to(self.device)
        self.transformer.load_state_dict(
            torch.load(self.config.transformer_path, map_location=self.device)
        )
        self.transformer.eval()

        self.catboost_model = joblib.load(
            os.path.join(self.config.model_dir, "model.joblib")
        )
        self.selector = joblib.load(
            os.path.join(self.config.model_dir, "selector.joblib")
        )

        self.council_feature_names = self.sol_council.columns.tolist()

        raw_solute_names = [f"Solute_{c}" for c in self.sol_raw.columns]
        raw_solvent_names = [f"Solvent_{c}" for c in self.solv_raw.columns]
        interact_names = [f"Interact_{c}" for c in self.council_feature_names]
        thermo_names = ["pred_Tm", "T_red", "T", "T_inv"]
        full_feature_names = (
            raw_solute_names + raw_solvent_names + interact_names + thermo_names
        )
        self.feature_names = np.array(full_feature_names)[
            self.selector.get_support()
        ].tolist()

        print(f"  Loaded {len(self.df_test)} test samples")
        print(f"  Loaded {len(self.feature_names)} trained features")

    def generate_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_sol = self.sol_council.loc[df["Solute"]].values.astype(np.float32)
        X_solv = self.solv_council.loc[df["Solvent"]].values.astype(np.float32)

        embeds, attns = [], []
        with torch.no_grad():
            for i in range(len(df)):
                sol = torch.tensor(X_sol[i : i + 1]).to(self.device)
                solv = torch.tensor(X_solv[i : i + 1]).to(self.device)
                _, feats, attn = self.transformer(sol, solv)
                embeds.append(feats.cpu().numpy())
                attns.append(attn.cpu().numpy())

        X_embed = np.vstack(embeds)
        attention = np.vstack(attns)

        T = df["Temperature"].values.reshape(-1, 1)
        T_inv = (1000 / df["Temperature"]).values.reshape(-1, 1)
        Tm = self.sol_raw.loc[df["Solute"], "pred_Tm"].values.reshape(-1, 1)
        T_red = T / Tm

        X_reshaped = X_embed.reshape(-1, 24, 32)
        X_mod = np.linalg.norm(X_reshaped, axis=2)
        X_sign = np.sign(X_reshaped.mean(axis=2))
        X_interact = (X_sign * X_mod) * T_inv

        X_raw = np.hstack(
            [self.sol_raw.loc[df["Solute"]].values, self.solv_raw.loc[df["Solvent"]].values]
        )

        X_full = np.hstack([X_raw, X_interact, Tm, T_red, T, T_inv])
        return X_full, attention

    def select_samples(self) -> Dict[str, List[Dict]]:
        """Select unique solute-solvent pairs and compile evidence bundles."""
        print("\nGenerating predictions for all test samples...")

        X_full, attention_weights = self.generate_features(self.df_test)
        X_pruned = self.selector.transform(X_full)
        predictions = self.catboost_model.predict(X_pruned)

        true_values = self.df_test["LogS"].values
        errors = np.abs(true_values - predictions)

        pool = Pool(X_pruned, feature_names=self.feature_names)
        shap_vals = np.array(
            self.catboost_model.get_feature_importance(pool, type="ShapValues")
        )[:, :-1]
        leaf_paths = self.catboost_model.calc_leaf_indexes(pool)

        print(
            f"  Error distribution: min={errors.min():.4f}, "
            f"max={errors.max():.4f}, mean={errors.mean():.4f}"
        )

        selected, seen_pairs = [], set()
        for idx in range(len(self.df_test)):
            pair = (
                self.df_test.iloc[idx]["Solute"],
                self.df_test.iloc[idx]["Solvent"],
            )
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                selected.append(idx)
            if len(selected) >= self.config.num_samples:
                break

        print(f"  Selected {len(selected)} unique pair predictions")

        KEY_FEATURES = [
            "Solute_num_C", "Solute_num_O", "Solute_num_N", "Solute_num_Cl",
            "Solute_num_S", "Solute_num_F", "Solute_total_atoms",
            "Solute_MolLogP", "Solute_MolWt", "Solute_TPSA", "Solute_BertzCT",
            "Solute_NumHDonors", "Solute_NumHAcceptors", "Solute_NumRotatableBonds",
            "Solute_HallKierAlpha", "Solute_LabuteASA", "Solute_HeavyAtomCount",
            "Solute_NumAromaticRings", "Solute_MaxPartialCharge", "Solute_MinPartialCharge",
            "Solvent_num_C", "Solvent_num_O", "Solvent_num_N",
            "Solvent_MolLogP", "Solvent_MolWt", "Solvent_TPSA",
            "Solvent_NumHDonors", "Solvent_NumHAcceptors", "Solvent_NumRotatableBonds",
            "pred_Tm", "T", "T_inv", "T_red",
        ]

        def compile_sample(idx: int) -> Dict:
            row = self.df_test.iloc[idx]
            structural = {}
            for feat in KEY_FEATURES:
                if feat in self.feature_names:
                    fi = self.feature_names.index(feat)
                    structural[feat] = float(X_pruned[idx, fi])

            return {
                "index": int(idx),
                "solute": row["Solute"],
                "solvent": row["Solvent"],
                "temperature": float(row["Temperature"]),
                "y_true": float(row["LogS"]),
                "y_pred": float(predictions[idx]),
                "abs_error": float(errors[idx]),
                "cross_attention_weights": attention_weights[idx].tolist(),
                "council_feature_names": self.council_feature_names,
                "shap_values": dict(
                    zip(self.feature_names, shap_vals[idx].tolist())
                ),
                "leaf_path": leaf_paths[idx].tolist(),
                "structural_features": structural,
            }

        return {"selected": [compile_sample(i) for i in selected]}
