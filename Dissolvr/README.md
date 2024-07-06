# Molecular Solubility Prediction

## Overview

This repository contains the codebase for our molecular solubility prediction framework. We address solubility prediction across two distinct regimes:

- **Regime I**: Aqueous solubility prediction (single solvent — water)
- **Regime II**: Multi-solvent solubility prediction with temperature dependence

Our approach combines physics-informed molecular descriptors with learned interaction representations.

---

## Repository Structure

```
├── regime-i/                   # Aqueous solubility (single solvent)
│   ├── train.py               # Main training script
│   ├── featurizer.py          # Molecular feature extraction
│   ├── eval_test.py           # Test set evaluation
│   ├── ablations.py           # Ablation studies
│   └── all_datasets/          # Dataset curation scripts (AqSolDB, ESOL, SC2)
│
├── regime-ii/                  # Multi-solvent solubility with temperature
│   ├── train.py               # Main training pipeline
│   ├── train_transformer.py   # Interaction Transformer training
│   ├── council.py             # Council feature extraction (24 members)
│   ├── featurizer.py          # Full molecular featurization
│   ├── generate_features.py   # Feature store generation
│   ├── ablations.py           # Ablation study scripts
│   ├── eval_test.py           # Test set evaluation
│   ├── anomaly.py             # Feature threshold analysis
│   └── all_datasets/          # Dataset curation scripts (BigSol 1.0/2.0, Leeds)
│
├── baselines/                  # Baseline implementations
│   ├── regime-i/              # Aqueous baselines (SolTranNet, SolubNet, AqSolPred, etc.)
│   └── regime-ii/             # Multi-solvent baselines (FastSolv, Chemprop, GNNs, etc.)
│
├── explainer/                  # LLM-Augmented Mechanistic Explainer
│   ├── prompts.json           # All prompt templates (editable)
│   ├── stages.py              # Five LLM stages (description → evidence → decision → integration → validation → condensation)
│   ├── pipeline.py            # Async orchestrator
│   ├── sample_selector.py     # Sample selection and evidence compilation
│   ├── config.py              # Configuration and API key management
│   └── __main__.py            # CLI: python -m explainer --api-keys KEY1 KEY2
│
├── examples/                   # Documentation and worked examples
│   ├── prompts.md             # Complete prompt templates (appendix)
│   ├── sample_walkthrough.md  # Full worked example with real data
│   └── simple_walkthrough.md  # Simplified pipeline overview
│
├── apelblat/                   # Apelblat equation validation
│   ├── apelblat_experiment.py # Main experiment script
│   ├── council.py             # Council features for Apelblat
│   ├── featurizer.py          # Featurizer for Apelblat
│   └── train.py               # Training for Apelblat validation
│
└── requirements.txt            # Python dependencies
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 12.x (for GPU acceleration, optional)

### Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.1+ (with CUDA support for GPU)
- RDKit 2024+
- CatBoost
- scikit-learn
- pandas, numpy

---

## Quick Start

### Regime I: Aqueous Solubility

```bash
cd regime-i
python train.py
python eval_test.py
```

### Regime II: Multi-Solvent Solubility

```bash
cd regime-ii

# Step 1: Generate features
python generate_features.py

# Step 2: Train the Interaction Transformer
python train_transformer.py

# Step 3: Train the final model
python train.py

# Evaluate
python eval_test.py
```

### LLM-Augmented Explainer

```bash
python -m explainer --api-keys YOUR_GEMINI_KEY --dry-run  # test sample selection
python -m explainer --api-keys KEY1 KEY2 KEY3             # run full pipeline
```

See `examples/` for prompt templates and a worked example.

---

## Method Overview

### Core Architecture

1. **Council of Features (24 Members)**: Physics-informed molecular descriptors covering:
   - Global properties (MolLogP, TPSA, H-bond donors/acceptors, etc.)
   - Functional super-groups (Acidic, Basic, Protic, Polar, Halogen, Aromatic)
   - Thermodynamic proxies (Joback T_m, Abraham parameters)

2. **Interaction Transformer**: Cross-attention mechanism that learns solute-solvent interactions:
   - Projects solute and solvent council features into embedding space
   - Multi-head cross-attention captures molecular compatibility
   - Outputs learned interaction features

3. **Gradient Boosting Predictor**: CatBoost model with:
   - Physical feature backbone
   - Learned interaction embeddings
   - Temperature-modulated interaction terms
   - Monotonicity constraints for thermodynamic consistency

---

## Datasets

### Regime I
| Dataset | Train | Test | Description |
|---------|-------|------|-------------|
| AqSolDB | 8,000 | 1,000 | Curated aqueous solubility |
| ESOL | 1,024 | 128 | Delaney ESOL benchmark |
| SC2 | 2,300 | 287 | Solubility Challenge 2 |

### Regime II
| Dataset | Train | Test | Description |
|---------|-------|------|-------------|
| BigSol 1.0 | ~50,000 | ~6,000 | Multi-solvent solubility |
| BigSol 2.0 | ~60,000 | ~7,500 | Extended BigSol |
| Leeds | ~3,000 | ~400 | Temperature-varied data |

---

## Experiments

### Running Ablations

```bash
cd regime-ii
python ablations.py
```

### Running Baselines

```bash
cd baselines/regime-i
python baselining_generic_methods.py

cd baselines/regime-ii
python baselining_generic_methods.py
python baselining_gnn_methods.py
```

### Apelblat Validation

```bash
cd apelblat
python apelblat_experiment.py
```

---

## License

This code is released under the MIT License.
