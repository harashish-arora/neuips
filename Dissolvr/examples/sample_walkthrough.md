# Appendix: Sample Walkthrough — Benzoic Acid in Ethylene Glycol at 331 K

This appendix provides a complete worked example of the LLM-Augmented Mechanistic Explainer, showing the exact inputs and outputs at each stage. For a simplified overview of the pipeline, see `simple_walkthrough.md`. For the full prompt templates, see `prompts.md`.

**Prediction Summary:**

| Property | Value |
|---|---|
| Solute SMILES | `O=C(O)c1ccccc1` (Benzoic acid) |
| Solvent SMILES | `OCCO` (Ethylene glycol) |
| Temperature | 331.0 K |
| True LogS | 0.5107 |
| Predicted LogS | 0.6012 |
| Absolute Error | 0.0905 |

---

## Stage 0a — Solute Description

**System Prompt:** *You are a chemist describing molecular structures based on SMILES. Focus on core structure, key functional groups, polarity, hydrogen bonding capacity, size. Be direct and specific. No solubility predictions.*

**User Prompt:** *Describe the following molecule based ONLY on its SMILES structure. SMILES: O=C(O)c1ccccc1*

**LLM Response:**

> The molecule consists of a six-membered aromatic benzene ring directly substituted with a carboxylic acid functional group. This creates a distinctly amphiphilic structure, where the nonpolar phenyl ring provides rigid hydrophobic bulk, while the carboxylic acid acts as a highly polar, ionizable center capable of strong hydrogen bond donation and acceptance. As a relatively small molecule, its physicochemical character is defined by the direct conjugation between the electron-rich aromatic pi system and the electron-withdrawing carbonyl group.

---

## Stage 0b — Solvent Description

**User Prompt:** *Describe the following molecule based ONLY on its SMILES structure. SMILES: OCCO*

**LLM Response:**

> The molecule consists of a minimal two-carbon aliphatic chain with a primary hydroxyl group attached to each end. This highly symmetrical, low-molecular-weight structure is extremely polar due to the high ratio of electronegative oxygen atoms to carbon atoms. What makes this molecule distinctive is its exceptional hydrogen-bonding capacity relative to its very small size, acting simultaneously as a strong double donor and quadruple acceptor.

---

## Stage 1 — Evidence Summary (Blinded — No Prediction Shown)

**Data provided to LLM:**

SHAP Feature Contributions (Top 20):
```
POSITIVE CONTRIBUTIONS (increasing solubility):
  Solute_TPSA: +0.2405
  T: +0.1321
  T_inv: +0.1219
  Solute_BertzCT: +0.1015
  Solute_MolLogP: +0.0951
  Solute_NumHeteroatoms: +0.0826
  Solute_PEOE_VSA7: +0.0514
  Solute_PEOE_VSA8: +0.0406
  Solute_MolWt: +0.0350
  Solute_BalabanJ: +0.0348

NEGATIVE CONTRIBUTIONS (decreasing solubility):
  Solvent_PEOE_VSA8: -0.0482
  Solute_FractionCSP3: -0.0264
```

Group Contributions:
```
  Solute: +1.2522
  Solvent: +0.0842
  Interact: +0.0034
  Thermo: +0.2657

  Dominant group: Solute (78.0% of total signal)
```

Structural Feature Values (top 15 by magnitude):
```
  T: 331.0000
  Solute_BertzCT: 203.4160
  pred_Tm: 150.0000
  Solute_MolWt: 122.1230
  Solvent_MolWt: 62.0680
  Solute_LabuteASA: 52.7521
  Solvent_TPSA: 40.4600
  Solute_TPSA: 37.3000
  Solute_HeavyAtomCount: 9.0000
  T_inv: 3.0211
  T_red: 2.2067
  Solvent_NumHDonors: 2.0000
  Solvent_NumHAcceptors: 2.0000
  Solute_MolLogP: 1.3848
  Solute_HallKierAlpha: -1.3100
```

Unusual Values:
```
  Solvent_TPSA=40.4600 (HIGH: 2.46σ above mean, range: [2.46, 36.52])
  Solvent_NumHDonors=2.0000 (HIGH: 2.85σ above mean, range: [-0.56, 1.55])
```

**LLM Response:**

> The prediction is overwhelmingly driven by the solute's properties, which account for 78% of the total signal, alongside a strong thermal boost. Specifically, the solute's moderate polarity (`Solute_TPSA` = 37.3) and structural complexity (`Solute_BertzCT` = 203.4) combine with the elevated temperature (T = 331 K) to strongly increase the predicted solubility. Interestingly, the solvent possesses unusually high polarity and hydrogen-bonding capacity (`Solvent_TPSA` = 40.46 and `Solvent_NumHDonors` = 2.0, both >2.4σ above the mean), yet the solvent feature group contributes very little (+0.0842) to the overall model output. Competing effects are virtually non-existent, with only minor penalties from `Solvent_PEOE_VSA8` (-0.0482) and `Solute_FractionCSP3` (-0.0264) failing to meaningfully offset the broad positive consensus. Ultimately, the model relies on the solute's favorable physicochemical profile and the high temperature to predict solubility, largely bypassing the solvent's extreme hydrogen-bonding characteristics.

---

## Stage 2 — Decision Analysis (Prediction Revealed)

**Data provided to LLM:**

```
Predicted LogS: 0.6012
Temperature (K): 331.00

Top Cross-Attention Interactions:
  MolWt -> MolLogP: 0.5240
  abraham_V -> MolLogP: 0.3489
  MolLogP -> MolLogP: 0.3086
  MaxPartialCharge -> HeavyAtomCount: 0.2930
  Acidic -> MolLogP: 0.2832
  NumHAcceptors -> HeavyAtomCount: 0.2713
  NumHAcceptors -> abraham_E: 0.2534
  Basic -> abraham_B: 0.2519
  Acidic -> abraham_S: 0.2479
  TPSA -> MolLogP: 0.2151

Leaf Path Statistics:
  Number of trees: 3000
  Path variability: Mean leaf: 94.7, Std: 70.9

Key Decision Features:
  Solute_TPSA: +0.2405 (positive contribution)
  T: +0.1321 (positive contribution)
  T_inv: +0.1219 (positive contribution)
  Solute_BertzCT: +0.1015 (positive contribution)
  Solute_MolLogP: +0.0951 (positive contribution)
  Solute_NumHeteroatoms: +0.0826 (positive contribution)
  Solute_PEOE_VSA7: +0.0514 (positive contribution)
  Solvent_PEOE_VSA8: -0.0482 (negative contribution)
  Solute_PEOE_VSA8: +0.0406 (positive contribution)
  Solute_MolWt: +0.0350 (positive contribution)
```

**LLM Response:**

> The model's cross-attention heavily prioritizes the solvent's lipophilicity, with the strongest interactions occurring between the solute's size or volume and the solvent's `MolLogP` (`MolWt -> MolLogP` at 0.5240; `abraham_V -> MolLogP` at 0.3489). This indicates that the model primarily assesses compatibility by evaluating how the solute's steric bulk and hydrophobic properties align with the solvent's overall polarity. The positive predicted LogS of 0.6012 is predominantly driven by strong upward contributions from the solute's polar surface area (`Solute_TPSA`: +0.2405) and the elevated thermodynamic conditions (`T`: +0.1321, `T_inv`: +0.1219). Despite these strong, cohesive positive drivers, the decision path exhibits moderate variability (mean leaf: 94.7, std: 70.9). This standard deviation suggests a somewhat mixed consensus among the 3000 trees, likely as different branches weigh the dominant polar and thermal effects against minor negative solvent penalties like `Solvent_PEOE_VSA8` (-0.0482).

---

## Stage 3 — Final Integrated Explanation

**Input:** Molecule descriptions from Stage 0, evidence summary from Stage 1, decision analysis from Stage 2, and the prediction.

**LLM Response:**

> **Prediction and Key Drivers**
> The model predicts a LogS of 0.6012, classifying the molecule as very highly soluble. This outcome is overwhelmingly driven by the solute's properties, which account for 78 percent of the total signal, particularly its moderate polar surface area (37.3) and structural complexity (203.4). The elevated temperature of 331.00 K also acts as a primary driver, providing a strong positive thermodynamic boost (+0.1321) to the overall solubility.
>
> **Solute-Solvent Compatibility**
> The model's attention mechanism focuses heavily on the interplay between the solute's physical dimensions and the solvent's overall lipophilicity. The strongest cross-attention weights link the solvent's partition coefficient to the solute's molecular weight (0.5240) and volumetric size (0.3489). This indicates that the model evaluates compatibility by assessing how the solute's steric bulk and hydrophobic phenyl ring align with the highly polar, low-lipophilicity environment of the solvent.
>
> **Mechanistic Interpretation**
> The dissolution process is largely dictated by the solute's favorable physicochemical profile and the high thermal energy of the system. Interestingly, while the solvent has an exceptionally high polar surface area (40.46) and hydrogen-bonding capacity (2.0 donors), these solvent-specific traits contribute very little (+0.0842) to the overall prediction. Instead, the strong upward driving force from the solute's polarity (+0.2405) and the thermal energy (+0.1321) suggests that at 331 K, the kinetic energy easily overcomes the solute's crystal lattice, allowing its carboxylic acid group to dissolve without needing to heavily weight specific solvent interactions.
>
> **Confidence and Caveats**
> The prediction is generally reliable due to a broad positive consensus among the primary thermal and solute-driven features, with only negligible penalties like a minor solvent surface area effect (-0.0482). However, the decision path exhibits moderate variability, with a standard deviation of 70.9 against a mean leaf value of 94.7. This variance likely reflects slight algorithmic uncertainty in how to balance the dominant thermal effects against the model's unusual choice to largely ignore the solvent's extreme hydrogen-bonding capabilities.

---

## Stage 3.5 — Validation

**Input:** The Stage 3 explanation alongside all source evidence (SHAP values, cross-attention weights, group contributions, structural features, tree statistics, SMILES).

**LLM Response:**

```json
{
  "verdict": "supported",
  "unsupported_claims": [],
  "supported_points": [
    "The predicted LogS of 0.6012 and its classification as very highly soluble.",
    "The solute group contribution of 78 percent.",
    "The solute's TPSA (37.3) and BertzCT (203.4) values.",
    "The temperature of 331.00 K and its SHAP contribution of +0.1321.",
    "The cross-attention weights for MolWt -> MolLogP (0.5240) and abraham_V -> MolLogP (0.3489).",
    "The solvent's unusual TPSA (40.46) and NumHDonors (2.0).",
    "The solvent group's minor contribution of +0.0842.",
    "The solute TPSA's SHAP contribution of +0.2405.",
    "The minor penalty from Solvent_PEOE_VSA8 (-0.0482).",
    "The tree statistics (mean leaf 94.7, std 70.9)."
  ],
  "generic_claims_to_remove": [],
  "evidence_specific_points_to_keep": [
    "The mechanistic interpretation that high thermal energy and solute polarity drive dissolution, overriding the need for strong solvent-specific interactions despite the solvent's high hydrogen-bonding capacity.",
    "The connection between the cross-attention features (MolWt, abraham_V, MolLogP) and the physical interplay of steric bulk and lipophilicity."
  ],
  "correction_instructions": [],
  "needs_revision": false
}
```

> Validation passed — no correction needed (Stage 3.6 skipped).

---

## Stage 4 — Condensed Explanation (Final Output)

**LLM Response:**

> The model predicts a very highly soluble LogS of 0.6012 at an elevated temperature of 331.00 K. This outcome is predominantly driven by solute properties, which account for 78 percent of the total predictive signal. The most influential molecular features driving this solubility are the moderate polar surface area of 37.3 and a structural complexity of 203.4. Furthermore, the high thermal energy provides a strong positive thermodynamic boost of +0.1321 to the overall prediction.
>
> The attention mechanism heavily emphasizes the interplay between the physical dimensions of the solute and the overall lipophilicity of the solvent. Strong cross-attention weights link the solvent partition coefficient to the solute molecular weight at 0.5240 and volumetric size at 0.3489. This indicates compatibility is evaluated by assessing how the solute steric bulk and hydrophobic phenyl ring align with the highly polar, low-lipophilicity solvent environment. Despite possessing an exceptionally high polar surface area of 40.46 and a hydrogen-bonding capacity of 2.0 donors, these specific solvent traits contribute a minimal +0.0842 to the overall prediction, offering only marginal assistance to dissolution.
>
> The dissolution process is primarily solute-limited, dictated by the favorable physicochemical profile of the solute and the high thermal energy of the system. A strong upward driving force from the solute polarity contribution of +0.2405 and the thermal energy boost of +0.1321 facilitates the process. At 331.00 K, the kinetic energy easily overcomes the solute crystal lattice. This allows the carboxylic acid group to dissolve efficiently without requiring significant stabilization from specific solvent interactions.
>
> The prediction is generally reliable due to a broad positive consensus among the primary thermal and solute-driven features. There are only negligible conflicting signals, such as a minor solvent surface area penalty of -0.0482. However, the decision path exhibits moderate variability, evidenced by a standard deviation of 70.9 against a mean leaf value of 94.7. This variance indicates slight algorithmic uncertainty within expected model behavior, reflecting the challenge of balancing dominant thermal effects against the unusual choice to largely ignore the extreme hydrogen-bonding capabilities of the solvent.
