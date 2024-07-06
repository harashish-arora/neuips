# Appendix: LLM Explainer Pipeline — Overview

This appendix provides a simplified overview of DISSOLVR's 6-stage LLM-Augmented Mechanistic Explainer pipeline. For the complete prompt templates, see `prompts.md`. For a full worked example with real inputs and outputs, see `sample_walkthrough.md`.

---

## Pipeline Summary

The explainer takes a single prediction from DISSOLVR (a solute–solvent pair at a given temperature) and produces a chemist-readable natural-language explanation grounded in the model's actual decision logic. The pipeline is designed with two key safeguards: (1) **blinding** — the predicted LogS value is withheld from the LLM until Stage 2, forcing it to reason from evidence before seeing the answer; and (2) **validation** — a separate LLM call cross-checks every claim against the source data.

---

## Stage 0: Molecular Description (Blinded)

**Input:** SMILES string for the solute; SMILES string for the solvent (processed independently).

**Task:** The LLM describes the molecular structure, functional groups, and physicochemical character based solely on the SMILES. It is explicitly prohibited from making solubility predictions.

**Output:** A 3–4 sentence structural characterisation for each molecule.

**Why:** Establishes a chemical context for later stages without biasing the LLM toward any prediction.

---

## Stage 1: Evidence Summary (Blinded)

**Input:** Top 20 SHAP feature contributions, group-level contributions (solute/solvent/interaction/thermo), raw structural feature values, and any statistically unusual feature values.

**Task:** The LLM identifies which features are driving the prediction, whether there are competing effects, and whether the solute or solvent dominates the signal.

**Output:** A 4–6 sentence analytical summary of the model evidence.

**Why:** The LLM interprets the statistical evidence *before* seeing the predicted value, preventing post-hoc rationalisation. It must reason from SHAP attributions, not from chemical intuition about what the answer "should" be.

---

## Stage 2: Decision Analysis (Prediction Revealed)

**Input:** The predicted LogS value, temperature, top cross-attention weights from the Interaction Transformer, leaf path statistics (number of trees, mean/std of leaf values), and key decision features.

**Task:** The LLM analyses which feature interactions the model prioritised, whether the decision path is consistent or mixed, and what patterns led to the predicted value.

**Output:** A 4–5 sentence analysis of the model's reasoning process.

**Why:** The prediction is revealed for the first time. The LLM can now connect the evidence patterns identified in Stage 1 to the actual predicted value, but its prior reasoning is already locked in.

---

## Stage 3: Final Integrated Explanation

**Input:** The molecule descriptions (Stage 0), evidence summary (Stage 1), decision analysis (Stage 2), and the prediction.

**Task:** The LLM synthesises all prior outputs into a unified explanation structured around four sections: Prediction & Key Drivers, Solute–Solvent Compatibility, Mechanistic Interpretation, and Confidence & Caveats. It is instructed to translate raw feature names (e.g., `Solute_MACCS_105`) into plain chemical language.

**Output:** A structured, chemist-readable explanation with specific numerical citations.

**Why:** This is the core synthesis step that connects statistical model behaviour to chemical reasoning.

---

## Stage 3.5: Validation

**Input:** The Stage 3 explanation alongside all source evidence (SHAP values, cross-attention weights, group contributions, structural features, tree statistics, and the original SMILES).

**Task:** A separate LLM call acts as a strict validator, checking every numerical claim, structural description, and mechanistic statement against the source data. It flags hallucinations, unsupported claims, generic textbook statements not grounded in evidence, and chemical implausibilities.

**Output:** A JSON verdict (`supported`, `partially_supported`, or `unsupported`) with lists of supported points, unsupported claims, and correction instructions. If `needs_revision` is true, a corrected explanation is provided.

**Why:** This prevents the "salesman" failure mode — where the LLM produces plausible-sounding but unfaithful explanations. If validation fails, the corrected explanation is fed back through Stage 3 for revision (Stage 3.6).

---

## Stage 4: Condensation

**Input:** The validated (and possibly corrected) explanation from Stage 3/3.6.

**Task:** The LLM condenses the explanation into exactly 4 dense paragraphs covering: (1) prediction summary, (2) solute–solvent dynamics, (3) mechanistic interpretation, and (4) confidence and uncertainty.

**Output:** A compact, information-dense 4-paragraph explanation suitable for reports or direct consumption by chemists.

**Why:** The full Stage 3 explanation can be verbose. Condensation preserves all numerical values and causal reasoning while eliminating redundancy.

---

## Summary of Information Flow

```
Stage 0:  SMILES → Structural descriptions (blinded, no prediction)
Stage 1:  SHAP + features → Evidence analysis (blinded, no prediction)
Stage 2:  Prediction + attention + paths → Decision analysis (prediction revealed)
Stage 3:  All prior outputs → Integrated explanation
Stage 3.5: Explanation + all evidence → Validation (accept/reject/correct)
Stage 3.6: [If needed] Corrected explanation → Re-integration
Stage 4:  Validated explanation → Condensed 4-paragraph output
```
