"""
Five LLM stages of the explanation pipeline.

Stage 0 – Molecule description (solute & solvent)
Stage 1 – Evidence summary (SHAP, groups, structural features)
Stage 2 – Decision analysis (cross-attention, leaf paths)
Stage 3 – Integration (final explanation)
Stage 3.5 – Validation + optional revision
Stage 4 – Condensation
"""

import json
import asyncio
import re
import numpy as np
import aiohttp
from typing import Dict, Tuple, Optional

from .config import APIKeyManager, PipelineConfig, gemini_call


def _format_shap(shap_dict: Dict, top_n: int = 20) -> str:
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    pos = [f"  {n}: +{v:.4f}" for n, v in items if v > 0]
    neg = [f"  {n}: {v:.4f}" for n, v in items if v < 0]
    out = []
    if pos:
        out.append("POSITIVE CONTRIBUTIONS (increasing solubility):")
        out.extend(pos[:10])
    if neg:
        out.append("\nNEGATIVE CONTRIBUTIONS (decreasing solubility):")
        out.extend(neg[:10])
    return "\n".join(out)


def _group_contributions(shap_dict: Dict) -> Tuple[Dict, str, str]:
    groups = {"Solute": 0.0, "Solvent": 0.0, "Interact": 0.0, "Thermo": 0.0}
    for name, val in shap_dict.items():
        if name.startswith("Solute_"):
            groups["Solute"] += val
        elif name.startswith("Solvent_"):
            groups["Solvent"] += val
        elif name.startswith("Interact_"):
            groups["Interact"] += val
        else:
            groups["Thermo"] += val

    dominant = max(groups.items(), key=lambda x: abs(x[1]))
    total = sum(abs(v) for v in groups.values())
    lines = [f"  {k}: {v:+.4f}" for k, v in groups.items()]
    lines.append(
        f"\n  Dominant group: {dominant[0]} "
        f"({abs(dominant[1]) / total * 100:.1f}% of total signal)"
    )
    return groups, "\n".join(lines), dominant[0]


def _unusual_features(structural: Dict, threshold_file: str) -> str:
    unusual = []
    try:
        with open(threshold_file, "r") as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        return "  (threshold file not found — skipped)"

    for feat, value in structural.items():
        if feat not in thresholds:
            continue
        low, high = thresholds[feat]["low"], thresholds[feat]["high"]
        mean, std = thresholds[feat]["mean"], thresholds[feat]["std"]
        if value < low:
            z = (value - mean) / std if std > 0 else 0
            unusual.append(
                f"  {feat}={value:.4f} (LOW: {z:.2f}σ below mean, range: [{low:.2f}, {high:.2f}])"
            )
        elif value > high:
            z = (value - mean) / std if std > 0 else 0
            unusual.append(
                f"  {feat}={value:.4f} (HIGH: {z:.2f}σ above mean, range: [{low:.2f}, {high:.2f}])"
            )
    return "\n".join(unusual) if unusual else "  No unusual values detected"


def _cross_attention_summary(sample: Dict) -> str:
    attn = np.array(sample["cross_attention_weights"])
    names = sample["council_feature_names"]
    top_idx = np.argsort(attn.flatten())[-10:][::-1]
    items = []
    for idx in top_idx:
        i, j = idx // attn.shape[1], idx % attn.shape[1]
        sf = names[i] if i < len(names) else f"Solute_{i}"
        vf = names[j] if j < len(names) else f"Solvent_{j}"
        items.append(f"  {sf} → {vf}: {attn[i, j]:.4f}")
    return "\n".join(items)


def _top_features_str(shap_dict: Dict, n: int = 10) -> str:
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
    return "\n".join(
        f"  {name}: {val:+.4f} ({'positive' if val > 0 else 'negative'} contribution)"
        for name, val in top
    )


# ── Stage 0 ──────────────────────────────────────────────────────────────────

async def stage0_molecule_descriptions(
    sample: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
) -> Tuple[str, str]:
    """Return (solute_description, solvent_description)."""
    p = prompts["molecule_description"]

    async def _describe(smiles: str) -> str:
        text, _ = await gemini_call(
            session,
            p["user"].format(smiles=smiles),
            p["system"],
            sem, km, cfg.gemini_model,
            temperature=0.2,
        )
        return text

    solute_desc = await _describe(sample["solute"])
    solvent_desc = await _describe(sample["solvent"])
    return solute_desc, solvent_desc


# ── Stage 1 ──────────────────────────────────────────────────────────────────

async def stage1_evidence_summary(
    sample: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
) -> str:
    p = prompts["evidence_summary"]
    shap_str = _format_shap(sample["shap_values"])
    _, group_str, _ = _group_contributions(sample["shap_values"])
    unusual_str = _unusual_features(
        sample["structural_features"], cfg.feature_thresholds_file
    )
    struct_items = sorted(
        sample["structural_features"].items(), key=lambda x: abs(x[1]), reverse=True
    )[:15]
    struct_str = "\n".join(f"  {k}: {v:.4f}" for k, v in struct_items)

    prompt = p["user"].format(
        shap_features=shap_str,
        group_contributions=group_str,
        structural_features=struct_str + "\n\nUNUSUAL VALUES:\n" + unusual_str,
    )
    text, _ = await gemini_call(
        session, prompt, p["system"], sem, km, cfg.gemini_model, temperature=0.3
    )
    await asyncio.sleep(cfg.rate_limit_delay)
    return text


# ── Stage 2 ──────────────────────────────────────────────────────────────────

async def stage2_decision_analysis(
    sample: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
) -> str:
    p = prompts["decision_analysis"]
    attn_str = _cross_attention_summary(sample)
    feat_str = _top_features_str(sample["shap_values"])
    path = sample["leaf_path"]

    prompt = p["user"].format(
        y_pred=sample["y_pred"],
        temperature=sample["temperature"],
        cross_attention_summary=attn_str,
        num_trees=len(path),
        path_stats=f"Mean leaf: {np.mean(path):.1f}, Std: {np.std(path):.1f}",
        top_features=feat_str,
    )
    text, _ = await gemini_call(
        session, prompt, p["system"], sem, km, cfg.gemini_model, temperature=0.3
    )
    await asyncio.sleep(cfg.rate_limit_delay)
    return text


# ── Stage 3 ──────────────────────────────────────────────────────────────────

async def stage3_integration(
    sample: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
    solute_desc: str,
    solvent_desc: str,
    evidence_summary: str,
    decision_analysis: str,
) -> str:
    p = prompts["integration"]
    prompt = p["user"].format(
        solute_smiles=sample["solute"],
        solute_description=solute_desc,
        solvent_smiles=sample["solvent"],
        solvent_description=solvent_desc,
        evidence_summary=evidence_summary,
        decision_analysis=decision_analysis,
        y_pred=sample["y_pred"],
        temperature=sample["temperature"],
    )
    text, _ = await gemini_call(
        session, prompt, p["system"], sem, km, cfg.gemini_model, temperature=0.4
    )
    await asyncio.sleep(cfg.rate_limit_delay)
    return text


# ── Stage 3.5 – Validation + Revision ────────────────────────────────────────

async def stage3_5_validate_and_revise(
    sample: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
    explanation: str,
    solute_desc: str,
    solvent_desc: str,
) -> Tuple[str, Dict]:
    """Returns (final_explanation, validation_dict)."""
    vp = prompts["validation"]
    shap_str = _format_shap(sample["shap_values"], top_n=20)
    _, group_str, _ = _group_contributions(sample["shap_values"])
    attn_str = _cross_attention_summary(sample)
    unusual_str = _unusual_features(
        sample["structural_features"], cfg.feature_thresholds_file
    )
    struct_items = sorted(
        sample["structural_features"].items(), key=lambda x: abs(x[1]), reverse=True
    )[:15]
    struct_str = "\n".join(f"  {k}: {v:.4f}" for k, v in struct_items)
    feat_str = _top_features_str(sample["shap_values"])
    path = sample["leaf_path"]
    tree_stats = (
        f"Number of trees: {len(path)}, "
        f"Mean leaf: {np.mean(path):.1f}, Std: {np.std(path):.1f}"
    )

    prompt = vp["user"].format(
        explanation=explanation,
        solute_smiles=sample["solute"],
        solute_description=solute_desc,
        solvent_smiles=sample["solvent"],
        solvent_description=solvent_desc,
        shap_features=shap_str,
        unusual_features=unusual_str,
        decision_features=feat_str,
        group_contributions=group_str,
        cross_attention_summary=attn_str,
        structural_features=struct_str,
        tree_stats=tree_stats,
        y_pred=sample["y_pred"],
        temperature=sample["temperature"],
    )
    raw_resp, success = await gemini_call(
        session, prompt, vp["system"], sem, km, cfg.gemini_model, temperature=0.1
    )
    await asyncio.sleep(cfg.rate_limit_delay)

    validation = {
        "verdict": "unknown",
        "unsupported_claims": [],
        "supported_points": [],
        "generic_claims_to_remove": [],
        "evidence_specific_points_to_keep": [],
        "correction_instructions": [],
        "needs_revision": False,
        "raw_response": raw_resp,
    }
    if success:
        try:
            m = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                for k in validation:
                    if k != "raw_response" and k in parsed:
                        validation[k] = parsed[k]
        except json.JSONDecodeError:
            pass

    final = explanation
    if validation.get("needs_revision", False):
        rp = prompts["revision"]
        evidence_pts = "\n".join(
            f"  - {x}" for x in validation.get("evidence_specific_points_to_keep", [])
        ) or "  - Keep only claims directly supported by the strongest supplied evidence."
        generic_pts = "\n".join(
            f"  - {x}" for x in validation.get("generic_claims_to_remove", [])
        ) or "  - Remove generic chemistry narration that could apply to many molecules."
        instructions = "\n".join(
            f"  - {x}" for x in validation.get("correction_instructions", [])
        ) or "  - Remove unsupported claims and stay close to the supplied evidence."

        rev_prompt = rp["user"].format(
            solute=sample.get("solute", "unknown"),
            solvent=sample.get("solvent", "unknown"),
            y_pred=sample["y_pred"],
            explanation=explanation,
            evidence_points=evidence_pts,
            generic_points=generic_pts,
            instructions=instructions,
        )
        revised, _ = await gemini_call(
            session, rev_prompt, rp["system"], sem, km, cfg.gemini_model, temperature=0.2
        )
        await asyncio.sleep(cfg.rate_limit_delay)
        final = revised

    return final, validation


# ── Stage 4 ──────────────────────────────────────────────────────────────────

async def stage4_condensation(
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
    explanation: str,
) -> str:
    p = prompts["condensation"]
    prompt = p["user"].format(original_explanation=explanation)
    text, _ = await gemini_call(
        session, prompt, p["system"], sem, km, cfg.gemini_model, temperature=0.2
    )
    return text
