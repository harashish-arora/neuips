"""
Agreement Evaluator
===================
Compares two explanations for the same molecule (e.g. model-generated vs human)
using an LLM judge that scores agreement across five dimensions.

Usage:
    python -m explainer.agreement_evaluator \
        --api-keys KEY1 KEY2 \
        --pairs pairs.csv \
        --output agreement_results.json

pairs.csv must have columns: smiles, source_a, explanation_a, source_b, explanation_b
"""

import argparse
import asyncio
import json
import re
import aiohttp
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm.asyncio import tqdm as atqdm

from .config import APIKeyManager, PipelineConfig, gemini_call, load_prompts

DIMENSIONS = [
    "conclusion_agreement",
    "reasoning_alignment",
    "causal_attribution",
    "completeness_overlap",
    "overall_agreement",
]


def _parse_agreement_json(raw: str) -> Dict:
    """Extract the agreement JSON from the LLM response."""
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except json.JSONDecodeError:
        pass
    return {}


async def evaluate_pair(
    row: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
) -> Dict:
    """Evaluate agreement for a single explanation pair."""
    p = prompts["agreement"]
    prompt = p["user"].format(
        smiles=row["smiles"],
        source_a=row["source_a"],
        explanation_a=row["explanation_a"],
        source_b=row["source_b"],
        explanation_b=row["explanation_b"],
    )

    raw, success = await gemini_call(
        session, prompt, p["system"], sem, km,
        model=cfg.gemini_model, temperature=0.1,
    )

    scores = _parse_agreement_json(raw) if success else {}

    result = {"smiles": row["smiles"], "success": success, "raw_response": raw}
    for dim in DIMENSIONS:
        entry = scores.get(dim, {})
        result[f"{dim}_score"] = entry.get("score", None)
        result[f"{dim}_justification"] = entry.get("justification", "")

    return result


async def run_async(
    api_keys: List[str],
    pairs_file: str,
    output_file: str,
    cfg: PipelineConfig | None = None,
):
    if cfg is None:
        cfg = PipelineConfig()

    prompts = load_prompts(cfg.prompts_file)
    df = pd.read_csv(pairs_file)

    required = {"smiles", "source_a", "explanation_a", "source_b", "explanation_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {pairs_file}: {missing}")

    print(f"Evaluating {len(df)} pairs with {len(api_keys)} API key(s)...")

    km = APIKeyManager(api_keys)
    sem = asyncio.Semaphore(10)
    conn = aiohttp.TCPConnector(limit=15)

    results = []
    async with aiohttp.ClientSession(connector=conn) as session:
        with atqdm(total=len(df), desc="Agreement eval") as pbar:
            tasks = []
            for _, row in df.iterrows():
                tasks.append(evaluate_pair(
                    row.to_dict(), prompts, session, sem, km, cfg
                ))

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

    succeeded = [r for r in results if r["success"]]
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_pairs": len(df),
        "successful": len(succeeded),
        "mean_scores": {},
    }
    for dim in DIMENSIONS:
        vals = [r[f"{dim}_score"] for r in succeeded if r[f"{dim}_score"] is not None]
        summary["mean_scores"][dim] = round(sum(vals) / len(vals), 3) if vals else None

    output = {"summary": summary, "results": results}

    Path(output_file).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_file}")
    print(f"  Successful: {len(succeeded)}/{len(df)}")
    for dim, val in summary["mean_scores"].items():
        print(f"  {dim}: {val}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate explanation agreement")
    parser.add_argument("--api-keys", nargs="+", required=True)
    parser.add_argument("--pairs", required=True, help="CSV with explanation pairs")
    parser.add_argument("--output", default="agreement_results.json")
    args = parser.parse_args()

    cfg = PipelineConfig()
    asyncio.run(run_async(args.api_keys, args.pairs, args.output, cfg))


if __name__ == "__main__":
    main()
