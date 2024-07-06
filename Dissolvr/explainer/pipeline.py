"""
Orchestrator: runs every sample through the five LLM stages, saves
intermediate artefacts, and writes a pipeline summary.
"""

import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from tqdm.asyncio import tqdm as atqdm

from .config import PipelineConfig, APIKeyManager, load_prompts
from .sample_selector import SampleSelector
from . import stages


async def _process_sample(
    sample: Dict,
    prompts: Dict,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    km: APIKeyManager,
    cfg: PipelineConfig,
    output_dir: Path,
    pbar: atqdm,
) -> Dict:
    """Run a single sample through all stages and persist results."""
    sample_dir = output_dir / "samples" / f"sample_{sample['index']}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    (sample_dir / "metadata.json").write_text(json.dumps(sample, indent=2))

    # Stage 0
    solute_desc, solvent_desc = await stages.stage0_molecule_descriptions(
        sample, prompts, session, sem, km, cfg
    )
    (sample_dir / "stage0_molecule_descriptions.json").write_text(
        json.dumps(
            {
                "solute": {"smiles": sample["solute"], "description": solute_desc},
                "solvent": {"smiles": sample["solvent"], "description": solvent_desc},
            },
            indent=2,
        )
    )

    # Stage 1
    evidence = await stages.stage1_evidence_summary(
        sample, prompts, session, sem, km, cfg
    )
    (sample_dir / "stage1_evidence_summary.md").write_text(evidence)

    # Stage 2
    decision = await stages.stage2_decision_analysis(
        sample, prompts, session, sem, km, cfg
    )
    (sample_dir / "stage2_decision_analysis.md").write_text(decision)

    # Stage 3
    explanation = await stages.stage3_integration(
        sample, prompts, session, sem, km, cfg,
        solute_desc, solvent_desc, evidence, decision,
    )
    (sample_dir / "stage3_explanation.md").write_text(explanation)

    # Stage 3.5
    final_explanation, validation = await stages.stage3_5_validate_and_revise(
        sample, prompts, session, sem, km, cfg,
        explanation, solute_desc, solvent_desc,
    )
    (sample_dir / "stage3_5_validation.json").write_text(
        json.dumps(validation, indent=2)
    )
    if validation.get("needs_revision"):
        (sample_dir / "stage3_explanation_ORIGINAL.md").write_text(explanation)
        (sample_dir / "stage3_explanation.md").write_text(final_explanation)

    # Stage 4
    condensed = await stages.stage4_condensation(
        prompts, session, sem, km, cfg, final_explanation
    )
    (sample_dir / "stage4_condensed.md").write_text(condensed)

    pbar.update(1)

    return {
        "sample_index": sample["index"],
        "abs_error": sample["abs_error"],
        "validation_verdict": validation.get("verdict", "unknown"),
        "revision_applied": validation.get("needs_revision", False),
        "output_dir": str(sample_dir),
    }


async def run_async(
    api_keys: List[str],
    cfg: PipelineConfig | None = None,
    dry_run: bool = False,
):
    if cfg is None:
        cfg = PipelineConfig()

    prompts = load_prompts(cfg.prompts_file)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Explanation Pipeline")
    print("=" * 70)
    print(f"  Timestamp : {datetime.now().isoformat()}")
    print(f"  Output    : {cfg.output_dir}")
    print(f"  API keys  : {len(api_keys)}")
    print(f"  Prompts   : {cfg.prompts_file}")

    km = APIKeyManager(api_keys)
    selector = SampleSelector(cfg)
    selector.load_models_and_data()
    samples = selector.select_samples()

    if dry_run:
        print(f"\n[DRY RUN] Would process {len(samples['selected'])} samples")
        return samples

    sem = asyncio.Semaphore(10)
    conn = aiohttp.TCPConnector(limit=15)

    async with aiohttp.ClientSession(connector=conn) as session:
        with atqdm(total=len(samples["selected"]), desc="Samples") as pbar:
            tasks = [
                _process_sample(s, prompts, session, sem, km, cfg, output_dir, pbar)
                for s in samples["selected"]
            ]
            raw = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for r in raw:
        if isinstance(r, Exception):
            print(f"  Error: {r}")
        else:
            results.append(r)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_processed": len(results),
        "api_key_status": km.get_status(),
        "samples": results,
    }
    (output_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone — {len(results)} samples processed. See {output_dir}/")
    return summary


def run(api_keys: List[str], cfg: PipelineConfig | None = None, dry_run: bool = False):
    """Synchronous entry point."""
    return asyncio.run(run_async(api_keys, cfg, dry_run))
