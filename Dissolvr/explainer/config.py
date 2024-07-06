"""
Configuration and API key management for the explanation pipeline.
"""

import json
import os
import asyncio
import aiohttp
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class PipelineConfig:
    """Configuration for the explanation pipeline."""

    test_file: str = "data/test.csv"
    store_dir: str = "feature_store"
    model_dir: str = "model"
    cgboost_dir: str = "cgboost_explanations"
    output_dir: str = "enhanced_explanations"
    transformer_path: str = "transformer.pth"
    feature_thresholds_file: str = "threshold_analysis/feature_thresholds.json"
    prompts_file: str = os.path.join(os.path.dirname(__file__), "prompts.json")

    num_samples: int = 30

    gemini_model: str = "gemini-3.1-pro-preview"
    max_retries: int = 3
    retry_delay: float = 2.0
    rate_limit_delay: float = 1.0
    timeout: float = 300.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_prompts(path: str) -> Dict:
    """Load prompt templates from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class APIKeyManager:
    """Manages multiple Gemini API keys with rotation on rate-limit."""

    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise ValueError("At least one API key must be provided")
        self.api_keys = api_keys
        self.current_index = 0
        self.exhausted_keys: set = set()
        self.key_usage_count = {i: 0 for i in range(len(api_keys))}

    def get_current_key(self) -> str:
        return self.api_keys[self.current_index]

    def rotate_key(self, mark_exhausted: bool = True) -> bool:
        if mark_exhausted:
            self.exhausted_keys.add(self.current_index)
            print(f"  API key {self.current_index + 1}/{len(self.api_keys)} exhausted")
        for _ in range(len(self.api_keys)):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            if self.current_index not in self.exhausted_keys:
                return True
        return False

    def record_usage(self):
        self.key_usage_count[self.current_index] += 1

    def get_status(self) -> Dict:
        return {
            "total_keys": len(self.api_keys),
            "current_index": self.current_index,
            "exhausted_count": len(self.exhausted_keys),
            "usage_counts": self.key_usage_count,
        }

    def has_available_keys(self) -> bool:
        return len(self.exhausted_keys) < len(self.api_keys)


async def gemini_call(
    session: aiohttp.ClientSession,
    prompt: str,
    system_prompt: Optional[str],
    sem: asyncio.Semaphore,
    key_manager: APIKeyManager,
    model: str,
    temperature: float = 0.3,
    max_retries: int = 4,
) -> Tuple[str, bool]:
    """Async Gemini REST call with retry and key rotation."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 8192},
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    last_err = None
    for attempt in range(max_retries * len(key_manager.api_keys)):
        api_key = key_manager.get_current_key()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )

        async with sem:
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status in (429, 503):
                        await asyncio.sleep(2)
                        key_manager.rotate_key(mark_exhausted=False)
                        continue

                    body = await resp.json()

                    if resp.status != 200:
                        err_msg = body.get("error", {}).get("message", str(body))
                        if "quota" in err_msg.lower() or "exhausted" in err_msg.lower():
                            if not key_manager.rotate_key(mark_exhausted=True):
                                return "ERROR: All keys exhausted", False
                            continue
                        return f"ERROR: {err_msg}", False

                    candidates = body.get("candidates", [])
                    if not candidates:
                        feedback = body.get("promptFeedback", {})
                        if feedback.get("blockReason"):
                            return f"ERROR: Blocked ({feedback['blockReason']})", False
                        return json.dumps({"error": "no candidates"}), False

                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts)
                    key_manager.record_usage()
                    return text.strip(), True

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = str(e)
                await asyncio.sleep(2**attempt)

    return f"ERROR: Max retries exceeded: {last_err}", False
