"""JudgeBase — shared infrastructure for all eval judges.

Uses GenClient from eval/generators/llm.py to stay decoupled from app/.
Provides semaphore-based concurrency control and error handling.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from eval.config import EVAL_JUDGE_MODEL
from eval.generators.llm import GenClient

logger = logging.getLogger(__name__)


class JudgeBase:
    """Base class for all eval judges."""

    def __init__(self, max_concurrency: int = 5):
        self._client = GenClient(model=EVAL_JUDGE_MODEL)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def judge_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a judge prompt and return parsed JSON. Returns {} on failure."""
        async with self._semaphore:
            try:
                result = await self._client.json(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return result if isinstance(result, dict) else {}
            except Exception as e:
                logger.warning("Judge call failed: %s", e)
                return {}


CONVERSATIONS_DIR = Path(__file__).parent.parent / "data" / "conversations"


def load_conversations(directory: Path | None = None) -> list[dict]:
    """Load all conversation JSON files from a directory."""
    d = directory or CONVERSATIONS_DIR
    if not d.exists():
        return []
    conversations = []
    for f in sorted(d.glob("*.json")):
        try:
            conversations.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", f, e)
    return conversations
