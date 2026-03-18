"""Minimal standalone Gemini client for dataset generation.

Intentionally does not import from app/ — eval generators must be
self-contained so they can run without the production app installed.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig

from eval.config import GEMINI_API_KEY, GENERATOR_MODEL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class GenClient:
    """Thin Gemini wrapper for the eval generators."""

    def __init__(self, model: str = GENERATOR_MODEL):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._model = model

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    async def json(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> Any:
        last_text = ""
        for attempt in range(1, retries + 1):
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )

            # Log finish_reason for diagnostics
            finish_reason = None
            try:
                finish_reason = response.candidates[0].finish_reason
            except Exception:
                pass

            # Extract text safely — response.text raises if finish_reason is MAX_TOKENS
            try:
                text = response.text or "{}"
            except Exception:
                try:
                    text = response.candidates[0].content.parts[0].text or "{}"
                    logger.warning("Partial response (finish_reason=%s), attempting salvage", finish_reason)
                except Exception:
                    text = "{}"

            last_text = text

            # Try parsing directly
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Salvage: strip markdown fences then retry parse
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass

            # Salvage: find outermost complete JSON object or array
            for start_ch, end_ch in [("{", "}"), ("[", "]")]:
                s = text.find(start_ch)
                if s < 0:
                    continue
                for e in range(len(text), s, -1):
                    if text[e - 1] == end_ch:
                        try:
                            return json.loads(text[s:e])
                        except json.JSONDecodeError:
                            continue

            if attempt < retries:
                logger.warning(
                    "Truncated JSON on attempt %d/%d (finish_reason=%s): %s",
                    attempt, retries, finish_reason, text[:200],
                )
                await asyncio.sleep(0.5 * attempt)
            else:
                logger.warning(
                    "Could not parse JSON after %d attempts (finish_reason=%s): %s",
                    retries, finish_reason, last_text[:200],
                )

        return {}

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = await asyncio.to_thread(
            self._client.models.embed_content,
            model=EMBEDDING_MODEL,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]
