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
        max_tokens: int = 2048,
    ) -> Any:
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
        text = response.text or "{}"
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Salvage: find first complete JSON object or array
            for start_ch, end_ch in [("{", "}"), ("[", "]")]:
                s = text.find(start_ch)
                e = text.rfind(end_ch) + 1
                if s >= 0 and e > s:
                    try:
                        return json.loads(text[s:e])
                    except json.JSONDecodeError:
                        pass
            logger.warning("Could not parse JSON from response: %s", text[:200])
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
