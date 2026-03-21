"""
Gemini LLM client wrapper.

Single interface for all LLM operations: text generation, structured JSON
extraction, and embedding vectors for RAG.
"""

import asyncio
import json
import logging
import math
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)

# Transient errors worth retrying
_RETRYABLE = (ConnectionError, TimeoutError, OSError)
try:
    from google.api_core.exceptions import (
        InternalServerError,
        ResourceExhausted,
        ServiceUnavailable,
        DeadlineExceeded,
    )
    _RETRYABLE = (*_RETRYABLE, ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError)
except ImportError:
    pass
try:
    from google.genai.errors import ClientError as GenAIClientError
    _RETRYABLE = (*_RETRYABLE, GenAIClientError)
except ImportError:
    pass

_retry_policy = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(_RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class GeminiClient:
    """Wraps all Gemini API calls behind a clean interface."""

    def __init__(self):
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.GEMINI_UTILITY_MODEL
        self._embedding_model = settings.GEMINI_EMBEDDING_MODEL

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        model: str | None = None,
        max_output_tokens: int = 2048,
        timeout: float | None = None,
        json_output: bool = False,
    ) -> str:
        """Generate text from a prompt."""
        try:
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if json_output:
                config.response_mime_type = "application/json"
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.generate_content),
                model=model or self._model,
                contents=prompt,
                config=config,
            )
            if timeout is not None:
                response = await asyncio.wait_for(coro, timeout=timeout)
            else:
                response = await coro
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini generate failed: {e}")
            raise

    async def extract_json(
        self,
        prompt: str,
        temperature: float = 0.0,
        model: str | None = None,
        max_output_tokens: int = 1024,
        timeout: float | None = None,
        disable_thinking: bool = False,
    ) -> Any:
        """Generate structured JSON output. Returns parsed JSON."""
        try:
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            )
            if disable_thinking:
                config.thinking_config = ThinkingConfig(thinking_budget=0)
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.generate_content),
                model=model or self._model,
                contents=prompt,
                config=config,
            )
            if timeout is not None:
                response = await asyncio.wait_for(coro, timeout=timeout)
            else:
                response = await coro
            text = response.text or "[]"
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned non-JSON, attempting text parse")
            text = response.text or ""
            # Try object first (most callers expect dicts), then array
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {}
        except Exception as e:
            logger.error(f"Gemini extract_json failed: {e}")
            raise

    async def search_web(
        self,
        query: str,
        timeout: float | None = None,
    ) -> tuple[str, list[dict]]:
        """Perform a grounded web search via Gemini Flash + GoogleSearch.

        Returns (answer_text, sources) where sources is a list of
        {"title": str, "url": str} dicts from grounding metadata.
        """
        try:
            from google.genai import types

            config = GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
                max_output_tokens=1024,
            )
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.generate_content),
                model=self._model,
                contents=query,
                config=config,
            )
            if timeout is not None:
                response = await asyncio.wait_for(coro, timeout=timeout)
            else:
                response = await coro

            # Extract answer text
            answer = response.text or ""

            # Extract grounding sources from response.candidates[0].grounding_metadata.grounding_chunks
            # Each chunk has a .web attribute with .uri and .title fields
            sources: list[dict] = []
            if response.candidates:
                metadata = response.candidates[0].grounding_metadata
                if metadata and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            sources.append({
                                "title": getattr(chunk.web, "title", "") or "",
                                "url": getattr(chunk.web, "uri", "") or "",
                            })

            return answer, sources
        except Exception as e:
            logger.error(f"Gemini search_web failed: {e}")
            raise

    async def embed(self, text: str, timeout: float | None = None) -> list[float]:
        """Get embedding vector for a single text."""
        coro = asyncio.to_thread(
            _retry_policy(self._client.models.embed_content),
            model=self._embedding_model,
            contents=text,
        )
        if timeout is not None:
            result = await asyncio.wait_for(coro, timeout=timeout)
        else:
            result = await coro
        return list(result.embeddings[0].values)

    async def embed_batch(self, texts: list[str], timeout: float | None = None) -> list[list[float]]:
        """Get embedding vectors for a batch of texts.

        Gemini allows at most 100 texts per batchEmbedContents request,
        so we chunk the input and concatenate the results.
        """
        if not texts:
            return []

        MAX_BATCH = 100
        all_embeddings: list[list[float]] = []
        # Per-batch timeout must be generous to accommodate rate-limit retries
        per_batch_timeout = max(timeout or 120, 120)
        num_batches = math.ceil(len(texts) / MAX_BATCH)

        for i, start in enumerate(range(0, len(texts), MAX_BATCH)):
            batch = texts[start : start + MAX_BATCH]
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.embed_content),
                model=self._embedding_model,
                contents=batch,
            )
            result = await asyncio.wait_for(coro, timeout=per_batch_timeout)
            all_embeddings.extend(list(e.values) for e in result.embeddings)
            # Throttle to stay under 3,000 RPM (30 batches of 100/min = 1 every 2s)
            if i < num_batches - 1:
                await asyncio.sleep(2.0)

        return all_embeddings
