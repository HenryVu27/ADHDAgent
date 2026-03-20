"""
Gemini LLM client wrapper.

Single interface for all LLM operations: text generation, structured JSON
extraction, and embedding vectors for RAG.
"""

import asyncio
import json
import logging
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig
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

_retry_policy = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
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
    ) -> Any:
        """Generate structured JSON output. Returns parsed JSON."""
        try:
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.generate_content),
                model=model or self._model,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                ),
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

        for start in range(0, len(texts), MAX_BATCH):
            batch = texts[start : start + MAX_BATCH]
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.embed_content),
                model=self._embedding_model,
                contents=batch,
            )
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
            all_embeddings.extend(list(e.values) for e in result.embeddings)

        return all_embeddings
