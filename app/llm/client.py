"""
Gemini LLM client wrapper.

Single interface for all LLM operations: text generation, structured JSON
extraction, and embedding vectors for RAG.
"""

import json
import logging
from typing import Any

from google import genai

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Wraps all Gemini API calls behind a clean interface."""

    def __init__(self):
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self._model = settings.GEMINI_MODEL
        self._embedding_model = settings.GEMINI_EMBEDDING_MODEL

    async def generate(self, prompt: str, temperature: float = 0.7, model: str | None = None) -> str:
        """Generate text from a prompt."""
        try:
            response = self._client.models.generate_content(
                model=model or self._model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini generate failed: {e}")
            raise

    async def extract_json(self, prompt: str, temperature: float = 0.0, model: str | None = None) -> Any:
        """Generate structured JSON output. Returns parsed JSON."""
        try:
            response = self._client.models.generate_content(
                model=model or self._model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            text = response.text or "[]"
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Gemini returned non-JSON, attempting text parse")
            # Try to extract JSON from the response
            text = response.text or ""
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return []
        except Exception as e:
            logger.error(f"Gemini extract_json failed: {e}")
            raise

    def embed(self, text: str) -> list[float]:
        """Get embedding vector for a single text."""
        result = self._client.models.embed_content(
            model=self._embedding_model,
            contents=text,
        )
        return list(result.embeddings[0].values)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embedding vectors for a batch of texts."""
        if not texts:
            return []
        # Gemini embedding API supports batch via multiple contents
        result = self._client.models.embed_content(
            model=self._embedding_model,
            contents=texts,
        )
        return [list(e.values) for e in result.embeddings]
