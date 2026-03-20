import json
import logging
import google.genai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class GeminiClient:
    """Standalone Gemini Flash client for relevance scoring."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_flash(self, prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text or ""

    def _parse_json_response(self, raw: str) -> list[dict]:
        """Parse JSON from Gemini response, handling markdown fences and edge cases."""
        text = raw.strip()
        if not text:
            return []
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        if not text:
            return []
        return json.loads(text)

    async def score_relevance(self, docs: list[dict], max_retries: int = 2) -> list[dict]:
        """Score a batch of documents for ADHD parenting relevance (0-10)."""
        if not docs:
            return []

        doc_lines = []
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")[:500]
            tags = ", ".join(doc.get("tags", []))
            doc_lines.append(f"[{i}] Title: {title} | Abstract: {abstract} | Tags: {tags}")

        prompt = (
            "Rate each document's relevance to an ADHD parenting coaching chatbot (0-10):\n"
            "- 9-10: Directly about ADHD parenting strategies, child behavior management, or family support\n"
            "- 7-8: About ADHD in children/adolescents with practical implications for parents\n"
            "- 5-6: About ADHD generally (neuroscience, adult ADHD, pharmacology) or general parenting\n"
            "- 3-4: Tangentially related (general child psychology, education theory)\n"
            "- 0-2: Not relevant\n\n"
            "Documents:\n" + "\n".join(doc_lines) + "\n\n"
            "Return ONLY a JSON array, no other text: "
            '[{"index": int, "score": int, "reason": str}, ...]'
        )

        for attempt in range(max_retries + 1):
            try:
                raw = await self._call_flash(prompt)
                return self._parse_json_response(raw)
            except (json.JSONDecodeError, Exception) as e:
                if attempt < max_retries:
                    logger.warning(f"Gemini JSON parse failed (attempt {attempt + 1}), retrying: {e}")
                else:
                    logger.warning(f"Gemini scoring failed after {max_retries + 1} attempts, keeping all {len(docs)} docs: {e}")
                    # Return max scores so all docs are kept when scoring fails
                    return [{"index": i, "score": 10, "reason": "scoring failed"} for i in range(1, len(docs) + 1)]
