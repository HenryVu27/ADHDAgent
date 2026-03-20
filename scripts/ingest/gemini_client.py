import json
import google.genai as genai
from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiClient:
    """Standalone Gemini Flash client for relevance scoring."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_flash(self, prompt: str) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text

    async def score_relevance(self, docs: list[dict]) -> list[dict]:
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
            'Return JSON array: [{"index": int, "score": int, "reason": str}, ...]'
        )

        raw = await self._call_flash(prompt)
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
