from scripts.ingest.gemini_client import GeminiClient


class RelevanceFilter:
    """Filters documents by Gemini Flash relevance score."""

    def __init__(self, client: GeminiClient, threshold: int = 6, batch_size: int = 20):
        self._client = client
        self._threshold = threshold
        self._batch_size = batch_size

    async def filter(self, docs: list[dict]) -> list[dict]:
        """Return only documents scoring >= threshold."""
        if not docs:
            return []

        kept = []
        for i in range(0, len(docs), self._batch_size):
            batch = docs[i:i + self._batch_size]
            batch_for_scoring = [
                {
                    "title": d.get("name", ""),
                    "abstract": d.get("description", ""),
                    "tags": d.get("tags", []),
                }
                for d in batch
            ]
            scores = await self._client.score_relevance(batch_for_scoring)

            score_map = {s["index"]: s["score"] for s in scores}
            for j, doc in enumerate(batch, 1):
                if score_map.get(j, 0) >= self._threshold:
                    kept.append(doc)

        return kept
