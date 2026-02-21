# Gemini-based cross-encoder reranker
# Scores (query, document) pairs for relevance using the existing Gemini client.
# Zero new dependencies — uses the same LLM already in the stack.

import logging

from app.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)

RERANK_PROMPT = """Rate the relevance of this document to the search query on a scale of 0.0 to 1.0.
Consider: Does the document directly address the query? Is the advice actionable for the situation described?

Query: {query}

Document title: {title}
Document content: {content}

Respond with ONLY a decimal number between 0.0 and 1.0, nothing else."""


class GeminiReranker:
    """Cross-encoder reranker using Gemini to score query-document relevance."""

    def __init__(self, gemini_client):
        self._gemini = gemini_client

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank results by scoring each (query, doc) pair with Gemini."""
        if not results or not self._gemini:
            return results[:top_k]

        scored = []
        for result in results:
            try:
                prompt = RERANK_PROMPT.format(
                    query=query,
                    title=result.document_name,
                    content=result.content[:800],
                )
                response = await self._gemini.generate(
                    prompt, temperature=0.0, model=None,
                )
                score = float(response.strip())
                score = max(0.0, min(1.0, score))
            except (ValueError, Exception) as e:
                logger.debug(f"Rerank score parse failed for '{result.document_name}': {e}")
                score = result.score  # Fall back to original score
            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [result for _, result in scored[:top_k]]

        logger.info(
            f"Reranked {len(results)} candidates -> top {len(reranked)} "
            f"(scores: {[f'{s:.2f}' for s, _ in scored[:top_k]]})"
        )
        return reranked
