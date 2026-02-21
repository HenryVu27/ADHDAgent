# FastEmbed cross-encoder reranker
# Scores (query, document) pairs using a trained cross-encoder model.
# Runs locally via ONNX — no API calls, no extra cost.

import logging

from app.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class FastEmbedReranker:
    """Cross-encoder reranker using FastEmbed's TextCrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)
        logger.info(f"FastEmbed reranker loaded (model={model_name})")

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank results by scoring each (query, doc) pair with a cross-encoder."""
        if not results:
            return results[:top_k]

        documents = [r.content for r in results]
        raw_scores = list(self._model.rerank(query, documents))

        # FastEmbed rerank() returns raw floats in input order for most models.
        scored = sorted(
            zip(raw_scores, results),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = [result for _, result in scored[:top_k]]

        logger.info(
            f"Reranked {len(results)} candidates -> top {len(reranked)} "
            f"(scores: {[f'{s:.4f}' for s, _ in scored[:top_k]]})"
        )
        return reranked
