# FastEmbed cross-encoder reranker
# Scores (query, document) pairs using a trained cross-encoder model.
# Runs locally via ONNX — no API calls, no extra cost.

import asyncio
import logging
import math
import time

from app.models.schemas import RetrievalResult

logger = logging.getLogger(__name__)


class FastEmbedReranker:
    """Cross-encoder reranker using FastEmbed's TextCrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", event_bus=None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)
        self._event_bus = event_bus
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

        t0 = time.monotonic()
        documents = [r.content for r in results]
        # Run CPU-bound ONNX inference off the event loop
        raw_scores = await asyncio.to_thread(
            lambda: list(self._model.rerank(query, documents))
        )

        # BAAI/bge-reranker-base returns raw logits; apply sigmoid to normalize to [0, 1].
        scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
        scored = sorted(
            zip(scores, results),
            key=lambda x: x[0],
            reverse=True,
        )
        reranked = []
        for ce_score, result in scored[:top_k]:
            result.score = ce_score
            reranked.append(result)

        logger.info(
            f"Reranked {len(results)} candidates -> top {len(reranked)} "
            f"(scores: {[f'{s:.4f}' for s, _ in scored[:top_k]]})"
        )
        event_bus = getattr(self, "_event_bus", None)
        if event_bus:
            await event_bus.emit(
                "rag", "rerank", duration_ms=(time.monotonic() - t0) * 1000,
                detail={"input_count": len(results), "output_count": len(reranked)},
            )
        return reranked
