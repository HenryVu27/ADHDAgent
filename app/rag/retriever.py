from __future__ import annotations

# Hybrid RAG Retriever
# Qdrant dense+sparse with RRF fusion, query term tag boosting, and query rewriting.

import asyncio
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.rag.colbert_index import ColBERTIndex

from app.config import settings
from app.models.schemas import (
    FacetCounts,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalResult,
    SessionState,
)
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import FastEmbedReranker

logger = logging.getLogger(__name__)

TAG_BOOST = 0.15


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors (numpy-accelerated)."""
    va, vb = np.asarray(a), np.asarray(b)
    norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class HybridRetriever:
    # Qdrant hybrid search with keyword fallback

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        gemini_client=None,
        query_rewriter: QueryRewriter | None = None,
        reranker: FastEmbedReranker | None = None,
        colbert_index: ColBERTIndex | None = None,
    ):
        self._store = knowledge_store
        self._gemini = gemini_client
        self._rewriter = query_rewriter
        self._reranker = reranker
        self._colbert = colbert_index
        # Embedding-based query result cache: avoids redundant Qdrant searches
        # when the agent rephrases a query it already searched for.
        self._query_cache: list[tuple[list[float], list[RetrievalResult], float]] = []

    # Full retrieval pipeline: rewrite -> hybrid search -> facets -> trim
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        state: SessionState | None = None,
        skip_rewrite: bool = False,
    ) -> RetrievalResponse:
        top_k = top_k or settings.RAG_TOP_K
        rewritten_query = None

        # Step 1: Query rewriting
        search_query = query
        if (
            not skip_rewrite
            and self._rewriter
            and state
            and settings.RAG_USE_QUERY_REWRITE
            and state.conversation_history
        ):
            rewritten = await self._rewriter.rewrite(
                query=query,
                conversation_history=state.conversation_history,
                family_profile=state.family_profile.model_dump() if state.family_profile else None,
            )
            if rewritten != query:
                rewritten_query = rewritten
                search_query = rewritten
                logger.info("[rag] rewrite: %.70s -> %.70s", query, search_query)

        # Step 2: Hybrid search (RRF + tag boosting)
        fetch_k = (settings.RAG_RERANK_CANDIDATES if self._reranker else top_k)
        candidates = await self._hybrid_search(search_query, fetch_k, filters)

        # Step 3: Rerank (optional, behind config flag)
        if self._reranker and len(candidates) > top_k:
            try:
                candidates = await self._reranker.rerank(search_query, candidates, top_k)
            except Exception as e:
                logger.warning("Reranker failed, using pre-rerank order: %s", e)

        # Step 3b: Relevance threshold (only with reranker — RRF scores aren't calibrated)
        if self._reranker:
            pre_filter = len(candidates)
            candidates = [r for r in candidates if r.score >= settings.RAG_RELEVANCE_THRESHOLD]
            if pre_filter > len(candidates):
                logger.info("Relevance threshold %.2f filtered %d -> %d results",
                            settings.RAG_RELEVANCE_THRESHOLD, pre_filter, len(candidates))

        # Step 4: Compute facets
        facets = self._compute_facets(candidates)

        # Step 5: Trim to top_k
        results = candidates[:top_k]

        logger.info(
            "[rag] %d results: %s",
            len(results),
            ", ".join(f"{r.document_name}({r.score:.2f})" for r in results),
        )

        return RetrievalResponse(
            results=results,
            facets=facets,
            rewritten_query=rewritten_query,
        )

    # Faceted follow-up search with explicit filters
    async def retrieve_filtered(
        self,
        query: str,
        filters: RetrievalFilters,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        return await self.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
        )

    # Qdrant hybrid search with keyword fallback
    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalResult]:
        # Use query terms for tag boosting
        query_tags = set(query.lower().split())

        # Qdrant hybrid: dense + sparse + RRF
        if self._store.has_sparse and self._gemini:
            query_vector = await self._gemini.embed(query, timeout=settings.RAG_EMBED_TIMEOUT_S)

            # Check embedding cache — return cached results for semantically similar queries
            cached = self._cache_lookup(query_vector)
            if cached is not None:
                logger.info("Query cache hit (%d cached results) for: %.60s", len(cached), query)
                return cached[:top_k]

            colbert_prefetch = None
            if self._colbert is not None:
                colbert_prefetch = await asyncio.to_thread(
                    self._colbert.make_prefetch, query, top_k
                )

            hybrid_results = self._store.search_hybrid(
                query_vector=query_vector,
                query_text=query,
                top_k=top_k,
                filters=filters,
                colbert_prefetch=colbert_prefetch,
            )
            if hybrid_results:
                results = self._build_results(hybrid_results, query_tags)
                self._cache_store(query_vector, results)
                return results

        # Fallback: keyword scoring
        return self._keyword_fallback(query, top_k)

    # Build results from Qdrant output with tag boosting
    def _build_results(
        self,
        qdrant_results: list[tuple[int, float, dict]],
        query_tags: set[str],
    ) -> list[RetrievalResult]:
        results = []
        for chunk_idx, score, _payload in qdrant_results:
            chunk = self._store.chunks[chunk_idx]

            boost = 0.0
            chunk_tag_words = set()
            for t in chunk["tags"]:
                chunk_tag_words.update(t.lower().replace("_", " ").split())
            matches = query_tags & chunk_tag_words
            if matches:
                boost = TAG_BOOST * len(matches)

            match_type = "hybrid+tag" if boost > 0 else "hybrid"
            results.append(self._chunk_to_result(chunk, score + boost, match_type))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # Keyword scoring when Qdrant is unavailable
    def _keyword_fallback(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        results = []
        for chunk in self._store.chunks:
            chunk_text = chunk["text"].lower()
            chunk_tag_words = set()
            for t in chunk["tags"]:
                chunk_tag_words.update(t.lower().replace("_", " ").split())

            score = sum(1.0 for term in query_terms if term in chunk_text)
            tag_matches = query_terms & chunk_tag_words
            score += 2.0 * len(tag_matches)

            if score > 0:
                results.append(self._chunk_to_result(chunk, score, "keyword"))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # Convert chunk dict to RetrievalResult
    def _chunk_to_result(self, chunk: dict, score: float, match_type: str) -> RetrievalResult:
        return RetrievalResult(
            document_id=chunk["document_id"],
            document_name=chunk["document_name"],
            content=chunk["text"],
            score=score,
            match_type=match_type,
            source=chunk.get("source", ""),
            tags=chunk.get("tags", []),
            evidence_level=chunk.get("evidence_level", ""),
            document_type=chunk.get("document_type", ""),
            age_range=chunk.get("age_range", []),
            citations=chunk.get("citations", []),
            full_doc=chunk.get("full_doc", {}),
        )

    # --- Embedding-based query cache ---

    def _cache_lookup(self, query_vector: list[float]) -> list[RetrievalResult] | None:
        """Return cached results if a semantically similar query was recently searched.

        Cached vectors are pre-normalized, so similarity is a single dot product.
        """
        now = time.time()
        ttl = settings.RAG_QUERY_CACHE_TTL_S
        threshold = settings.RAG_QUERY_CACHE_SIMILARITY

        # Evict expired entries
        self._query_cache = [
            entry for entry in self._query_cache if now - entry[2] < ttl
        ]
        if not self._query_cache:
            return None

        # Normalize query vector once, then dot-product against cached normals
        qv = np.asarray(query_vector)
        qn = np.linalg.norm(qv)
        if qn == 0:
            return None
        qv_norm = qv / qn

        for cached_norm, cached_results, _ts in self._query_cache:
            sim = float(np.dot(qv_norm, cached_norm))
            if sim >= threshold:
                return cached_results
        return None

    def _cache_store(self, query_vector: list[float], results: list[RetrievalResult]) -> None:
        """Cache pre-normalized query embedding and its results."""
        v = np.asarray(query_vector)
        norm = np.linalg.norm(v)
        if norm == 0:
            return
        self._query_cache.append((v / norm, results, time.time()))
        max_size = settings.RAG_QUERY_CACHE_MAX_SIZE
        if len(self._query_cache) > max_size:
            self._query_cache = self._query_cache[-max_size:]

    # --- Facets ---

    # Aggregate facet counts over retrieval results
    @staticmethod
    def _compute_facets(results: list[RetrievalResult]) -> FacetCounts:
        doc_type_counts: dict[str, int] = defaultdict(int)
        tag_counts: dict[str, int] = defaultdict(int)
        age_counts: dict[str, int] = defaultdict(int)
        evidence_counts: dict[str, int] = defaultdict(int)
        source_counts: dict[str, int] = defaultdict(int)

        for r in results:
            if r.document_type:
                doc_type_counts[r.document_type] += 1
            for tag in r.tags:
                tag_counts[tag] += 1
            for age in r.age_range:
                age_counts[age] += 1
            if r.evidence_level:
                evidence_counts[r.evidence_level] += 1
            if r.source:
                source_counts[r.source] += 1

        return FacetCounts(
            document_type=dict(doc_type_counts),
            tags=dict(tag_counts),
            age_range=dict(age_counts),
            evidence_level=dict(evidence_counts),
            source=dict(source_counts),
        )
