# Hybrid RAG Retriever
# Qdrant dense+sparse with RRF fusion, query term tag boosting, and query rewriting.

import logging
from collections import defaultdict

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

logger = logging.getLogger(__name__)

TAG_BOOST = 0.15


class HybridRetriever:
    # Qdrant hybrid search with keyword fallback

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        gemini_client=None,
        query_rewriter: QueryRewriter | None = None,
    ):
        self._store = knowledge_store
        self._gemini = gemini_client
        self._rewriter = query_rewriter

    # Full retrieval pipeline: rewrite -> hybrid search -> facets -> trim
    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        state: SessionState | None = None,
    ) -> RetrievalResponse:
        top_k = top_k or settings.RAG_TOP_K
        rewritten_query = None

        # Step 1: Query rewriting
        search_query = query
        if (
            self._rewriter
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

        # Step 2: Hybrid search (RRF + tag boosting)
        candidates = await self._hybrid_search(search_query, top_k, filters)

        # Step 3: Compute facets
        facets = self._compute_facets(candidates)

        # Step 4: Trim to top_k
        results = candidates[:top_k]

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
            query_vector = self._gemini.embed(query)
            hybrid_results = self._store.search_hybrid(
                query_vector=query_vector,
                query_text=query,
                top_k=top_k,
                filters=filters,
            )
            if hybrid_results:
                return self._build_results(hybrid_results, query_tags)

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
            chunk_tags = {t.lower() for t in chunk["tags"]}
            matches = query_tags & chunk_tags
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
            chunk_tags = {t.lower() for t in chunk["tags"]}

            score = sum(1.0 for term in query_terms if term in chunk_text)
            tag_matches = query_terms & chunk_tags
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
        )

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
