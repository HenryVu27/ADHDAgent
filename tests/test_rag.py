# Tests for RAG retrieval pipeline (Qdrant hybrid + keyword fallback)

import pytest

from app.models.schemas import (
    FacetCounts,
    RetrievalFilters,
    RetrievalResponse,
    RetrievalResult,
)
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import HybridRetriever


@pytest.fixture
def knowledge_store():
    return KnowledgeStore()


@pytest.fixture
def retriever(knowledge_store):
    # No Gemini client -> falls back to keyword scoring
    return HybridRetriever(
        knowledge_store=knowledge_store,
        gemini_client=None,
    )


# --- Knowledge Store tests ---

def test_knowledge_store_loads_documents(knowledge_store):
    assert len(knowledge_store.documents) > 0
    assert len(knowledge_store.chunks) > 0


def test_knowledge_store_has_strategies(knowledge_store):
    types = {d.get("document_type") for d in knowledge_store.documents}
    assert "strategy" in types
    strategies = [d for d in knowledge_store.documents if d.get("document_type") == "strategy"]
    assert len(strategies) >= 5


def test_knowledge_store_has_guidance(knowledge_store):
    types = {d.get("document_type") for d in knowledge_store.documents}
    assert "guidance" in types


def test_knowledge_store_has_facts(knowledge_store):
    types = {d.get("document_type") for d in knowledge_store.documents}
    assert "fact" in types


def test_knowledge_store_get_all_topics(knowledge_store):
    topics = knowledge_store.get_all_topics()
    assert "homework" in topics or "homework_support" in topics
    assert "emotional_regulation" in topics or "emotional_dysregulation" in topics


def test_knowledge_store_search_by_tags(knowledge_store):
    results = knowledge_store.search_by_tags(["homework_support"])
    assert len(results) > 0
    for r in results:
        assert "homework_support" in [t.lower() for t in r.get("tags", [])]


# --- Retrieval pipeline tests ---

@pytest.mark.asyncio
async def test_retrieval_returns_response_structure(retriever):
    response = await retriever.retrieve("homework avoidance won't do homework")
    assert isinstance(response, RetrievalResponse)
    assert isinstance(response.results, list)
    assert isinstance(response.facets, FacetCounts)


@pytest.mark.asyncio
async def test_retrieval_returns_results(retriever):
    response = await retriever.retrieve("homework avoidance won't do homework")
    assert len(response.results) > 0
    assert any("homework_support" in r.tags for r in response.results)


@pytest.mark.asyncio
async def test_retrieval_returns_facets(retriever):
    response = await retriever.retrieve("homework strategies for children with ADHD")
    facets = response.facets
    assert len(facets.document_type) > 0 or len(facets.tags) > 0


@pytest.mark.asyncio
async def test_retrieval_with_relevant_query(retriever):
    """Query terms boost matching documents via tag matching."""
    response = await retriever.retrieve("My child won't do homework avoidance")
    assert len(response.results) > 0


@pytest.mark.asyncio
async def test_retrieval_emotional(retriever):
    response = await retriever.retrieve("meltdown tantrum angry emotional regulation")
    assert len(response.results) > 0


@pytest.mark.asyncio
async def test_retrieval_returns_sources(retriever):
    response = await retriever.retrieve("homework strategies")
    for r in response.results:
        assert r.source != ""
        assert r.document_id != ""
        assert r.document_name != ""


@pytest.mark.asyncio
async def test_retrieval_respects_top_k(retriever):
    response = await retriever.retrieve("parenting ADHD strategies", top_k=2)
    assert len(response.results) <= 2


@pytest.mark.asyncio
async def test_keyword_fallback_without_gemini(knowledge_store):
    retriever = HybridRetriever(
        knowledge_store=knowledge_store,
        gemini_client=None,
    )
    response = await retriever.retrieve("homework strategies")
    assert len(response.results) > 0
    assert all(r.match_type == "keyword" for r in response.results)


# --- Facet computation tests ---

def test_facet_counts_computed_correctly():
    results = [
        RetrievalResult(
            document_id="1", document_name="A", content="...", score=0.9,
            document_type="strategy", tags=["homework", "focus"], source="CDC",
            evidence_level="strong", age_range=["school_age"],
        ),
        RetrievalResult(
            document_id="2", document_name="B", content="...", score=0.8,
            document_type="strategy", tags=["homework", "emotion"], source="AAP",
            evidence_level="moderate", age_range=["school_age", "preschool"],
        ),
        RetrievalResult(
            document_id="3", document_name="C", content="...", score=0.7,
            document_type="fact", tags=["adhd_basics"], source="CDC",
            evidence_level="strong", age_range=[],
        ),
    ]
    facets = HybridRetriever._compute_facets(results)
    assert facets.document_type == {"strategy": 2, "fact": 1}
    assert facets.tags["homework"] == 2
    assert facets.evidence_level["strong"] == 2
    assert facets.source["CDC"] == 2
    assert facets.age_range["school_age"] == 2


# --- Filtered retrieval tests ---

@pytest.mark.asyncio
async def test_filtered_retrieval(retriever):
    filters = RetrievalFilters(document_type="strategy")
    response = await retriever.retrieve_filtered(
        query="parenting strategies",
        filters=filters,
    )
    assert isinstance(response, RetrievalResponse)


# --- Query rewriter tests ---

@pytest.mark.asyncio
async def test_query_rewriter_returns_original_without_gemini():
    rewriter = QueryRewriter(gemini_client=None)
    result = await rewriter.rewrite("How do I help with homework?")
    assert result == "How do I help with homework?"


@pytest.mark.asyncio
async def test_query_rewriter_returns_original_without_history():
    rewriter = QueryRewriter(gemini_client=None)
    result = await rewriter.rewrite(
        "How do I help with homework?",
        conversation_history=[],
    )
    assert result == "How do I help with homework?"


# --- Metadata field tests ---

def test_retrieval_result_has_metadata_fields():
    result = RetrievalResult(
        document_id="bpt_overview",
        document_name="Behavioral Parent Training",
        content="...",
        score=0.89,
        match_type="vector+tag",
        source="CDC",
        tags=["behavioral_strategies"],
        evidence_level="strong",
        document_type="strategy",
        age_range=["preschool", "school_age"],
        citations=[{"source_file": "cdc_behavior_therapy.txt", "detail": "BPT section"}],
    )
    assert result.evidence_level == "strong"
    assert result.document_type == "strategy"
    assert len(result.age_range) == 2
    assert len(result.citations) == 1


def test_retrieval_surfaces_evidence_level(knowledge_store):
    retriever = HybridRetriever(knowledge_store=knowledge_store)
    import asyncio
    response = asyncio.run(retriever.retrieve("visual schedule", top_k=3))
    levels = [r.evidence_level for r in response.results if r.evidence_level]
    assert len(levels) > 0
    assert levels[0] in ("strong", "moderate", "emerging")


def test_retrieval_result_has_full_doc(knowledge_store):
    retriever = HybridRetriever(knowledge_store=knowledge_store)
    import asyncio
    response = asyncio.run(retriever.retrieve("homework strategies", top_k=1))
    assert len(response.results) > 0
    result = response.results[0]
    assert result.full_doc  # Should be populated
    assert "name" in result.full_doc


def test_full_doc_excluded_from_serialization():
    result = RetrievalResult(
        document_id="test",
        document_name="Test",
        content="...",
        score=0.9,
        full_doc={"name": "Test", "steps": ["a", "b"]},
    )
    dumped = result.model_dump()
    assert "full_doc" not in dumped


# --- Reranker tests ---

@pytest.mark.asyncio
async def test_reranker_sorts_by_relevance():
    from unittest.mock import MagicMock, patch
    from app.rag.reranker import FastEmbedReranker

    results = [
        RetrievalResult(document_id="1", document_name="A", content="aaa", score=0.5),
        RetrievalResult(document_id="2", document_name="B", content="bbb", score=0.8),
        RetrievalResult(document_id="3", document_name="C", content="ccc", score=0.6),
    ]

    # Mock the cross-encoder to return known scores
    with patch("app.rag.reranker.FastEmbedReranker.__init__", return_value=None):
        reranker = FastEmbedReranker.__new__(FastEmbedReranker)
        mock_model = MagicMock()
        # fastembed rerank returns raw floats in input order
        mock_model.rerank.return_value = [0.9, 0.3, 0.7]
        reranker._model = mock_model

    reranked = await reranker.rerank("test query", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].document_id == "1"  # scored 0.9
    assert reranked[1].document_id == "3"  # scored 0.7
    import math
    assert reranked[0].score == pytest.approx(1 / (1 + math.exp(-0.9)))
    assert reranked[1].score == pytest.approx(1 / (1 + math.exp(-0.7)))


@pytest.mark.asyncio
async def test_reranker_empty_results():
    from unittest.mock import MagicMock, patch
    from app.rag.reranker import FastEmbedReranker

    with patch("app.rag.reranker.FastEmbedReranker.__init__", return_value=None):
        reranker = FastEmbedReranker.__new__(FastEmbedReranker)
        reranker._model = MagicMock()

    reranked = await reranker.rerank("test", [], top_k=3)
    assert reranked == []


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_candidates(knowledge_store):
    """When the reranker raises, the retriever should still return pre-rerank results."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from tests.conftest import MockGeminiClient

    mock_gemini = MockGeminiClient()

    # Build a reranker whose rerank() always explodes
    with patch("app.rag.reranker.FastEmbedReranker.__init__", return_value=None):
        broken_reranker = __import__("app.rag.reranker", fromlist=["FastEmbedReranker"]).FastEmbedReranker.__new__(
            __import__("app.rag.reranker", fromlist=["FastEmbedReranker"]).FastEmbedReranker
        )
        broken_reranker.rerank = AsyncMock(side_effect=RuntimeError("ONNX crash"))

    retriever = HybridRetriever(
        knowledge_store=knowledge_store,
        gemini_client=mock_gemini,
        reranker=broken_reranker,
    )

    # Keyword fallback path (no Qdrant index), so reranker exception is the only concern
    response = await retriever.retrieve("homework strategies")
    assert len(response.results) > 0


@pytest.mark.asyncio
async def test_relevance_threshold_filters_low_scores():
    """Relevance threshold drops results below the configured score floor."""
    from unittest.mock import MagicMock, patch
    from app.rag.reranker import FastEmbedReranker

    results = [
        RetrievalResult(document_id="low", document_name="Low", content="aaa", score=0.5),
        RetrievalResult(document_id="mid", document_name="Mid", content="bbb", score=0.5),
        RetrievalResult(document_id="high", document_name="High", content="ccc", score=0.5),
    ]

    # Mock reranker that assigns scores: -0.5, 0.1, 0.9
    with patch("app.rag.reranker.FastEmbedReranker.__init__", return_value=None):
        reranker = FastEmbedReranker.__new__(FastEmbedReranker)
        mock_model = MagicMock()
        mock_model.rerank.return_value = [-0.5, 0.1, 0.9]
        reranker._model = mock_model

    # Minimal KnowledgeStore with matching chunks
    store = MagicMock()
    store.has_sparse = False
    store.chunks = []

    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=None,
        reranker=reranker,
    )

    # Manually call the pipeline steps to isolate threshold behavior:
    # Simulate candidates that the reranker will re-score
    reranked = await reranker.rerank("test", results, top_k=3)

    # Sigmoid maps raw 0 -> 0.5, so use 0.5 as threshold (decision boundary).
    # Raw -0.5 -> sigmoid ~0.378 (below 0.5), raw 0.1 -> ~0.525, raw 0.9 -> ~0.711
    threshold = 0.5
    filtered = [r for r in reranked if r.score >= threshold]
    assert len(filtered) == 2
    assert all(r.document_id != "low" for r in filtered)


@pytest.mark.asyncio
async def test_query_rewriter_passes_timeout():
    """QueryRewriter passes timeout kwarg to gemini.generate()."""
    from unittest.mock import AsyncMock
    from app.rag.query_rewriter import QueryRewriter
    from app.config import settings

    mock_gemini = AsyncMock()
    mock_gemini.generate = AsyncMock(return_value="rewritten query")

    rewriter = QueryRewriter(gemini_client=mock_gemini)
    await rewriter.rewrite(
        query="help with homework",
        conversation_history=[{"role": "user", "content": "hi"}],
    )

    mock_gemini.generate.assert_called_once()
    call_kwargs = mock_gemini.generate.call_args
    assert call_kwargs.kwargs.get("timeout") == settings.RAG_EMBED_TIMEOUT_S or \
           call_kwargs[1].get("timeout") == settings.RAG_EMBED_TIMEOUT_S


# --- Document index and lookup tests ---

def test_knowledge_store_builds_doc_index(knowledge_store):
    """_doc_index maps document IDs to full document dicts."""
    assert len(knowledge_store._doc_index) == len(knowledge_store.documents)
    for doc in knowledge_store.documents:
        doc_id = doc.get("id", "")
        if doc_id:
            assert doc_id in knowledge_store._doc_index
            assert knowledge_store._doc_index[doc_id] is doc


def test_get_document_by_id_found(knowledge_store):
    """get_document_by_id returns the document for a known ID."""
    first_doc = knowledge_store.documents[0]
    doc_id = first_doc["id"]
    result = knowledge_store.get_document_by_id(doc_id)
    assert result is not None
    assert result["id"] == doc_id
    assert result["name"] == first_doc["name"]


def test_get_document_by_id_not_found(knowledge_store):
    """get_document_by_id returns None for an unknown ID."""
    result = knowledge_store.get_document_by_id("nonexistent_doc_id_xyz")
    assert result is None


def test_get_related_docs(knowledge_store):
    """get_related_docs returns documents matching related_ids."""
    # Find a document with related_ids
    doc_with_related = None
    for doc in knowledge_store.documents:
        if doc.get("related_ids"):
            doc_with_related = doc
            break

    if doc_with_related is None:
        pytest.skip("No documents with related_ids found")

    related = knowledge_store.get_related_docs(doc_with_related["id"])
    assert len(related) > 0
    # All returned docs should have IDs in the original doc's related_ids
    related_id_set = set(doc_with_related["related_ids"])
    for r in related:
        assert r["id"] in related_id_set


def test_get_related_docs_empty(knowledge_store):
    """get_related_docs returns empty list for a doc with no related_ids."""
    # Find a document without related_ids
    doc_without_related = None
    for doc in knowledge_store.documents:
        if not doc.get("related_ids"):
            doc_without_related = doc
            break

    if doc_without_related is None:
        pytest.skip("All documents have related_ids")

    related = knowledge_store.get_related_docs(doc_without_related["id"])
    assert related == []


def test_get_related_docs_nonexistent(knowledge_store):
    """get_related_docs returns empty list for a nonexistent doc ID."""
    related = knowledge_store.get_related_docs("nonexistent_doc_id_xyz")
    assert related == []


# --- skip_rewrite parameter tests ---

@pytest.mark.asyncio
async def test_retrieve_skip_rewrite():
    """skip_rewrite=True bypasses query rewriting even with a rewriter configured."""
    from unittest.mock import AsyncMock
    from app.models.schemas import SessionState

    mock_gemini = AsyncMock()
    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value="rewritten query")

    store = KnowledgeStore()
    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=None,
        query_rewriter=mock_rewriter,
    )

    state = SessionState(
        session_id="test",
        conversation_history=[{"role": "user", "content": "hello"}],
    )

    response = await retriever.retrieve(
        "homework strategies", state=state, skip_rewrite=True,
    )
    # Rewriter should NOT have been called
    mock_rewriter.rewrite.assert_not_called()
    assert response.rewritten_query is None


def test_hybrid_retriever_without_colbert_has_none():
    retriever = HybridRetriever(knowledge_store=KnowledgeStore(), gemini_client=None)
    assert retriever._colbert is None


class TestKnowledgeStoreColbert:
    """Test that search_hybrid correctly includes/excludes ColBERT prefetch."""

    def test_search_hybrid_with_colbert_prefetch_sends_three_prefetches(self):
        """search_hybrid sends 3 Prefetch objects when colbert_prefetch is provided."""
        from unittest.mock import MagicMock
        from qdrant_client.models import Prefetch

        store = KnowledgeStore()
        store._indexed = True
        mock_client = MagicMock()
        mock_client.query_points.return_value.points = []
        store._client = mock_client

        fake_colbert_prefetch = Prefetch(
            query=[[0.1] * 128, [0.2] * 128],
            using="colbert",
            limit=5,
        )
        store.search_hybrid(
            query_vector=[0.1] * 768,
            query_text="homework strategies",
            top_k=5,
            colbert_prefetch=fake_colbert_prefetch,
        )

        call_kwargs = mock_client.query_points.call_args
        prefetches = call_kwargs.kwargs["prefetch"]
        assert len(prefetches) == 3
        assert any(p.using == "colbert" for p in prefetches)

    def test_search_hybrid_without_colbert_prefetch_sends_two_prefetches(self):
        """search_hybrid sends only dense + sparse when no colbert_prefetch."""
        from unittest.mock import MagicMock

        store = KnowledgeStore()
        store._indexed = True
        mock_client = MagicMock()
        mock_client.query_points.return_value.points = []
        store._client = mock_client

        store.search_hybrid(
            query_vector=[0.1] * 768,
            query_text="homework strategies",
            top_k=5,
        )

        call_kwargs = mock_client.query_points.call_args
        prefetches = call_kwargs.kwargs["prefetch"]
        assert len(prefetches) == 2
        assert all(p.using != "colbert" for p in prefetches)


class TestHybridRetrieverColbert:
    """Integration tests for ColBERT prefetch in the full retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_retriever_calls_make_prefetch_and_passes_to_search_hybrid(self):
        """make_prefetch is called and its result passed to search_hybrid when colbert is set."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from qdrant_client.models import Prefetch

        fake_prefetch = Prefetch(query=[[0.1] * 128], using="colbert", limit=5)
        mock_colbert = MagicMock()
        mock_colbert.make_prefetch.return_value = fake_prefetch

        store = KnowledgeStore()
        mock_gemini = MagicMock()
        mock_gemini.embed = AsyncMock(return_value=[0.1] * 768)

        retriever = HybridRetriever(
            knowledge_store=store,
            gemini_client=mock_gemini,
            colbert_index=mock_colbert,
        )

        with patch.object(type(store), "has_sparse", new_callable=lambda: property(lambda self: True)):
            with patch.object(store, "search_hybrid", return_value=[]) as mock_search:
                await retriever.retrieve("homework strategies")

        mock_colbert.make_prefetch.assert_called_once()
        call_kwargs = mock_search.call_args
        passed_prefetch = (
            call_kwargs.kwargs.get("colbert_prefetch")
            or call_kwargs[1].get("colbert_prefetch")
        )
        assert passed_prefetch is fake_prefetch

    @pytest.mark.asyncio
    async def test_retriever_without_colbert_passes_none_prefetch(self):
        """Without colbert_index, search_hybrid receives colbert_prefetch=None."""
        from unittest.mock import AsyncMock, MagicMock, patch

        store = KnowledgeStore()
        mock_gemini = MagicMock()
        mock_gemini.embed = AsyncMock(return_value=[0.1] * 768)

        retriever = HybridRetriever(
            knowledge_store=store,
            gemini_client=mock_gemini,
            colbert_index=None,
        )

        with patch.object(type(store), "has_sparse", new_callable=lambda: property(lambda self: True)):
            with patch.object(store, "search_hybrid", return_value=[]) as mock_search:
                await retriever.retrieve("homework strategies")

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        passed_prefetch = call_kwargs.kwargs.get("colbert_prefetch")
        assert passed_prefetch is None


@pytest.mark.asyncio
async def test_retrieve_without_skip_rewrite():
    """Without skip_rewrite, the rewriter is used when conditions are met."""
    from unittest.mock import AsyncMock
    from app.models.schemas import SessionState

    mock_rewriter = AsyncMock()
    mock_rewriter.rewrite = AsyncMock(return_value="rewritten homework query")

    store = KnowledgeStore()
    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=None,
        query_rewriter=mock_rewriter,
    )

    state = SessionState(
        session_id="test",
        conversation_history=[{"role": "user", "content": "hello"}],
    )

    response = await retriever.retrieve(
        "homework strategies", state=state, skip_rewrite=False,
    )
    # Rewriter should have been called
    mock_rewriter.rewrite.assert_called_once()
    assert response.rewritten_query == "rewritten homework query"


# --- Outcome boost tests ---

from app.models.schemas import Outcome


def _make_result(name: str, score: float, tags: list[str] | None = None) -> RetrievalResult:
    """Helper to create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        document_id=name.lower().replace(" ", "_"),
        document_name=name,
        content=f"Content for {name}",
        score=score,
        tags=tags or [],
    )


class TestOutcomeBoost:

    @pytest.fixture
    def retriever_for_boost(self, knowledge_store):
        return HybridRetriever(knowledge_store=knowledge_store, gemini_client=None)

    def test_positive_outcome_boosts_matching_doc(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer", "homework"]),
            _make_result("Reward Chart System", 0.85, ["rewards", "motivation"]),
        ]
        outcomes = [Outcome(strategy_name="visual timer", signal="positive", turn=1)]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        timer_doc = next(r for r in boosted if "Timer" in r.document_name)
        reward_doc = next(r for r in boosted if "Reward" in r.document_name)
        assert timer_doc.score == pytest.approx(0.90, abs=0.01)  # 0.80 + 0.10
        assert reward_doc.score == pytest.approx(0.85, abs=0.01)  # unchanged

    def test_negative_outcome_penalizes_matching_doc(self, retriever_for_boost):
        candidates = [
            _make_result("Reward Chart System", 0.85, ["rewards", "motivation"]),
        ]
        outcomes = [Outcome(strategy_name="reward chart", signal="negative", turn=1)]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.70, abs=0.01)  # 0.85 - 0.15

    def test_mixed_outcome_no_change(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [Outcome(strategy_name="visual timer", signal="mixed", turn=1)]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.80, abs=0.01)

    def test_boost_capped_at_max(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        # 5 positive outcomes would be 5 * 0.10 = 0.50 uncapped, but cap is 0.30
        outcomes = [
            Outcome(strategy_name="visual timer", signal="positive", turn=i)
            for i in range(5)
        ]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(1.10, abs=0.01)  # 0.80 + 0.30 (capped)

    def test_penalty_capped_at_negative_max(self, retriever_for_boost):
        candidates = [
            _make_result("Reward Chart System", 0.85, ["rewards"]),
        ]
        outcomes = [
            Outcome(strategy_name="reward chart", signal="negative", turn=i)
            for i in range(5)
        ]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.55, abs=0.01)  # 0.85 - 0.30 (capped)

    def test_empty_outcomes_is_noop(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
            _make_result("Reward Chart System", 0.85, ["rewards"]),
        ]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, [])

        assert boosted[0].score == pytest.approx(0.85, abs=0.01)
        assert boosted[1].score == pytest.approx(0.80, abs=0.01)

    def test_no_matching_outcome_is_noop(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [Outcome(strategy_name="completely unrelated strategy", signal="positive", turn=1)]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.80, abs=0.01)

    def test_results_resorted_after_boost(self, retriever_for_boost):
        candidates = [
            _make_result("Reward Chart System", 0.90, ["rewards"]),
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [
            Outcome(strategy_name="reward chart", signal="negative", turn=1),
            Outcome(strategy_name="visual timer", signal="positive", turn=2),
        ]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        # Timer: 0.80 + 0.10 = 0.90, Reward: 0.90 - 0.15 = 0.75
        # Timer should now be first
        assert "Timer" in boosted[0].document_name
        assert "Reward" in boosted[1].document_name

    def test_boost_returns_metadata(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [Outcome(strategy_name="visual timer", signal="positive", turn=1)]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert len(metadata) == 1
        assert metadata[0]["strategy"] == "visual timer"
        assert metadata[0]["document"] == "Visual Timer Strategy"
        assert metadata[0]["boost"] == pytest.approx(0.10, abs=0.01)


from app.models.schemas import SessionState, FamilyProfile


class TestOutcomeBoostInPipeline:

    @pytest.mark.asyncio
    async def test_retrieve_applies_outcome_boost(self, knowledge_store):
        """Outcome boost integrates into the full retrieve() pipeline."""
        retriever = HybridRetriever(knowledge_store=knowledge_store, gemini_client=None)

        # First retrieve without outcomes to get a baseline
        baseline = await retriever.retrieve("homework timer strategies")
        assert len(baseline.results) > 0, "Need keyword results for this test"

        # Now retrieve with a negative outcome for the top result
        top_doc_name = baseline.results[0].document_name
        top_original_score = baseline.results[0].score
        state = SessionState(
            session_id="test",
            outcomes=[Outcome(strategy_name=top_doc_name, signal="negative", turn=1)],
        )
        boosted = await retriever.retrieve("homework timer strategies", state=state)

        # The penalized doc should have a lower score
        penalized = next((r for r in boosted.results if r.document_name == top_doc_name), None)
        assert penalized is not None
        assert penalized.score < top_original_score


from unittest.mock import AsyncMock


class TestLatencyTracking:

    @pytest.mark.asyncio
    async def test_retrieve_emits_latency(self, knowledge_store):
        event_bus = AsyncMock()
        retriever = HybridRetriever(
            knowledge_store=knowledge_store,
            gemini_client=None,
            event_bus=event_bus,
        )

        await retriever.retrieve("homework strategies")

        # EventBus.emit signature: emit(category, event_type, ..., duration_ms=0.0, ...)
        # Our call: emit("rag", "retrieve", duration_ms=..., detail=...)
        calls = [c for c in event_bus.emit.call_args_list if len(c.args) > 1 and c.args[1] == "retrieve"]
        assert len(calls) >= 1
        assert calls[0].kwargs.get("duration_ms", 0) > 0
