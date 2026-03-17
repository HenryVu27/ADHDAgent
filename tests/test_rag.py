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
    assert reranked[0].score == pytest.approx(0.9)
    assert reranked[1].score == pytest.approx(0.7)


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

    # With default threshold 0.0, -0.5 should be dropped
    from app.config import settings
    filtered = [r for r in reranked if r.score >= settings.RAG_RELEVANCE_THRESHOLD]
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


# --- BM25 sparse encoding tests ---

def _store_with_controlled_vocab(sparse_mode: str) -> KnowledgeStore:
    """Return a KnowledgeStore with a hand-crafted vocab and avgdl for unit testing."""
    store = KnowledgeStore(sparse_mode=sparse_mode)
    # Override vocab so "hello" and "world" are always present at known indices
    store._vocab = {"hello": 0, "world": 1, "foo": 2}
    store._avgdl = 5.0  # pretend average doc length is 5 tokens
    return store


def test_tfidf_sparse_doc_uses_raw_counts():
    store = _store_with_controlled_vocab("tfidf")
    vec = store._text_to_sparse_doc("hello hello world")
    hello_pos = vec.indices.index(0)  # index 0 = "hello"
    assert vec.values[hello_pos] == 2.0


def test_bm25_sparse_doc_saturates_tf():
    store = _store_with_controlled_vocab("bm25")
    # Single occurrence
    vec1 = store._text_to_sparse_doc("hello world")
    val1 = vec1.values[vec1.indices.index(0)]

    # Ten occurrences — BM25 saturation should prevent 10x linear growth
    vec10 = store._text_to_sparse_doc(" ".join(["hello"] * 10 + ["world"]))
    val10 = vec10.values[vec10.indices.index(0)]

    assert val10 < val1 * 10, "BM25 TF should saturate — not linear in count"
    assert val10 > val1, "Higher count should still increase score"


def test_bm25_sparse_doc_differs_from_tfidf():
    tfidf_store = _store_with_controlled_vocab("tfidf")
    bm25_store = _store_with_controlled_vocab("bm25")
    text = "hello hello hello world"
    tfidf_vec = tfidf_store._text_to_sparse_doc(text)
    bm25_vec = bm25_store._text_to_sparse_doc(text)
    tfidf_val = tfidf_vec.values[tfidf_vec.indices.index(0)]
    bm25_val = bm25_vec.values[bm25_vec.indices.index(0)]
    # BM25 should produce a different (saturated) value than raw TF=3
    assert bm25_val != tfidf_val


def test_query_sparse_uses_raw_counts_in_bm25_mode():
    store = _store_with_controlled_vocab("bm25")
    vec = store._text_to_sparse_query("hello hello world")
    hello_pos = vec.indices.index(0)
    # Query-side: raw count regardless of sparse_mode
    assert vec.values[hello_pos] == 2.0


def test_collection_name_override():
    store = KnowledgeStore(collection_name="my_collection")
    assert store._collection == "my_collection"


def test_default_collection_name_uses_settings():
    from app.config import settings
    store = KnowledgeStore()
    assert store._collection == settings.QDRANT_COLLECTION


def test_avgdl_computed():
    # Requires knowledge JSON files to be present (standard test environment assumption)
    store = KnowledgeStore()
    assert store._avgdl > 0.0, "Expected avgdl > 0 with real knowledge docs loaded"
