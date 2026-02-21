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
    from app.rag.reranker import GeminiReranker
    from unittest.mock import AsyncMock

    mock_gemini = AsyncMock()
    # Return descending scores so doc3 should come first
    mock_gemini.generate = AsyncMock(side_effect=["0.9", "0.3", "0.7"])

    reranker = GeminiReranker(gemini_client=mock_gemini)
    results = [
        RetrievalResult(document_id="1", document_name="A", content="aaa", score=0.5),
        RetrievalResult(document_id="2", document_name="B", content="bbb", score=0.8),
        RetrievalResult(document_id="3", document_name="C", content="ccc", score=0.6),
    ]
    reranked = await reranker.rerank("test query", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].document_id == "1"  # scored 0.9
    assert reranked[1].document_id == "3"  # scored 0.7


@pytest.mark.asyncio
async def test_reranker_handles_parse_failure():
    from app.rag.reranker import GeminiReranker
    from unittest.mock import AsyncMock

    mock_gemini = AsyncMock()
    mock_gemini.generate = AsyncMock(side_effect=["not_a_number", "0.5"])

    reranker = GeminiReranker(gemini_client=mock_gemini)
    results = [
        RetrievalResult(document_id="1", document_name="A", content="aaa", score=0.3),
        RetrievalResult(document_id="2", document_name="B", content="bbb", score=0.8),
    ]
    reranked = await reranker.rerank("test", results, top_k=2)
    assert len(reranked) == 2


@pytest.mark.asyncio
async def test_reranker_empty_results():
    from app.rag.reranker import GeminiReranker
    from unittest.mock import AsyncMock

    reranker = GeminiReranker(gemini_client=AsyncMock())
    reranked = await reranker.rerank("test", [], top_k=3)
    assert reranked == []
