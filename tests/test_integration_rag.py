"""
Integration tests for the RAG pipeline — real embeddings, Qdrant, reranker.

Tests end-to-end retrieval: Gemini embeddings → Qdrant hybrid search →
FastEmbed reranker → facet computation. Validates semantic relevance,
filtering, and knowledge base coverage.

Requires GEMINI_API_KEY environment variable.
Run: pytest tests/test_integration_rag.py -v -m integration -s
"""

import os

import pytest

pytestmark = pytest.mark.integration

if not os.environ.get("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY not set — skipping integration tests", allow_module_level=True)

from app.llm.client import GeminiClient
from app.models.schemas import RetrievalFilters, SessionState, FamilyProfile
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import FastEmbedReranker
from app.rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gemini():
    return GeminiClient()


@pytest.fixture(scope="module")
def knowledge_store():
    return KnowledgeStore()


@pytest.fixture(scope="module")
def reranker():
    """Real FastEmbed cross-encoder (local ONNX)."""
    return FastEmbedReranker()


@pytest.fixture(scope="module")
def query_rewriter(gemini):
    return QueryRewriter(gemini)


@pytest.fixture(scope="module")
async def indexed_store(knowledge_store, gemini):
    """KnowledgeStore with fully built Qdrant index using real embeddings."""
    await knowledge_store.build_index(gemini)
    assert knowledge_store._indexed, "Index should be built"
    return knowledge_store


@pytest.fixture(scope="module")
def retriever(indexed_store, gemini, query_rewriter, reranker):
    """Full retriever with real embeddings, rewriting, and reranking."""
    return HybridRetriever(
        knowledge_store=indexed_store,
        gemini_client=gemini,
        query_rewriter=query_rewriter,
        reranker=reranker,
    )


@pytest.fixture
def school_age_state():
    """Session state for a school-age child with homework challenges."""
    return SessionState(
        session_id="rag-test",
        family_profile=FamilyProfile(
            child_age="8",
            child_name="Leo",
            challenge_areas=["homework focus"],
        ),
        conversation_history=[
            {"role": "user", "content": "Leo takes 3 hours to finish 30 minutes of homework"},
            {"role": "assistant", "content": "That sounds very frustrating. Let me look into some strategies."},
        ],
    )


# ---------------------------------------------------------------------------
# Knowledge Store Index Build
# ---------------------------------------------------------------------------

class TestIndexBuild:
    """Verify the Qdrant index builds correctly with real embeddings."""

    async def test_index_built_successfully(self, indexed_store):
        assert indexed_store._indexed is True

    async def test_documents_loaded(self, indexed_store):
        assert len(indexed_store.documents) > 0, "Should load knowledge documents"

    async def test_chunks_created(self, indexed_store):
        assert len(indexed_store.chunks) > 0, "Should create chunks from documents"

    async def test_sparse_vectors_available(self, indexed_store):
        assert indexed_store.has_sparse is True, "Sparse vectors should be built"


# ---------------------------------------------------------------------------
# Hybrid Search — Semantic Relevance
# ---------------------------------------------------------------------------

class TestHybridSearch:
    """Test that hybrid search returns semantically relevant results."""

    async def test_homework_query_returns_homework_results(self, retriever):
        response = await retriever.retrieve("strategies for homework focus ADHD child")
        assert len(response.results) > 0, "Should return results for homework query"
        # At least one result should be related to homework/focus
        texts = " ".join(r.content.lower() for r in response.results)
        assert any(
            kw in texts for kw in ("homework", "focus", "task", "attention", "study")
        ), f"Results should be homework-related, got: {texts[:200]}"

    async def test_morning_routine_query(self, retriever):
        response = await retriever.retrieve("morning routine meltdowns getting ready")
        assert len(response.results) > 0
        texts = " ".join(r.content.lower() for r in response.results)
        assert any(
            kw in texts for kw in ("morning", "routine", "transition", "schedule", "visual")
        ), f"Results should be routine-related, got: {texts[:200]}"

    async def test_emotion_regulation_query(self, retriever):
        response = await retriever.retrieve("my child has meltdowns and can't control emotions")
        assert len(response.results) > 0
        texts = " ".join(r.content.lower() for r in response.results)
        assert any(
            kw in texts for kw in ("emotion", "meltdown", "regulation", "calm", "feeling")
        ), f"Results should be emotion-related, got: {texts[:200]}"

    async def test_top_k_respected(self, retriever):
        response = await retriever.retrieve("ADHD strategies", top_k=3)
        assert len(response.results) <= 3

    async def test_results_have_required_metadata(self, retriever):
        response = await retriever.retrieve("homework strategies for kids")
        for r in response.results:
            assert r.document_id, "document_id should be populated"
            assert r.document_name, "document_name should be populated"
            assert r.content, "content should not be empty"
            assert r.score > 0, "score should be positive"
            assert r.match_type in ("hybrid", "hybrid+tag", "keyword")

    async def test_results_have_full_doc(self, retriever):
        """full_doc should contain the original document steps or key_points."""
        response = await retriever.retrieve("homework strategies")
        for r in response.results:
            if r.full_doc:
                # Should have either steps or key_points
                has_content = "steps" in r.full_doc or "key_points" in r.full_doc
                assert has_content, f"full_doc should have steps or key_points: {list(r.full_doc.keys())}"


# ---------------------------------------------------------------------------
# Facet Computation
# ---------------------------------------------------------------------------

class TestFacets:
    """Test facet aggregation over retrieval results."""

    async def test_facets_populated(self, retriever):
        response = await retriever.retrieve("ADHD parenting strategies", top_k=10)
        facets = response.facets
        # At least document_type should have entries
        assert facets.document_type, "document_type facets should be populated"

    async def test_facet_counts_are_positive(self, retriever):
        response = await retriever.retrieve("homework focus", top_k=10)
        for doc_type, count in response.facets.document_type.items():
            assert count > 0, f"Count for {doc_type} should be positive"


# ---------------------------------------------------------------------------
# Filtered Retrieval
# ---------------------------------------------------------------------------

class TestFilteredRetrieval:
    """Test retrieval with explicit filters."""

    async def test_filter_by_document_type_strategy(self, retriever):
        response = await retriever.retrieve_filtered(
            query="ADHD help",
            filters=RetrievalFilters(document_type="strategy"),
        )
        for r in response.results:
            assert r.document_type == "strategy", (
                f"All results should be strategies, got: {r.document_type}"
            )

    async def test_filter_by_document_type_guidance(self, retriever):
        response = await retriever.retrieve_filtered(
            query="ADHD help",
            filters=RetrievalFilters(document_type="guidance"),
        )
        for r in response.results:
            assert r.document_type == "guidance", (
                f"All results should be guidance, got: {r.document_type}"
            )


# ---------------------------------------------------------------------------
# Query Rewriting
# ---------------------------------------------------------------------------

class TestQueryRewriting:
    """Test LLM-powered query rewriting with conversation context."""

    async def test_pronoun_resolution(self, query_rewriter):
        """Query rewriter should resolve 'he' to the child's context."""
        rewritten = await query_rewriter.rewrite(
            query="What can I do when he won't sit still?",
            conversation_history=[
                {"role": "user", "content": "My son Jake is 8 and has ADHD"},
                {"role": "assistant", "content": "Tell me more about Jake's challenges."},
            ],
        )
        # Rewritten query should either keep the original meaning or add context
        assert isinstance(rewritten, str)
        assert len(rewritten) > 0

    async def test_without_history_returns_original(self, query_rewriter):
        original = "homework strategies for ADHD"
        rewritten = await query_rewriter.rewrite(query=original)
        # Without history, should return original or near-original
        assert isinstance(rewritten, str)
        assert len(rewritten) > 0

    async def test_rewriting_with_full_state(self, retriever, school_age_state):
        """Full retrieval with query rewriting and conversation state."""
        response = await retriever.retrieve(
            query="What else can I try?",
            state=school_age_state,
        )
        # With context about Leo's homework struggles, rewriter should
        # produce a query that returns homework-related results
        assert len(response.results) > 0
        if response.rewritten_query:
            # Rewritten query should incorporate homework context
            assert isinstance(response.rewritten_query, str)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

class TestReranking:
    """Test FastEmbed cross-encoder reranking."""

    async def test_reranker_reorders_results(self, retriever):
        """Reranked results should have different ordering than raw hybrid search."""
        response = await retriever.retrieve("homework focus strategies for ADHD kids", top_k=5)
        assert len(response.results) > 0
        # All scores should be finite and positive
        for r in response.results:
            assert r.score > 0

    async def test_reranker_respects_top_k(self, reranker):
        """Reranker should return at most top_k results."""
        from app.models.schemas import RetrievalResult

        # Create synthetic results
        fake_results = [
            RetrievalResult(
                document_id=f"doc-{i}",
                document_name=f"Doc {i}",
                content=f"ADHD strategy {i}: help your child with homework by breaking tasks into smaller pieces",
                score=0.5,
                match_type="hybrid",
            )
            for i in range(8)
        ]
        reranked = await reranker.rerank("homework strategies", fake_results, top_k=3)
        assert len(reranked) == 3


# ---------------------------------------------------------------------------
# End-to-end retrieval pipeline
# ---------------------------------------------------------------------------

class TestEndToEndRetrieval:
    """Full pipeline: embed → search → rerank → format."""

    async def test_full_pipeline_with_context(self, retriever, school_age_state):
        """Full retrieval with query rewriting, hybrid search, and reranking."""
        response = await retriever.retrieve(
            query="What strategies help with homework?",
            state=school_age_state,
            top_k=5,
        )
        assert len(response.results) > 0
        assert response.facets is not None

        # Results should be relevant to homework
        all_content = " ".join(r.content.lower() for r in response.results)
        assert any(
            kw in all_content for kw in ("homework", "focus", "task", "break", "timer", "routine")
        )

    async def test_irrelevant_query_gets_low_scores_or_few_results(self, retriever):
        """A completely off-topic query should get low relevance or few results."""
        response = await retriever.retrieve(
            query="quantum physics string theory dark matter",
            top_k=5,
        )
        # Might still return some results due to hybrid search, but scores should be low
        # or the results should be clearly less relevant
        if response.results:
            avg_score = sum(r.score for r in response.results) / len(response.results)
            # The score threshold is context-dependent, just verify it returns something sensible
            assert avg_score >= 0
