# Tests for RAG chunking strategies

from app.config import settings
from app.rag.chunker import Chunk, NoneChunker, RecursiveContextualChunker


def test_chunking_config_defaults():
    assert settings.RAG_CHUNKING_STRATEGY == "none"
    assert settings.RAG_CHUNK_SIZE_TOKENS == 512
    assert settings.RAG_CHUNK_OVERLAP_TOKENS == 64
    assert settings.RAG_SEMANTIC_SIMILARITY_THRESHOLD == 0.75
    assert settings.RAG_CONTEXTUAL_HEADERS is True
    assert settings.RAG_PARENT_DEDUP is True


def test_chunk_dataclass():
    chunk = Chunk(
        chunk_id="doc1__step_0",
        parent_document_id="doc1",
        text="[Doc1] Step one text",
        raw_text="Step one text",
        context_header="[Doc1] ",
        chunk_type="step",
        chunk_index=0,
        metadata={"tags": ["homework"], "age_range": ["school_age"]},
    )
    assert chunk.chunk_id == "doc1__step_0"
    assert chunk.parent_document_id == "doc1"
    assert chunk.chunk_type == "step"
    assert chunk.chunk_index == 0
    assert chunk.metadata["tags"] == ["homework"]


def test_none_chunker_strategy_doc():
    doc = {
        "id": "test_strategy",
        "name": "Test Strategy",
        "description": "A test strategy for ADHD.",
        "document_type": "strategy",
        "tags": ["homework"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "Test Source",
        "citations": [],
        "steps": ["Step one.", "Step two.", "Step three."],
    }
    chunker = NoneChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "test_strategy__full_0"
    assert chunk.parent_document_id == "test_strategy"
    assert chunk.chunk_type == "full"
    assert "Test Strategy" in chunk.text
    assert "Step one." in chunk.text
    assert "Step two." in chunk.text
    assert chunk.context_header == ""
    assert chunk.metadata["tags"] == ["homework"]


def test_none_chunker_key_points_doc():
    doc = {
        "id": "test_fact",
        "name": "Test Fact",
        "description": "A fact about ADHD.",
        "document_type": "fact",
        "tags": ["adhd_basics"],
        "age_range": ["all"],
        "evidence_level": "moderate",
        "source": "Test",
        "citations": [],
        "key_points": ["Point A.", "Point B."],
    }
    chunker = NoneChunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "Point A." in chunks[0].text
    assert "Point B." in chunks[0].text


def test_recursive_chunker_structured_steps():
    doc = {
        "id": "strat1",
        "name": "Test Strategy",
        "description": "Strategy description here.",
        "document_type": "strategy",
        "tags": ["homework"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "Test",
        "citations": [],
        "steps": ["Step one.", "Step two.", "Step three."],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # description + 3 steps = 4 chunks
    assert len(chunks) == 4
    # First chunk is description
    assert chunks[0].chunk_type == "description"
    assert chunks[0].chunk_id == "strat1__description_0"
    assert "Strategy description here." in chunks[0].text
    # Remaining are steps
    for i, c in enumerate(chunks[1:], start=1):
        assert c.chunk_type == "step"
        assert c.chunk_id == f"strat1__step_{i}"
        assert c.parent_document_id == "strat1"
        assert c.metadata["tags"] == ["homework"]
    assert "Step one." in chunks[1].text
    assert "Step three." in chunks[3].text
    # Document name is prefixed for embedding context
    assert "[Test Strategy]" in chunks[1].text


def test_recursive_chunker_structured_key_points():
    doc = {
        "id": "fact1",
        "name": "ADHD Fact",
        "description": "Facts about ADHD.",
        "document_type": "fact",
        "tags": ["adhd_basics"],
        "age_range": ["all"],
        "evidence_level": "moderate",
        "source": "Test",
        "citations": [],
        "key_points": ["Point A.", "Point B."],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # description + 2 key_points = 3 chunks
    assert len(chunks) == 3
    assert chunks[0].chunk_type == "description"
    assert chunks[1].chunk_type == "key_point"
    assert chunks[2].chunk_type == "key_point"
    assert "Point A." in chunks[1].text


def test_recursive_chunker_description_only():
    doc = {
        "id": "desc_only",
        "name": "Desc Doc",
        "description": "Just a description, no steps or key points.",
        "document_type": "guidance",
        "tags": [],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "description"


def test_recursive_chunker_empty_doc():
    doc = {"id": "empty", "name": "", "description": ""}
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # Should still produce at least one chunk (description, even if empty)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "description"


def test_recursive_chunker_sectioned_text():
    doc = {
        "id": "sectioned1",
        "name": "Sectioned Guide",
        "description": "## Overview\nThis is the overview.\n\n## Details\nHere are the details of the guide.\n\n## Summary\nFinal summary.",
        "document_type": "guidance",
        "tags": ["guide"],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    chunker = RecursiveContextualChunker(contextual_headers=False)
    chunks = chunker.chunk(doc)
    # Should split on heading boundaries: 3 sections
    assert len(chunks) == 3
    assert all(c.chunk_type == "text_segment" for c in chunks)
    assert "Overview" in chunks[0].text
    assert "Details" in chunks[1].text
    assert "Summary" in chunks[2].text


def test_recursive_chunker_long_text_fallback():
    # Generate text longer than chunk_size to force recursive splitting
    long_text = ". ".join([f"Sentence number {i} with some extra words" for i in range(200)])
    doc = {
        "id": "longdoc",
        "name": "Long Document",
        "description": long_text,
        "document_type": "guidance",
        "tags": [],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    # Use small chunk size to force splitting
    chunker = RecursiveContextualChunker(chunk_size=200, chunk_overlap=50, contextual_headers=False)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    assert all(c.chunk_type == "text_segment" for c in chunks)
    assert all(c.parent_document_id == "longdoc" for c in chunks)
    # Each chunk should be within size limit (with some tolerance for boundary)
    for c in chunks:
        assert len(c.text) <= 300  # chunk_size + tolerance


def test_recursive_chunker_overlap():
    # Verify overlap between adjacent chunks
    sentences = [f"Unique sentence {i} about topic {i}." for i in range(50)]
    long_text = " ".join(sentences)
    doc = {
        "id": "overlap_test",
        "name": "Overlap Test",
        "description": long_text,
        "document_type": "guidance",
        "tags": [],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
    }
    chunker = RecursiveContextualChunker(chunk_size=200, chunk_overlap=80, contextual_headers=False)
    chunks = chunker.chunk(doc)
    assert len(chunks) > 2
    # Check that adjacent chunks share some text (overlap)
    for i in range(len(chunks) - 1):
        words_current = set(chunks[i].raw_text.split()[-10:])
        words_next = set(chunks[i + 1].raw_text.split()[:10])
        assert words_current & words_next, f"No overlap between chunks {i} and {i+1}"


# --- Contextual headers ---

from unittest.mock import AsyncMock
import asyncio


def test_contextual_headers_applied():
    """Test that contextual headers are prepended when enabled."""
    doc = {
        "id": "hdr_test",
        "name": "Header Test Strategy",
        "description": "Test description.",
        "document_type": "strategy",
        "tags": ["test"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "Test",
        "citations": [],
        "steps": ["Step one.", "Step two."],
    }

    mock_gemini = AsyncMock()
    mock_gemini.generate.return_value = "This chunk is from Header Test Strategy."

    chunker = RecursiveContextualChunker(contextual_headers=True, gemini_client=mock_gemini)
    chunks = chunker.chunk(doc)

    # Before add_contextual_headers is called, headers should be empty
    assert all(c.context_header == "" for c in chunks)

    # Now apply headers
    updated = asyncio.run(chunker.add_contextual_headers(chunks, doc))

    assert len(updated) == 3  # description + 2 steps
    for c in updated:
        assert c.context_header == "This chunk is from Header Test Strategy."
        assert c.text.startswith("This chunk is from Header Test Strategy. ")
    assert mock_gemini.generate.call_count == 3


def test_contextual_headers_disabled():
    """When contextual_headers=False, add_contextual_headers is a no-op."""
    chunks = [
        Chunk(
            chunk_id="x__step_1", parent_document_id="x",
            text="original", raw_text="original", context_header="",
            chunk_type="step", chunk_index=1, metadata={},
        )
    ]
    chunker = RecursiveContextualChunker(contextual_headers=False)
    updated = asyncio.run(chunker.add_contextual_headers(chunks, {}))
    assert updated[0].text == "original"
    assert updated[0].context_header == ""


def test_contextual_headers_graceful_failure():
    """If LLM call fails, chunk is kept without header."""
    doc = {
        "id": "fail_test",
        "name": "Fail Test",
        "description": "Desc.",
        "steps": ["Step."],
        "tags": [], "age_range": [], "evidence_level": "", "source": "", "citations": [],
    }

    mock_gemini = AsyncMock()
    mock_gemini.generate.side_effect = Exception("API error")

    chunker = RecursiveContextualChunker(contextual_headers=True, gemini_client=mock_gemini)
    chunks = chunker.chunk(doc)
    updated = asyncio.run(chunker.add_contextual_headers(chunks, doc))
    assert all(c.context_header == "" for c in updated)
    assert all("API error" not in c.text for c in updated)


# --- SemanticChunker ---

from app.rag.chunker import SemanticChunker
import numpy as np


def test_semantic_chunker_structured_doc():
    """Structured docs use steps/key_points as sentence units."""
    doc = {
        "id": "sem1",
        "name": "Semantic Test",
        "description": "A test strategy.",
        "document_type": "strategy",
        "tags": ["test"],
        "age_range": [],
        "evidence_level": "",
        "source": "",
        "citations": [],
        "steps": ["Step A about homework.", "Step B about homework.", "Step C about emotions."],
    }

    # Mock embeddings: steps A and B are similar, C is different
    mock_gemini = AsyncMock()
    embeddings = [
        [1.0, 0.0, 0.0],  # description
        [0.9, 0.1, 0.0],  # step A (similar to B)
        [0.85, 0.15, 0.0],  # step B (similar to A)
        [0.0, 0.0, 1.0],  # step C (different)
    ]
    mock_gemini.embed_batch = AsyncMock(return_value=embeddings)

    chunker = SemanticChunker(gemini_client=mock_gemini, similarity_threshold=0.5, chunk_size=50)
    chunks = asyncio.run(chunker.chunk_async(doc))

    # Should group A+B together (similar) and C separate, plus description
    # At minimum: boundary between B and C due to low similarity
    assert len(chunks) >= 2
    assert all(c.parent_document_id == "sem1" for c in chunks)
    assert all(c.chunk_type == "text_segment" for c in chunks)


def test_semantic_chunker_finds_boundaries():
    """Verify boundary detection with controlled embeddings."""
    # 5 embeddings: first 3 similar, then a gap, then 2 similar
    embeddings = [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.9, 0.1],
        [0.0, 1.0],  # big shift here
        [0.05, 0.95],
    ]

    chunker = SemanticChunker(similarity_threshold=0.5)
    boundaries = chunker._find_boundaries(embeddings, threshold=0.5)
    # Should find a boundary between index 2 and 3
    assert 3 in boundaries


def test_semantic_chunker_merge_small():
    """Small chunks below target size should be merged."""
    chunker = SemanticChunker(chunk_size=200, similarity_threshold=0.5)
    texts = ["Short A.", "Short B.", "Short C."]
    merged = chunker._merge_small_chunks(texts, target_size=200)
    # All three are tiny, should merge into one
    assert len(merged) == 1
    assert "Short A." in merged[0]
    assert "Short C." in merged[0]


# --- KnowledgeStore integration ---

from app.rag.knowledge_store import KnowledgeStore


def test_knowledge_store_with_none_chunker(monkeypatch):
    """KnowledgeStore with NoneChunker produces same chunk count as documents."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")
    store = KnowledgeStore()
    assert len(store.chunks) == len(store.documents)


def test_knowledge_store_with_recursive_chunker(monkeypatch):
    """KnowledgeStore with recursive chunker produces more chunks than documents."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "recursive_contextual")
    monkeypatch.setattr("app.config.settings.RAG_CONTEXTUAL_HEADERS", False)
    store = KnowledgeStore()
    assert len(store.chunks) > len(store.documents)
    # Each chunk dict should have document_id and document_name
    for chunk in store.chunks:
        assert "document_id" in chunk
        assert "document_name" in chunk
        assert "text" in chunk
        assert "tags" in chunk


def test_knowledge_store_collection_name_varies_by_strategy(monkeypatch):
    """Collection name should differ per strategy."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")
    store_none = KnowledgeStore()

    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "recursive_contextual")
    monkeypatch.setattr("app.config.settings.RAG_CONTEXTUAL_HEADERS", False)
    store_rc = KnowledgeStore()

    assert store_none._collection != store_rc._collection
    assert "recursive_contextual" in store_rc._collection


# --- Retriever integration ---

from app.models.schemas import RetrievalResult
from app.rag.retriever import HybridRetriever


def test_parent_dedup_keeps_highest_score(monkeypatch):
    """Parent dedup should keep the highest-scoring chunk per document."""
    monkeypatch.setattr("app.config.settings.RAG_PARENT_DEDUP", True)
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")

    store = KnowledgeStore()
    retriever = HybridRetriever(knowledge_store=store)

    results = [
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="chunk A",
            score=0.8, chunk_id="doc1__step_1", chunk_type="step",
        ),
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="chunk B",
            score=0.9, chunk_id="doc1__step_2", chunk_type="step",
        ),
        RetrievalResult(
            document_id="doc2", document_name="Doc 2", content="chunk C",
            score=0.7, chunk_id="doc2__step_1", chunk_type="step",
        ),
    ]

    deduped = retriever._deduplicate_by_parent(results)
    assert len(deduped) == 2
    # doc1's highest score (0.9) should win
    doc1_result = next(r for r in deduped if r.document_id == "doc1")
    assert doc1_result.score == 0.9
    assert doc1_result.chunk_id == "doc1__step_2"
    # doc2 appears once
    doc2_result = next(r for r in deduped if r.document_id == "doc2")
    assert doc2_result.score == 0.7


def test_parent_dedup_disabled(monkeypatch):
    """When dedup is disabled, all results pass through."""
    monkeypatch.setattr("app.config.settings.RAG_PARENT_DEDUP", False)
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")

    store = KnowledgeStore()
    retriever = HybridRetriever(knowledge_store=store)

    results = [
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="A", score=0.8,
        ),
        RetrievalResult(
            document_id="doc1", document_name="Doc 1", content="B", score=0.9,
        ),
    ]
    deduped = retriever._deduplicate_by_parent(results)
    assert len(deduped) == 2


def test_chunk_to_result_looks_up_full_doc(monkeypatch):
    """_chunk_to_result should look up full_doc from store, not chunk dict."""
    monkeypatch.setattr("app.config.settings.RAG_CHUNKING_STRATEGY", "none")
    store = KnowledgeStore()
    retriever = HybridRetriever(knowledge_store=store)

    # Pick a known document
    first_doc = store.documents[0]
    chunk = {
        "document_id": first_doc["id"],
        "document_name": first_doc["name"],
        "text": "some chunk text",
        "tags": first_doc.get("tags", []),
        "source": first_doc.get("source", ""),
        "evidence_level": first_doc.get("evidence_level", ""),
        "document_type": first_doc.get("document_type", ""),
        "age_range": first_doc.get("age_range", []),
        "citations": first_doc.get("citations", []),
        "chunk_id": f"{first_doc['id']}__full_0",
        "chunk_type": "full",
    }
    result = retriever._chunk_to_result(chunk, 0.5, "keyword")
    assert result.full_doc == first_doc
    assert result.chunk_id == f"{first_doc['id']}__full_0"
