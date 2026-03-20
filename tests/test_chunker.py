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
