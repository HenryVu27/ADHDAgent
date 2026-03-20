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
