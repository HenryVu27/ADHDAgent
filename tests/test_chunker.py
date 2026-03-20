# Tests for RAG chunking strategies

from app.config import settings
from app.rag.chunker import Chunk, NoneChunker


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
