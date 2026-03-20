# Tests for RAG chunking strategies

from app.config import settings


from app.rag.chunker import Chunk


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
