from unittest.mock import MagicMock, patch

import pytest


import numpy as np


def make_mock_model(n_tokens: int = 3, dim: int = 128):
    """Return a mock LateInteractionTextEmbedding that yields numpy token matrices.

    Uses np.array to match real FastEmbed output — ensures .tolist() conversion
    is tested rather than the trivially-correct list() on plain Python lists.
    """
    model = MagicMock()
    model.embed.return_value = iter([np.array([[0.1] * dim] * n_tokens)])
    model.query_embed.return_value = iter([np.array([[0.2] * dim] * 2)])
    return model


@patch("app.rag.colbert_index.LateInteractionTextEmbedding")
def test_embed_chunks_returns_one_matrix_per_chunk(mock_cls):
    mock_cls.return_value = make_mock_model(n_tokens=3)
    from app.rag.colbert_index import ColBERTIndex
    index = ColBERTIndex()
    chunks = [{"text": "hello world"}, {"text": "foo bar"}]
    # Reset mock to return 2 numpy matrices
    index._model.embed.return_value = iter([
        np.array([[0.1] * 128] * 3),
        np.array([[0.2] * 128] * 5),
    ])
    result = index.embed_chunks(chunks)
    assert len(result) == 2
    assert len(result[0]) == 3       # 3 tokens
    assert len(result[0][0]) == 128
    # Values must be plain Python floats, not numpy scalars
    assert isinstance(result[0][0][0], float)


@patch("app.rag.colbert_index.LateInteractionTextEmbedding")
def test_make_prefetch_returns_prefetch_object(mock_cls):
    mock_cls.return_value = make_mock_model()
    from app.rag.colbert_index import ColBERTIndex
    from qdrant_client.models import Prefetch
    index = ColBERTIndex()
    index._model.query_embed.return_value = iter([np.array([[0.2] * 128] * 2)])
    prefetch = index.make_prefetch("what strategies help with homework", limit=10)
    assert isinstance(prefetch, Prefetch)
    # Verify query vector elements are plain Python floats, not numpy scalars
    assert isinstance(prefetch.query[0][0], float)
