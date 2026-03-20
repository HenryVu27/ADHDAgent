"""Unit tests for comparison_runner — mocks all external dependencies."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig


def make_mock_retrieval_response(doc_ids: list[str]):
    from app.models.schemas import FacetCounts, RetrievalResponse, RetrievalResult
    results = [
        RetrievalResult(
            document_id=doc_id,
            document_name="test",
            content="test content",
            score=0.9,
            match_type="hybrid",
            source="",
            tags=[],
            evidence_level="",
            document_type="",
            age_range=[],
            citations=[],
            full_doc={},
        )
        for doc_id in doc_ids
    ]
    return RetrievalResponse(results=results, facets=FacetCounts(), rewritten_query=None)


def test_index_key_mapping():
    """Each config maps to the right use_colbert store key."""
    from eval.runners.comparison_runner import _index_key

    assert _index_key(ABLATION_CONFIGS[0]) is False   # baseline
    assert _index_key(ABLATION_CONFIGS[3]) is True     # +colbert
    assert _index_key(ABLATION_CONFIGS[4]) is True     # full


def test_all_configs_have_valid_index_keys():
    from eval.runners.comparison_runner import _index_key

    valid_keys = {True, False}
    for cfg in ABLATION_CONFIGS:
        assert _index_key(cfg) in valid_keys


@pytest.mark.asyncio
async def test_run_single_config_produces_metrics():
    """_run_config returns a dict with MRR and recall keys."""
    from eval.runners.comparison_runner import _run_config

    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(
        return_value=make_mock_retrieval_response(["doc_001", "doc_002"])
    )

    dataset = [
        {"question": "what helps with homework?", "expected_doc_ids": ["doc_001"], "question_type": "fact_single"},
        {"question": "emotional regulation tips?", "expected_doc_ids": ["doc_999"], "question_type": "reasoning"},
    ]

    result = await _run_config(mock_retriever, dataset, ks=[1, 3, 5])
    assert "mrr" in result["overall"]
    assert "recall@5" in result["overall"]
    assert "fact_single" in result["by_question_type"]
