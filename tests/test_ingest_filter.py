import pytest
from unittest.mock import AsyncMock, patch
from scripts.ingest.gemini_client import GeminiClient
from scripts.ingest.pipeline.filter import RelevanceFilter


@pytest.mark.asyncio
async def test_gemini_client_score_batch():
    client = GeminiClient(api_key="test-key")
    mock_response = '[{"index": 1, "score": 8, "reason": "relevant"}]'
    with patch.object(client, "_call_flash", new_callable=AsyncMock, return_value=mock_response):
        results = await client.score_relevance([{
            "title": "ADHD Parent Training",
            "abstract": "A study on parent training for ADHD.",
            "tags": ["ADHD", "parenting"],
        }])
    assert len(results) == 1
    assert results[0]["score"] == 8


@pytest.mark.asyncio
async def test_relevance_filter_keeps_above_threshold():
    docs = [
        {"id": "doc1", "name": "High relevance", "description": "ADHD parenting"},
        {"id": "doc2", "name": "Low relevance", "description": "Quantum physics"},
    ]
    mock_client = AsyncMock()
    mock_client.score_relevance.return_value = [
        {"index": 1, "score": 8, "reason": "relevant"},
        {"index": 2, "score": 2, "reason": "not relevant"},
    ]
    filt = RelevanceFilter(client=mock_client, threshold=6, batch_size=20)
    kept = await filt.filter(docs)
    assert len(kept) == 1
    assert kept[0]["id"] == "doc1"


@pytest.mark.asyncio
async def test_relevance_filter_empty_input():
    mock_client = AsyncMock()
    filt = RelevanceFilter(client=mock_client, threshold=6, batch_size=20)
    kept = await filt.filter([])
    assert kept == []
    mock_client.score_relevance.assert_not_called()
