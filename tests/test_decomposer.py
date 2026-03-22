import pytest
from unittest.mock import AsyncMock

from app.agent.decomposer import decompose_message
from app.models.schemas import DecomposedMessage


@pytest.mark.asyncio
async def test_single_concern_message():
    mock_client = AsyncMock()
    mock_client.extract_json.return_value = {
        "concerns": [
            {"description": "homework struggles", "intent": "strategy_request", "search_query": "homework strategies ADHD"}
        ],
        "is_multi_concern": False,
    }
    result = await decompose_message("My kid won't do homework", mock_client)
    assert isinstance(result, DecomposedMessage)
    assert len(result.concerns) == 1
    assert not result.is_multi_concern


@pytest.mark.asyncio
async def test_multi_concern_message():
    mock_client = AsyncMock()
    mock_client.extract_json.return_value = {
        "concerns": [
            {"description": "homework refusal", "intent": "strategy_request", "search_query": "homework strategies"},
            {"description": "bedtime meltdowns", "intent": "strategy_request", "search_query": "bedtime routine ADHD"},
        ],
        "is_multi_concern": True,
    }
    result = await decompose_message(
        "My kid won't do homework AND has meltdowns at bedtime",
        mock_client,
    )
    assert result.is_multi_concern
    assert len(result.concerns) == 2


@pytest.mark.asyncio
async def test_greeting_skips_decomposition():
    result = await decompose_message("Hi there!", None)
    assert len(result.concerns) == 1
    assert result.concerns[0].intent == "greeting"
    assert not result.is_multi_concern


@pytest.mark.asyncio
async def test_decomposer_fallback_on_error():
    mock_client = AsyncMock()
    mock_client.extract_json.side_effect = Exception("API error")
    result = await decompose_message("Help with homework and bedtime", mock_client)
    assert len(result.concerns) == 1
    assert not result.is_multi_concern
