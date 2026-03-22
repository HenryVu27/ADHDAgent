import pytest
from unittest.mock import AsyncMock

from app.agent.response_structurer import structure_response
from app.models.schemas import AgentResponse


@pytest.mark.asyncio
async def test_structure_response_extracts_concerns():
    mock_client = AsyncMock()
    mock_client.extract_json.return_value = {
        "response_text": "Here are some homework strategies...",
        "concerns_addressed": ["homework difficulty"],
        "strategies_referenced": ["structured_homework_time"],
        "follow_up_question": "What does homework time look like right now?",
        "needs_more_info": False,
    }
    result = await structure_response(
        response_text="Here are some homework strategies...",
        user_message="My kid won't do homework",
        gemini_client=mock_client,
    )
    assert isinstance(result, AgentResponse)
    assert len(result.concerns_addressed) > 0
    assert result.response_text == "Here are some homework strategies..."


@pytest.mark.asyncio
async def test_structure_response_fallback_on_error():
    mock_client = AsyncMock()
    mock_client.extract_json.side_effect = Exception("API error")
    result = await structure_response(
        response_text="Here are some strategies...",
        user_message="help with homework",
        gemini_client=mock_client,
    )
    assert isinstance(result, AgentResponse)
    assert result.response_text == "Here are some strategies..."
    assert result.concerns_addressed == []
