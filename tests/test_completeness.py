import pytest
from unittest.mock import AsyncMock

from app.agent.completeness import check_completeness
from app.models.schemas import Concern


@pytest.mark.asyncio
async def test_all_concerns_addressed():
    mock_client = AsyncMock()
    mock_client.extract_json.return_value = {
        "addressed": ["homework struggles", "bedtime meltdowns"],
        "missed": [],
    }
    result = await check_completeness(
        response_text="For homework, try timers. For bedtime, try visual schedules.",
        concerns=[
            Concern(description="homework struggles", intent="strategy_request"),
            Concern(description="bedtime meltdowns", intent="strategy_request"),
        ],
        gemini_client=mock_client,
    )
    assert result.all_addressed
    assert len(result.missed) == 0


@pytest.mark.asyncio
async def test_missed_concern_detected():
    mock_client = AsyncMock()
    mock_client.extract_json.return_value = {
        "addressed": ["homework struggles"],
        "missed": ["bedtime meltdowns"],
    }
    result = await check_completeness(
        response_text="For homework, try timers.",
        concerns=[
            Concern(description="homework struggles", intent="strategy_request"),
            Concern(description="bedtime meltdowns", intent="strategy_request"),
        ],
        gemini_client=mock_client,
    )
    assert not result.all_addressed
    assert "bedtime meltdowns" in result.missed


@pytest.mark.asyncio
async def test_single_concern_skips_check():
    result = await check_completeness(
        response_text="Here are some strategies...",
        concerns=[Concern(description="homework", intent="strategy_request")],
        gemini_client=None,
    )
    assert result.all_addressed
