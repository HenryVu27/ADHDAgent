import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eval.judges.base import JudgeBase


class ConcreteJudge(JudgeBase):
    """Minimal concrete subclass for testing the base class."""
    pass


@pytest.fixture
def mock_gen_client():
    client = MagicMock()
    client.json = AsyncMock(return_value={"score": 5})
    return client


class TestJudgeBase:

    def test_init_default_concurrency(self):
        judge = ConcreteJudge()
        assert judge._semaphore._value == 5

    def test_init_custom_concurrency(self):
        judge = ConcreteJudge(max_concurrency=10)
        assert judge._semaphore._value == 10

    @pytest.mark.asyncio
    async def test_judge_json_calls_gen_client(self, mock_gen_client):
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ConcreteJudge()
        result = await judge.judge_json("test prompt")
        assert result == {"score": 5}
        mock_gen_client.json.assert_called_once()

    @pytest.mark.asyncio
    async def test_judge_json_respects_semaphore(self, mock_gen_client):
        """Verify semaphore limits concurrency."""
        call_count = 0
        max_concurrent = 0

        async def slow_json(*args, **kwargs):
            nonlocal call_count, max_concurrent
            call_count += 1
            current = call_count
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            call_count -= 1
            return {"score": 1}

        mock_gen_client.json = slow_json

        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ConcreteJudge(max_concurrency=2)

        tasks = [judge.judge_json("prompt") for _ in range(5)]
        await asyncio.gather(*tasks)
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_judge_json_returns_empty_on_failure(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(side_effect=Exception("API error"))
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ConcreteJudge()
        result = await judge.judge_json("test prompt")
        assert result == {}


from eval.judges.response_judge import ResponseJudge


class TestResponseJudge:

    @pytest.mark.asyncio
    async def test_score_turn_parses_valid_response(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(return_value={
            "helpfulness": {"score": 4, "rationale": "Addresses the issue"},
            "accuracy": {"score": 5, "rationale": "Evidence-based"},
            "empathy": {"score": 3, "rationale": "Could validate more"},
            "boundary_compliance": {"score": 5, "rationale": "Stays in scope"},
            "groundedness": {"score": 4, "rationale": "Based on retrieved docs"},
            "actionability": {"score": 4, "rationale": "Concrete steps given"},
        })
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ResponseJudge()

        result = await judge.score_turn(
            user_message="My kid can't focus on homework",
            assistant_response="Here are some strategies based on research...",
            tool_calls=[],
            conversation_history=[],
        )
        assert result["scores"]["helpfulness"] == 4
        assert result["scores"]["accuracy"] == 5
        assert result["overall"] == pytest.approx(4.17, abs=0.1)
        assert "rationales" in result

    @pytest.mark.asyncio
    async def test_score_turn_handles_judge_failure(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(return_value={})
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ResponseJudge()

        result = await judge.score_turn(
            user_message="test",
            assistant_response="test",
            tool_calls=[],
            conversation_history=[],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_score_turn_with_tool_calls(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(return_value={
            "helpfulness": {"score": 5, "rationale": "r"},
            "accuracy": {"score": 5, "rationale": "r"},
            "empathy": {"score": 5, "rationale": "r"},
            "boundary_compliance": {"score": 5, "rationale": "r"},
            "groundedness": {"score": 5, "rationale": "r"},
            "actionability": {"score": 5, "rationale": "r"},
        })
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ResponseJudge()

        result = await judge.score_turn(
            user_message="How do I help with homework?",
            assistant_response="Based on the visual timer strategy...",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "homework"}, "result": "Visual timer..."}],
            conversation_history=[],
        )
        assert result["scores"]["groundedness"] == 5
        # Verify the prompt included tool call info
        call_args = mock_gen_client.json.call_args
        assert "search_knowledge_base" in call_args[1].get("prompt", call_args[0][0] if call_args[0] else "")
