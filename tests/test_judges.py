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


from eval.judges.tool_use_judge import ToolUseJudge, run_heuristics


class TestToolUseHeuristics:

    def test_unused_result_detected(self):
        flags = run_heuristics(
            user_message="How do I help with homework?",
            assistant_response="That sounds tough. Tell me more.",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "homework"}, "result": "Visual timer strategy: use a timer to break homework into 10-min chunks"}],
        )
        assert any(f["type"] == "low_result_utilization" for f in flags)

    def test_no_flags_when_result_used(self):
        flags = run_heuristics(
            user_message="How do I help with homework?",
            assistant_response="A great approach is the visual timer strategy: break homework into 10-minute chunks.",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "homework"}, "result": "Visual timer strategy: use a timer to break homework into 10-min chunks"}],
        )
        assert not any(f["type"] == "low_result_utilization" for f in flags)

    def test_missed_search_detected(self):
        flags = run_heuristics(
            user_message="My kid has terrible meltdowns during homework every single evening",
            assistant_response="That sounds challenging.",
            tool_calls=[],
        )
        assert any(f["type"] == "likely_missed_search" for f in flags)

    def test_no_missed_search_for_greetings(self):
        flags = run_heuristics(
            user_message="Hi there!",
            assistant_response="Hello! How can I help?",
            tool_calls=[],
        )
        assert not any(f["type"] == "likely_missed_search" for f in flags)


class TestToolUseJudge:

    @pytest.mark.asyncio
    async def test_evaluate_turn(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(return_value={
            "tool_verdicts": [
                {"name": "search_knowledge_base", "verdict": "appropriate", "argument_quality": 4, "result_utilization": 5}
            ],
            "missed_tools": [],
        })
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ToolUseJudge()

        result = await judge.evaluate_turn(
            user_message="How do I help with homework?",
            assistant_response="Based on the visual timer strategy...",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "homework"}, "result": "Visual timer..."}],
            conversation_history=[],
        )
        assert result is not None
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["argument_accuracy"] == 4

    @pytest.mark.asyncio
    async def test_evaluate_turn_no_tools(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(return_value={
            "tool_verdicts": [],
            "missed_tools": [],
        })
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ToolUseJudge()

        result = await judge.evaluate_turn(
            user_message="Hi!",
            assistant_response="Hello!",
            tool_calls=[],
            conversation_history=[],
        )
        assert result is not None
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0


from eval.judges.coherence_judge import CoherenceJudge


class TestCoherenceJudge:

    @pytest.mark.asyncio
    async def test_score_conversation(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(return_value={
            "scores": {
                "progressive_profiling": 4,
                "repetition_avoidance": 5,
                "follow_up": 2,
                "topic_management": 4,
            },
            "rationale": {
                "progressive_profiling": "Learned name and age by turn 3",
                "follow_up": "Never checked back on timer strategy",
            },
            "notable_moments": [
                {"turn": 4, "type": "missed_follow_up", "detail": "Timer recommended, no check-in"}
            ],
        })
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = CoherenceJudge()

        turns = [
            {"turn": i, "user_message": f"msg {i}", "assistant_response": f"resp {i}", "tool_calls": []}
            for i in range(1, 8)
        ]

        result = await judge.score_conversation(turns=turns, family_profile={})
        assert result is not None
        assert result["scores"]["follow_up"] == 2
        assert result["overall"] == pytest.approx(3.75)
        assert len(result["notable_moments"]) == 1

    @pytest.mark.asyncio
    async def test_skips_short_conversations(self, mock_gen_client):
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = CoherenceJudge()

        turns = [
            {"turn": i, "user_message": f"msg {i}", "assistant_response": f"resp {i}", "tool_calls": []}
            for i in range(1, 4)  # Only 3 turns
        ]

        result = await judge.score_conversation(turns=turns, family_profile={})
        assert result is None


from eval.judges.pairwise_judge import PairwiseJudge


class TestPairwiseJudge:

    @pytest.mark.asyncio
    async def test_compare_turn_consistent(self, mock_gen_client):
        """When both orderings agree, use the result."""
        mock_gen_client.json = AsyncMock(side_effect=[
            # First call: A/B order
            {"winner": "A", "confidence": "high", "rationale": "A is better",
             "dimension_winners": {"helpfulness": "A", "empathy": "A"}},
            # Second call: B/A order (A is now B)
            {"winner": "B", "confidence": "high", "rationale": "B is better",
             "dimension_winners": {"helpfulness": "B", "empathy": "B"}},
        ])
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = PairwiseJudge()

        result = await judge.compare_turn(
            user_message="How do I help?",
            response_a="Strategy A response",
            response_b="Strategy B response",
        )
        assert result is not None
        assert result["winner"] == "A"  # Both agree A is better

    @pytest.mark.asyncio
    async def test_compare_turn_inconsistent_becomes_tie(self, mock_gen_client):
        """When orderings disagree, it's a tie."""
        mock_gen_client.json = AsyncMock(side_effect=[
            {"winner": "A", "confidence": "medium", "rationale": "A is better",
             "dimension_winners": {}},
            # Second call: still says A (which is response_b now) — disagreement
            {"winner": "A", "confidence": "medium", "rationale": "A is better",
             "dimension_winners": {}},
        ])
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = PairwiseJudge()

        result = await judge.compare_turn(
            user_message="How do I help?",
            response_a="Response A",
            response_b="Response B",
        )
        assert result is not None
        assert result["winner"] == "tie"
