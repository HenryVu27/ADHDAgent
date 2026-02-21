"""Tests for ConversationAnalyzer — quality analysis with mocked Gemini."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.analyzer import ConversationAnalyzer
from app.agent.session_store import InMemorySessionStore
from app.models.schemas import EnrichedTrace, ToolCallRecord


@pytest.fixture
def store():
    return InMemorySessionStore()


@pytest.fixture
def mock_gemini():
    client = MagicMock()
    client.extract_json = AsyncMock()
    return client


@pytest.fixture
def analyzer(store, mock_gemini):
    return ConversationAnalyzer(session_store=store, gemini_client=mock_gemini)


def _make_trace(session_id="s1", turn=1, tool_calls=None):
    return EnrichedTrace(
        session_id=session_id,
        turn=turn,
        tool_calls=tool_calls or [],
    )


class TestConversationAnalyzer:

    @pytest.mark.asyncio
    async def test_no_issues_detected(self, analyzer, mock_gemini, store):
        mock_gemini.extract_json.return_value = {
            "flags": [],
            "quality_score": 1.0,
            "summary": "No issues detected.",
            "tool_call_assessment": "appropriate",
        }

        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="How do I help with homework?",
            assistant_response="Here are some strategies...",
            enriched_trace=trace,
        )

        assert result is not None
        assert result.quality_score == 1.0
        assert len(result.flags) == 0
        # Verify it was saved
        analyses = store.get_analyses("s1")
        assert len(analyses) == 1

    @pytest.mark.asyncio
    async def test_broken_promise_flag(self, analyzer, mock_gemini, store):
        mock_gemini.extract_json.return_value = {
            "flags": [{
                "flag_type": "broken_promise",
                "severity": "error",
                "description": "Said 'I'll search' but delivered no strategies",
                "evidence": "I'll look for some strategies for you",
            }],
            "quality_score": 0.3,
            "summary": "Broken promise — no content delivered.",
            "tool_call_assessment": "missed",
        }

        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="My kid can't focus on homework",
            assistant_response="I'll look for some strategies for you. Is there anything else?",
            enriched_trace=trace,
        )

        assert result is not None
        assert len(result.flags) == 1
        assert result.flags[0].flag_type == "broken_promise"
        assert result.flags[0].severity == "error"
        assert result.quality_score == 0.3

    @pytest.mark.asyncio
    async def test_missed_tool_call_flag(self, analyzer, mock_gemini, store):
        mock_gemini.extract_json.return_value = {
            "flags": [{
                "flag_type": "missed_tool_call",
                "severity": "warning",
                "description": "Parent described specific challenge but no search was performed",
                "evidence": "he finds it hard to stay focused during homework",
            }],
            "quality_score": 0.5,
            "summary": "Should have searched knowledge base.",
            "tool_call_assessment": "missed",
        }

        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="During homework especially, he finds it hard to stay focused",
            assistant_response="That sounds really challenging. Tell me more about his homework routine.",
            enriched_trace=trace,
        )

        assert result is not None
        assert result.flags[0].flag_type == "missed_tool_call"

    @pytest.mark.asyncio
    async def test_with_tool_calls_in_trace(self, analyzer, mock_gemini, store):
        mock_gemini.extract_json.return_value = {
            "flags": [],
            "quality_score": 0.95,
            "summary": "Good turn with relevant tool use.",
            "tool_call_assessment": "appropriate",
        }

        trace = _make_trace(tool_calls=[
            ToolCallRecord(
                name="search_knowledge_base",
                args={"query": "homework focus strategies"},
                result="[1] Visual Timer Strategy: Use a visual timer...",
            ),
        ])
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="How can I help my kid focus on homework?",
            assistant_response="Here's a great strategy: visual timers...",
            enriched_trace=trace,
        )

        assert result is not None
        assert result.quality_score == 0.95
        assert len(result.flags) == 0

    @pytest.mark.asyncio
    async def test_gemini_failure_returns_none(self, analyzer, mock_gemini):
        mock_gemini.extract_json.side_effect = Exception("API timeout")

        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="test",
            assistant_response="test",
            enriched_trace=trace,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_gemini_returns_non_dict(self, analyzer, mock_gemini):
        mock_gemini.extract_json.return_value = "not a dict"

        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="test",
            assistant_response="test",
            enriched_trace=trace,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_no_gemini_client(self, store):
        analyzer = ConversationAnalyzer(session_store=store, gemini_client=None)
        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="test",
            assistant_response="test",
            enriched_trace=trace,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_flags(self, analyzer, mock_gemini, store):
        mock_gemini.extract_json.return_value = {
            "flags": [
                {"flag_type": "broken_promise", "severity": "error", "description": "d1", "evidence": "e1"},
                {"flag_type": "tone_issue", "severity": "warning", "description": "d2", "evidence": "e2"},
            ],
            "quality_score": 0.2,
            "summary": "Multiple issues.",
            "tool_call_assessment": "missed",
        }

        trace = _make_trace()
        result = await analyzer.analyze_turn(
            session_id="s1", turn=1,
            user_message="test",
            assistant_response="test",
            enriched_trace=trace,
        )

        assert result is not None
        assert len(result.flags) == 2
        assert result.flags[0].flag_type == "broken_promise"
        assert result.flags[1].flag_type == "tone_issue"
