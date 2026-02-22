"""Tests for MemoryManager — summary, fact extraction, episodic memory."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from app.agent.memory import MemoryManager
from app.agent.session_store import InMemorySessionStore
from app.models.schemas import EpisodicMemory, SessionSummary


def _make_store_with_messages(session_id="s1", turn_count=5):
    """Create a store with some conversation history."""
    store = InMemorySessionStore()
    for t in range(1, turn_count + 1):
        store.increment_turn(session_id)
        store.add_message(session_id, "user", f"User message turn {t}", turn=t)
        store.add_message(session_id, "assistant", f"Assistant response turn {t}", turn=t)
    return store


def _make_gemini_mock(generate_return="Summary text.", extract_json_return=None):
    """Create a mock GeminiClient."""
    mock = AsyncMock()
    mock.generate = AsyncMock(return_value=generate_return)
    mock.extract_json = AsyncMock(return_value=extract_json_return or {})
    return mock


class TestSummaryInterval:

    @pytest.mark.asyncio
    async def test_summary_triggered_at_interval(self):
        store = _make_store_with_messages(turn_count=5)
        gemini = _make_gemini_mock(generate_return="The parent discussed homework challenges.")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="This is turn five with enough content for processing",
            assistant_response="Got it.",
        )

        # Summary should have been saved (turn 5 % 5 == 0)
        summary = store.get_latest_summary("s1")
        assert summary is not None
        assert summary.covers_through_turn == 5
        assert "homework" in summary.summary

    @pytest.mark.asyncio
    async def test_summary_not_triggered_off_interval(self):
        store = _make_store_with_messages(turn_count=3)
        gemini = _make_gemini_mock()
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=3,
            user_message="Short",
            assistant_response="Got it.",
        )

        # Turn 3 % 5 != 0, no summary
        assert store.get_latest_summary("s1") is None

    @pytest.mark.asyncio
    async def test_summary_skipped_without_gemini(self):
        store = _make_store_with_messages(turn_count=5)
        mm = MemoryManager(session_store=store, gemini_client=None)

        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="Some message here",
            assistant_response="Response.",
        )

        assert store.get_latest_summary("s1") is None

    @pytest.mark.asyncio
    async def test_summary_includes_prior(self):
        """Second summary should include reference to prior summary in prompt."""
        store = _make_store_with_messages(turn_count=10)
        gemini = _make_gemini_mock(generate_return="Extended summary.")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        # Seed a prior summary
        store.save_summary("s1", SessionSummary(summary="First summary.", covers_through_turn=5))

        await mm.post_turn_tasks(
            session_id="s1", turn=10,
            user_message="Another long message that should be included in context",
            assistant_response="Response.",
        )

        # The generate prompt should have referenced the prior summary
        call_args = gemini.generate.call_args
        assert "First summary." in call_args[0][0]


class TestFactExtraction:

    @pytest.mark.asyncio
    async def test_extracts_facts_from_long_message(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(extract_json_return={"child_name": "Kai", "child_age": "7"})
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        long_msg = "My son Kai is 7 years old and he struggles with homework every evening"
        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message=long_msg,
            assistant_response="I understand.",
        )

        profile = store.get("s1").family_profile
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    @pytest.mark.asyncio
    async def test_skips_short_messages(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock()
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="Yes",  # < 40 chars
            assistant_response="Ok.",
        )

        # extract_json should NOT have been called
        gemini.extract_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_filters_invalid_fields(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(extract_json_return={
            "child_name": "Kai",
            "invalid_field": "should be ignored",
            "password": "secret",
        })
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        long_msg = "My son Kai is having trouble at school with paying attention in class"
        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message=long_msg,
            assistant_response="I see.",
        )

        profile = store.get("s1").family_profile
        assert profile.child_name == "Kai"

    @pytest.mark.asyncio
    async def test_skips_empty_extraction(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(extract_json_return={})
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        long_msg = "I appreciate the advice you gave me, it was really helpful thank you"
        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message=long_msg,
            assistant_response="You're welcome!",
        )

        # Profile should remain untouched
        profile = store.get("s1").family_profile
        assert profile.child_name is None


class TestFactExtractionPreFilter:

    @pytest.mark.asyncio
    async def test_fact_extraction_skips_non_informative_messages(self):
        """Messages with no profile-related content should skip the LLM call."""
        store = InMemorySessionStore()
        mock_gemini = AsyncMock()
        memory = MemoryManager(session_store=store, gemini_client=mock_gemini)

        # These are all >40 chars but contain no profile-related vocabulary
        await memory._extract_facts("test-session", "That sounds really helpful, thank you so much for explaining that to me", 3)
        mock_gemini.extract_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_fact_extraction_proceeds_for_informative_messages(self):
        """Messages with profile-related content should trigger the LLM call."""
        store = InMemorySessionStore()
        mock_gemini = AsyncMock()
        mock_gemini.extract_json = AsyncMock(return_value={})
        memory = MemoryManager(session_store=store, gemini_client=mock_gemini)

        await memory._extract_facts("test-session", "My son is 7 years old and was diagnosed with ADHD last year", 3)
        mock_gemini.extract_json.assert_called_once()

    def test_might_contain_facts_positive(self):
        """Messages mentioning children/age/challenges should pass the filter."""
        assert MemoryManager._might_contain_facts("My daughter is struggling with homework every night")
        assert MemoryManager._might_contain_facts("He was diagnosed with ADHD at age 7")
        assert MemoryManager._might_contain_facts("We tried the visual timer strategy last week")

    def test_might_contain_facts_negative(self):
        """Generic acknowledgments should fail the filter."""
        assert not MemoryManager._might_contain_facts("That sounds really helpful, thank you so much")
        assert not MemoryManager._might_contain_facts("I appreciate you sharing that with me today")
        assert not MemoryManager._might_contain_facts("Ok I will try that and let you know how it goes")


class TestEpisodicMemory:

    @pytest.mark.asyncio
    async def test_episode_created_on_track_outcome(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(generate_return="hopeful")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        tool_calls = [{
            "name": "track_outcome",
            "args": {
                "strategy_name": "visual timer",
                "outcome": "positive",
                "notes": "worked great for homework",
            },
        }]

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="The visual timer helped so much with homework!",
            assistant_response="That's wonderful!",
            tool_calls=tool_calls,
        )

        episodes = store.get_recent_episodes("s1")
        assert len(episodes) == 1
        assert episodes[0].event_type == "outcome_reported"
        assert "visual timer" in episodes[0].summary
        assert episodes[0].outcome == "positive"
        assert "visual timer" in episodes[0].strategies_involved
        assert episodes[0].emotional_context == "hopeful"

    @pytest.mark.asyncio
    async def test_no_episode_without_track_outcome(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock()
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        tool_calls = [{
            "name": "search_knowledge_base",
            "args": {"query": "homework strategies"},
        }]

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="I need help with homework routines for my child",
            assistant_response="Here are some strategies.",
            tool_calls=tool_calls,
        )

        episodes = store.get_recent_episodes("s1")
        assert len(episodes) == 0

    @pytest.mark.asyncio
    async def test_multiple_outcomes_create_multiple_episodes(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock()
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        tool_calls = [
            {
                "name": "track_outcome",
                "args": {"strategy_name": "visual timer", "outcome": "positive"},
            },
            {
                "name": "track_outcome",
                "args": {"strategy_name": "reward chart", "outcome": "negative"},
            },
        ]

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="The timer worked but the reward chart didn't help",
            assistant_response="Thanks for sharing.",
            tool_calls=tool_calls,
        )

        episodes = store.get_recent_episodes("s1")
        assert len(episodes) == 2


class TestEmotionInferenceLLM:

    @pytest.mark.asyncio
    async def test_llm_called_with_message(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(generate_return="frustrated")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        result = await mm._infer_emotion("s1", "I'm so frustrated with bedtime")

        assert result == "frustrated"
        prompt_arg = gemini.generate.call_args[0][0]
        assert "frustrated with bedtime" in prompt_arg

    @pytest.mark.asyncio
    async def test_includes_conversation_context(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        store.add_message("s1", "user", "My son has ADHD", turn=1)
        store.add_message("s1", "assistant", "Tell me more about him", turn=1)
        gemini = _make_gemini_mock(generate_return="anxious")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm._infer_emotion("s1", "I'm worried about his grades")

        prompt_arg = gemini.generate.call_args[0][0]
        assert "My son has ADHD" in prompt_arg
        assert "Tell me more about him" in prompt_arg

    @pytest.mark.asyncio
    async def test_invalid_output_defaults_empty(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(generate_return="something unexpected")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        result = await mm._infer_emotion("s1", "We tried the timer")

        assert result == ""

    @pytest.mark.asyncio
    async def test_neutral_returns_empty(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock(generate_return="neutral")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        result = await mm._infer_emotion("s1", "We tried the timer yesterday")

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_gemini_returns_empty(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        mm = MemoryManager(session_store=store, gemini_client=None)

        result = await mm._infer_emotion("s1", "I'm so frustrated")

        assert result == ""


class TestErrorResilience:

    @pytest.mark.asyncio
    async def test_summary_error_does_not_crash(self):
        store = _make_store_with_messages(turn_count=5)
        gemini = _make_gemini_mock()
        gemini.generate = AsyncMock(side_effect=Exception("API error"))
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        # Should not raise — errors are caught in post_turn_tasks
        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="Some long message that should trigger fact extraction here",
            assistant_response="Response.",
        )

    @pytest.mark.asyncio
    async def test_fact_extraction_error_does_not_crash(self):
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock()
        gemini.extract_json = AsyncMock(side_effect=Exception("Parse error"))
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="My son Kai is 7 and he has trouble with homework every day",
            assistant_response="I understand.",
        )

        # Profile should not be updated since extraction failed
        profile = store.get("s1").family_profile
        assert profile.child_name is None

    @pytest.mark.asyncio
    async def test_summary_timeout_raises(self):
        """TimeoutError from generate() should propagate (caught by gather)."""
        store = _make_store_with_messages(turn_count=5)
        gemini = _make_gemini_mock()
        gemini.generate = AsyncMock(side_effect=asyncio.TimeoutError())
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        # post_turn_tasks catches exceptions via gather(return_exceptions=True)
        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="Some long message that should trigger fact extraction here",
            assistant_response="Response.",
        )

        # Summary should NOT have been saved
        assert store.get_latest_summary("s1") is None

    @pytest.mark.asyncio
    async def test_fact_extraction_timeout_does_not_crash(self):
        """TimeoutError from extract_json() should not crash post_turn_tasks."""
        store = InMemorySessionStore()
        store.increment_turn("s1")
        gemini = _make_gemini_mock()
        gemini.extract_json = AsyncMock(side_effect=asyncio.TimeoutError())
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="My son Kai is 7 and he has trouble with homework every day",
            assistant_response="I understand.",
        )

        profile = store.get("s1").family_profile
        assert profile.child_name is None
