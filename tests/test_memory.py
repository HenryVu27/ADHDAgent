"""Tests for MemoryManager — summary, fact extraction, episodic memory."""

import asyncio

import pytest
from unittest.mock import AsyncMock, patch

from app.agent.memory import MemoryManager
from app.agent.session_store import create_in_memory_store
from app.models.schemas import EpisodicMemory, SessionSummary


async def _make_store_with_messages(session_id="s1", turn_count=5):
    """Create a store with some conversation history."""
    store = await create_in_memory_store()
    for t in range(1, turn_count + 1):
        await store.increment_turn(session_id)
        await store.add_message(session_id, "user", f"User message turn {t}", turn=t)
        await store.add_message(session_id, "assistant", f"Assistant response turn {t}", turn=t)
    await store.commit()
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
        store = await _make_store_with_messages(turn_count=5)
        gemini = _make_gemini_mock(generate_return="The parent discussed homework challenges.")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="This is turn five with enough content for processing",
            assistant_response="Got it.",
        )

        # Summary should have been saved (turn 5 % 5 == 0)
        summary = await store.get_latest_summary("s1")
        assert summary is not None
        assert summary.covers_through_turn == 5
        assert "homework" in summary.summary

    @pytest.mark.asyncio
    async def test_summary_not_triggered_off_interval(self):
        store = await _make_store_with_messages(turn_count=3)
        gemini = _make_gemini_mock()
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=3,
            user_message="Short",
            assistant_response="Got it.",
        )

        # Turn 3 % 5 != 0, no summary
        assert await store.get_latest_summary("s1") is None

    @pytest.mark.asyncio
    async def test_summary_skipped_without_gemini(self):
        store = await _make_store_with_messages(turn_count=5)
        mm = MemoryManager(session_store=store, gemini_client=None)

        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="Some message here",
            assistant_response="Response.",
        )

        assert await store.get_latest_summary("s1") is None

    @pytest.mark.asyncio
    async def test_summary_includes_prior(self):
        """Second summary should include reference to prior summary in prompt."""
        store = await _make_store_with_messages(turn_count=10)
        gemini = _make_gemini_mock(generate_return="Extended summary.")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        # Seed a prior summary
        await store.save_summary("s1", SessionSummary(summary="First summary.", covers_through_turn=5))
        await store.commit()

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
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock(extract_json_return={"child_name": "Kai", "child_age": "7"})
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        long_msg = "My son Kai is 7 years old and he struggles with homework every evening"
        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message=long_msg,
            assistant_response="I understand.",
        )

        profile = (await store.get("s1")).family_profile
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    @pytest.mark.asyncio
    async def test_skips_short_messages(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
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
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
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

        profile = (await store.get("s1")).family_profile
        assert profile.child_name == "Kai"

    @pytest.mark.asyncio
    async def test_skips_empty_extraction(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock(extract_json_return={})
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        long_msg = "I appreciate the advice you gave me, it was really helpful thank you"
        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message=long_msg,
            assistant_response="You're welcome!",
        )

        # Profile should remain untouched
        profile = (await store.get("s1")).family_profile
        assert profile.child_name is None


class TestEpisodicMemory:

    @pytest.mark.asyncio
    async def test_episode_created_on_track_outcome(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
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

        episodes = await store.get_recent_episodes("s1")
        # May include an emotional_shift episode in addition to outcome_reported
        assert len(episodes) >= 1
        outcome_eps = [ep for ep in episodes if ep.event_type == "outcome_reported"]
        assert len(outcome_eps) == 1
        assert "visual timer" in outcome_eps[0].summary
        assert outcome_eps[0].outcome == "positive"
        assert "visual timer" in outcome_eps[0].strategies_involved
        assert outcome_eps[0].emotional_context == "hopeful"

    @pytest.mark.asyncio
    async def test_no_episode_without_track_outcome(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
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

        episodes = await store.get_recent_episodes("s1")
        assert len(episodes) == 0

    @pytest.mark.asyncio
    async def test_multiple_outcomes_create_multiple_episodes(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
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

        episodes = await store.get_recent_episodes("s1")
        assert len(episodes) == 2


class TestEmotionInferenceLLM:

    @pytest.mark.asyncio
    async def test_llm_called_with_message(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock(generate_return="frustrated")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        result = await mm._infer_emotion("s1", "I'm so frustrated with bedtime")

        assert result == "frustrated"
        prompt_arg = gemini.generate.call_args[0][0]
        assert "frustrated with bedtime" in prompt_arg

    @pytest.mark.asyncio
    async def test_includes_conversation_context(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.add_message("s1", "user", "My son has ADHD", turn=1)
        await store.add_message("s1", "assistant", "Tell me more about him", turn=1)
        await store.commit()
        gemini = _make_gemini_mock(generate_return="anxious")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm._infer_emotion("s1", "I'm worried about his grades")

        prompt_arg = gemini.generate.call_args[0][0]
        assert "My son has ADHD" in prompt_arg
        assert "Tell me more about him" in prompt_arg

    @pytest.mark.asyncio
    async def test_invalid_output_defaults_empty(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock(generate_return="something unexpected")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        result = await mm._infer_emotion("s1", "We tried the timer")

        assert result == ""

    @pytest.mark.asyncio
    async def test_neutral_returns_empty(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock(generate_return="neutral")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        result = await mm._infer_emotion("s1", "We tried the timer yesterday")

        assert result == ""

    @pytest.mark.asyncio
    async def test_no_gemini_returns_empty(self):
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        mm = MemoryManager(session_store=store, gemini_client=None)

        result = await mm._infer_emotion("s1", "I'm so frustrated")

        assert result == ""


class TestErrorResilience:

    @pytest.mark.asyncio
    async def test_summary_error_does_not_crash(self):
        store = await _make_store_with_messages(turn_count=5)
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
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock()
        gemini.extract_json = AsyncMock(side_effect=Exception("Parse error"))
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="My son Kai is 7 and he has trouble with homework every day",
            assistant_response="I understand.",
        )

        # Profile should not be updated since extraction failed
        profile = (await store.get("s1")).family_profile
        assert profile.child_name is None

    @pytest.mark.asyncio
    async def test_summary_timeout_raises(self):
        """TimeoutError from generate() should propagate (caught by gather)."""
        store = await _make_store_with_messages(turn_count=5)
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
        assert await store.get_latest_summary("s1") is None

    @pytest.mark.asyncio
    async def test_fact_extraction_timeout_does_not_crash(self):
        """TimeoutError from extract_json() should not crash post_turn_tasks."""
        store = await create_in_memory_store()
        await store.increment_turn("s1")
        await store.commit()
        gemini = _make_gemini_mock()
        gemini.extract_json = AsyncMock(side_effect=asyncio.TimeoutError())
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="My son Kai is 7 and he has trouble with homework every day",
            assistant_response="I understand.",
        )

        profile = (await store.get("s1")).family_profile
        assert profile.child_name is None


class TestConflictDetection:

    @pytest.mark.asyncio
    async def test_negated_strategy_triggers_outcome(self):
        """When user says a strategy doesn't work, an outcome entry is created."""
        store = await create_in_memory_store()
        await store.update_profile("s1", attempted_strategies=["timer technique"])
        await store.commit()

        # LLM extracts a negation
        gemini = _make_gemini_mock(extract_json_return={
            "negated_strategies": ["timer technique"],
        })
        mm = MemoryManager(session_store=store, gemini_client=gemini)
        await mm._extract_facts(
            "s1",
            "We stopped using the timer, it just made him more anxious.",
            turn=3,
        )

        state = await store.get("s1")
        negative_outcomes = [o for o in state.outcomes if o.signal == "negative"]
        assert any("timer" in o.strategy_name.lower() for o in negative_outcomes)

    @pytest.mark.asyncio
    async def test_scalar_correction_emits_event(self):
        """When age changes, the event bus receives a correction event."""
        from unittest.mock import AsyncMock
        store = await create_in_memory_store()
        await store.update_profile("s1", child_age="7")
        await store.commit()

        event_bus = AsyncMock()
        event_bus.emit = AsyncMock()

        gemini = _make_gemini_mock(extract_json_return={"child_age": "8"})
        mm = MemoryManager(session_store=store, gemini_client=gemini, event_bus=event_bus)

        await mm._extract_facts("s1", "Oh, he just turned 8 last week.", turn=4)

        event_bus.emit.assert_called()
        call_args = [str(c) for c in event_bus.emit.call_args_list]
        assert any("correction" in arg.lower() or "profile_corrected" in arg for arg in call_args)

    @pytest.mark.asyncio
    async def test_no_negation_in_normal_message(self):
        """Normal message with no negations should not create negative outcomes."""
        store = await create_in_memory_store()
        await store.update_profile("s1", attempted_strategies=["timer technique"])
        await store.commit()

        gemini = _make_gemini_mock(extract_json_return={"child_age": "8"})
        mm = MemoryManager(session_store=store, gemini_client=gemini)
        await mm._extract_facts("s1", "He just turned 8.", turn=4)

        state = await store.get("s1")
        assert state.outcomes == []


class TestBroaderEpisodicEvents:

    @pytest.mark.asyncio
    async def test_goal_set_creates_episode(self):
        store = await _make_store_with_messages(turn_count=2)
        gemini = _make_gemini_mock(generate_return="neutral")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=2,
            user_message="Let's work on improving bedtime.",
            assistant_response="Great, I'll set that as a goal.",
            tool_calls=[{"name": "manage_goals", "args": {"action": "add", "description": "Improve bedtime routine"}}],
        )

        episodes = await store.get_recent_episodes("s1")
        assert any(ep.event_type == "goal_set" for ep in episodes)

    @pytest.mark.asyncio
    async def test_goal_completed_creates_episode(self):
        store = await _make_store_with_messages(turn_count=3)
        gemini = _make_gemini_mock(generate_return="hopeful")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=3,
            user_message="We actually got bedtime sorted!",
            assistant_response="That's wonderful progress.",
            tool_calls=[{"name": "manage_goals", "args": {"action": "complete", "description": "Improve bedtime routine"}}],
        )

        episodes = await store.get_recent_episodes("s1")
        assert any(ep.event_type == "goal_completed" for ep in episodes)
        completed = next(ep for ep in episodes if ep.event_type == "goal_completed")
        assert completed.outcome == "positive"

    @pytest.mark.asyncio
    async def test_emotional_shift_creates_episode(self):
        """A strongly negative emotion (overwhelmed) should create an emotional_shift episode."""
        store = await _make_store_with_messages(turn_count=4)
        # Mock: emotion inference returns "overwhelmed"
        gemini = _make_gemini_mock(generate_return="overwhelmed")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=4,
            user_message="I just don't know what to do anymore, I feel completely lost.",
            assistant_response="I hear you, this sounds really hard.",
        )

        episodes = await store.get_recent_episodes("s1")
        assert any(ep.event_type == "emotional_shift" for ep in episodes)

    async def test_hopeful_emotion_creates_episode(self):
        """A positive but mild emotion (hopeful) should also create an emotional_shift episode."""
        store = await _make_store_with_messages(turn_count=3)
        gemini = _make_gemini_mock(generate_return="hopeful")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=3,
            user_message="We tried the timer today and Alex actually finished his homework!",
            assistant_response="That is great to hear!",
        )

        episodes = await store.get_recent_episodes("s1")
        assert any(ep.event_type == "emotional_shift" for ep in episodes)

    async def test_any_non_neutral_emotion_creates_episode(self):
        """positive emotion should create an episode (currently blocked by high-intensity gate)."""
        store = await _make_store_with_messages(turn_count=2)
        gemini = _make_gemini_mock(generate_return="positive")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=2,
            user_message="Things are going really well this week.",
            assistant_response="I'm glad to hear that.",
        )

        episodes = await store.get_recent_episodes("s1")
        emotional = [ep for ep in episodes if ep.event_type == "emotional_shift"]
        assert len(emotional) == 1
        assert emotional[0].emotional_context == "positive"

    @pytest.mark.asyncio
    async def test_neutral_emotion_no_emotional_episode(self):
        """Neutral emotion should not create an emotional_shift episode."""
        store = await _make_store_with_messages(turn_count=2)
        gemini = _make_gemini_mock(generate_return="neutral")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm.post_turn_tasks(
            session_id="s1", turn=2,
            user_message="He did okay today at school.",
            assistant_response="Good to hear.",
        )

        episodes = await store.get_recent_episodes("s1")
        assert not any(ep.event_type == "emotional_shift" for ep in episodes)
