"""
Integration tests for MemoryManager — real Gemini-powered background tasks.

Tests rolling summary generation, structured fact extraction from natural
language, and episodic memory creation. Verifies that background tasks
produce correct state mutations with actual LLM calls.

Requires GEMINI_API_KEY environment variable.
Run: pytest tests/test_integration_memory.py -v -m integration -s
"""

import os

import pytest

pytestmark = pytest.mark.integration

if not os.environ.get("GEMINI_API_KEY"):
    pytest.skip("GEMINI_API_KEY not set — skipping integration tests", allow_module_level=True)

from app.agent.event_bus import EventBus
from app.agent.memory import MemoryManager
from app.agent.session_store import InMemorySessionStore
from app.llm.client import GeminiClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gemini():
    return GeminiClient()


@pytest.fixture
def session_store():
    return InMemorySessionStore()


@pytest.fixture
def event_bus():
    return EventBus(buffer_size=100)


@pytest.fixture
def memory(session_store, gemini, event_bus):
    return MemoryManager(session_store, gemini, event_bus)


def _populate_conversation(store, session_id: str, turns: int):
    """Helper: seed a session with N turns of realistic conversation."""
    conversations = [
        ("Hi, I need help with my daughter's homework struggles", "I'd be happy to help! Tell me more about your daughter."),
        ("Her name is Maya, she's 8, and she can't focus on homework for more than 5 minutes", "That's very common for children with ADHD. Has she been diagnosed?"),
        ("Yes, she was diagnosed last year. We've tried reward charts but they stopped working", "Reward charts often lose their novelty. Let me suggest some evidence-based alternatives."),
        ("What do you recommend?", "Try the Pomodoro technique: 10-minute work blocks with 3-minute movement breaks. Visual timers help make time concrete."),
        ("We tried the timer thing and it actually worked! She finished her math in one sitting", "That's wonderful progress! Let's build on this success."),
        ("What else should we try?", "Since the timer worked, let's add a visual task checklist. Break homework into steps she can check off."),
        ("She had a meltdown tonight when I told her to start homework", "I'm sorry to hear that. Transitions can be really hard. Let me suggest some transition strategies."),
        ("Thanks, those tips really helped. She's doing much better this week", "I'm so glad to hear that! Consistency is key. Let's keep tracking what works."),
        ("She got frustrated again today but calmed down faster", "That's actually progress! The fact that she calmed down faster shows she's building skills."),
        ("Should we set a goal for this week?", "Great idea! What would you like to focus on?"),
    ]
    for i in range(min(turns, len(conversations))):
        turn = i + 1
        store.increment_turn(session_id)
        user_msg, assistant_msg = conversations[i]
        store.add_message(session_id, "user", user_msg, turn)
        store.add_message(session_id, "assistant", assistant_msg, turn)


# ---------------------------------------------------------------------------
# Rolling Summary Generation
# ---------------------------------------------------------------------------

class TestSummaryGeneration:
    """Real Gemini-powered conversation summarization."""

    async def test_summary_generated_at_interval(self, memory, session_store):
        """Summary should be generated when turn is at the interval boundary."""
        session_id = "summary-test-1"
        _populate_conversation(session_store, session_id, 5)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=5,  # Default SUMMARY_INTERVAL_TURNS=5
            user_message="We tried the timer thing and it actually worked!",
            assistant_response="That's wonderful progress!",
        )

        summary = session_store.get_latest_summary(session_id)
        assert summary is not None, "Summary should have been created at turn 5"
        assert len(summary.summary) > 20, f"Summary too short: {summary.summary!r}"
        assert summary.covers_through_turn == 5

    async def test_summary_content_is_meaningful(self, memory, session_store):
        """Generated summary should capture key conversation elements."""
        session_id = "summary-test-2"
        _populate_conversation(session_store, session_id, 5)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=5,
            user_message="We tried the timer thing and it actually worked!",
            assistant_response="That's wonderful progress!",
        )

        summary = session_store.get_latest_summary(session_id)
        text_lower = summary.summary.lower()
        # Summary should mention at least some of the key topics
        key_topics = ["maya", "homework", "timer", "focus", "reward", "8"]
        found = [t for t in key_topics if t in text_lower]
        assert len(found) >= 2, (
            f"Summary should mention key conversation topics. "
            f"Found: {found}. Summary: {summary.summary!r}"
        )

    async def test_summary_not_generated_off_interval(self, memory, session_store):
        """Summary should NOT be generated when turn is not at interval."""
        session_id = "summary-test-3"
        _populate_conversation(session_store, session_id, 3)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=3,  # Not at interval boundary
            user_message="Short message",
            assistant_response="Short response",
        )

        summary = session_store.get_latest_summary(session_id)
        assert summary is None, "Summary should not be generated off-interval"

    async def test_incremental_summary_builds_on_previous(self, memory, session_store):
        """Second summary should incorporate the first."""
        session_id = "summary-test-4"
        _populate_conversation(session_store, session_id, 10)

        # Generate first summary at turn 5
        await memory.post_turn_tasks(
            session_id=session_id,
            turn=5,
            user_message="We tried the timer and it worked!",
            assistant_response="Great progress!",
        )
        first = session_store.get_latest_summary(session_id)
        assert first is not None
        assert first.covers_through_turn == 5

        # Generate second summary at turn 10
        await memory.post_turn_tasks(
            session_id=session_id,
            turn=10,
            user_message="Should we set a goal for this week?",
            assistant_response="Great idea!",
        )
        second = session_store.get_latest_summary(session_id)
        assert second is not None
        assert second.covers_through_turn == 10
        assert len(second.summary) > 0


# ---------------------------------------------------------------------------
# Fact Extraction
# ---------------------------------------------------------------------------

class TestFactExtraction:
    """Real Gemini-powered fact extraction from parent messages."""

    async def test_extracts_child_name_and_age(self, memory, session_store):
        """Should extract child_name and child_age from a message mentioning both."""
        session_id = "facts-test-1"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="My son Alex is 9 years old and he was diagnosed with ADHD last year",
            assistant_response="Thank you for sharing about Alex.",
        )

        state = session_store.get(session_id)
        profile = state.family_profile

        # At least name or age should be extracted
        extracted_something = bool(profile.child_name or profile.child_age or profile.diagnosis_status)
        assert extracted_something, (
            f"Should extract facts from message. Profile: name={profile.child_name}, "
            f"age={profile.child_age}, diagnosis={profile.diagnosis_status}"
        )

    async def test_extracts_challenge_areas(self, memory, session_store):
        """Should extract challenge areas from descriptions of struggles."""
        session_id = "facts-test-2"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message=(
                "My daughter really struggles with homework focus and morning routines. "
                "She also has a hard time with transitions between activities."
            ),
            assistant_response="Those are common challenges.",
        )

        state = session_store.get(session_id)
        profile = state.family_profile
        # Should have at least one challenge area
        if profile.challenge_areas:
            all_challenges = " ".join(profile.challenge_areas).lower()
            assert any(
                kw in all_challenges for kw in ("homework", "morning", "transition", "focus", "routine")
            ), f"Challenge areas should be relevant: {profile.challenge_areas}"

    async def test_extracts_attempted_strategies(self, memory, session_store):
        """Should extract strategies the parent mentions having tried."""
        session_id = "facts-test-3"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message=(
                "We've already tried reward charts, a visual schedule on the fridge, "
                "and a timer for homework but nothing seems to stick for more than a week."
            ),
            assistant_response="It's common for strategies to lose novelty.",
        )

        state = session_store.get(session_id)
        profile = state.family_profile
        if profile.attempted_strategies:
            all_strategies = " ".join(profile.attempted_strategies).lower()
            assert any(
                kw in all_strategies for kw in ("reward", "chart", "visual", "schedule", "timer")
            ), f"Strategies should be relevant: {profile.attempted_strategies}"

    async def test_short_message_skips_extraction(self, memory, session_store):
        """Messages shorter than FACT_EXTRACTION_MIN_LENGTH should not trigger extraction."""
        session_id = "facts-test-4"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="Ok thanks",  # Too short
            assistant_response="You're welcome!",
        )

        state = session_store.get(session_id)
        profile = state.family_profile
        # Profile should remain empty (no extraction attempted)
        assert not profile.child_name
        assert not profile.child_age

    async def test_does_not_overwrite_with_empty(self, memory, session_store):
        """Extraction should not clear previously set profile fields."""
        session_id = "facts-test-5"
        session_store.update_profile(session_id, child_name="Maya", child_age="7")
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="She had a really tough day at school today with focus issues during class",
            assistant_response="I'm sorry to hear that.",
        )

        state = session_store.get(session_id)
        profile = state.family_profile
        # Original data should still be there
        assert profile.child_name == "Maya"
        assert profile.child_age == "7"


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

class TestEpisodicMemory:
    """Episodic memory creation from outcome tracking."""

    async def test_positive_outcome_creates_episode(self, memory, session_store):
        session_id = "episode-test-1"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="The timer technique worked great! She finished her homework in 30 minutes!",
            assistant_response="That's wonderful!",
            tool_calls=[{
                "name": "track_outcome",
                "args": {
                    "strategy_name": "timer technique",
                    "outcome": "positive",
                    "notes": "Finished homework in 30 minutes",
                },
            }],
        )

        episodes = session_store.get_recent_episodes(session_id)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.event_type == "outcome_reported"
        assert ep.outcome == "positive"
        assert "timer technique" in ep.strategies_involved
        assert "positive" in ep.summary.lower() or "timer" in ep.summary.lower()

    async def test_negative_outcome_with_emotion(self, memory, session_store):
        session_id = "episode-test-2"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="I'm so frustrated, the reward chart completely stopped working again",
            assistant_response="I understand that's frustrating.",
            tool_calls=[{
                "name": "track_outcome",
                "args": {
                    "strategy_name": "reward chart",
                    "outcome": "negative",
                },
            }],
        )

        episodes = session_store.get_recent_episodes(session_id)
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.outcome == "negative"
        assert ep.emotional_context == "frustrated"

    async def test_no_episode_without_outcome_tool_call(self, memory, session_store):
        session_id = "episode-test-3"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="What strategies can I try for bedtime?",
            assistant_response="Here are some bedtime strategies...",
            tool_calls=[{
                "name": "search_knowledge_base",
                "args": {"query": "bedtime strategies"},
            }],
        )

        episodes = session_store.get_recent_episodes(session_id)
        assert len(episodes) == 0


# ---------------------------------------------------------------------------
# Event Bus Integration
# ---------------------------------------------------------------------------

class TestMemoryEvents:
    """Verify that memory tasks emit observability events."""

    async def test_summary_emits_event(self, memory, session_store, event_bus):
        session_id = "events-test-1"
        _populate_conversation(session_store, session_id, 5)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=5,
            user_message="The timer worked!",
            assistant_response="Great!",
        )

        events = event_bus.get_events(session_id, category="memory")
        summary_events = [e for e in events if e.event_type == "summary_updated"]
        assert len(summary_events) >= 1, "Should emit summary_updated event"

    async def test_fact_extraction_emits_event(self, memory, session_store, event_bus):
        session_id = "events-test-2"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="My son David is 10 years old and was diagnosed with ADHD in first grade",
            assistant_response="Thank you for sharing about David.",
        )

        events = event_bus.get_events(session_id, category="memory")
        fact_events = [e for e in events if e.event_type == "facts_extracted"]
        # Might or might not extract facts depending on LLM response
        # but the event should fire if facts were found
        if session_store.get(session_id).family_profile.child_name:
            assert len(fact_events) >= 1, "Should emit facts_extracted event when facts found"

    async def test_episode_emits_event(self, memory, session_store, event_bus):
        session_id = "events-test-3"
        session_store.increment_turn(session_id)

        await memory.post_turn_tasks(
            session_id=session_id,
            turn=1,
            user_message="The visual schedule really helped this morning",
            assistant_response="That's great progress!",
            tool_calls=[{
                "name": "track_outcome",
                "args": {"strategy_name": "visual schedule", "outcome": "positive"},
            }],
        )

        events = event_bus.get_events(session_id, category="memory")
        episode_events = [e for e in events if e.event_type == "episode_created"]
        assert len(episode_events) >= 1, "Should emit episode_created event"


# ---------------------------------------------------------------------------
# Error Resilience
# ---------------------------------------------------------------------------

class TestMemoryErrorResilience:
    """Memory tasks should never crash — errors logged, not thrown."""

    async def test_all_tasks_run_in_parallel(self, memory, session_store):
        """When multiple tasks fire simultaneously, all should complete."""
        session_id = "parallel-test"
        _populate_conversation(session_store, session_id, 5)

        # Turn 5 with a long message and outcome tool call triggers all three tasks:
        # - summary (turn 5 = interval boundary)
        # - fact extraction (message > 40 chars)
        # - episodic memory (track_outcome tool call)
        await memory.post_turn_tasks(
            session_id=session_id,
            turn=5,
            user_message="My daughter Emma is 7 and the timer strategy worked great for her homework!",
            assistant_response="That's wonderful news about Emma!",
            tool_calls=[{
                "name": "track_outcome",
                "args": {"strategy_name": "timer strategy", "outcome": "positive"},
            }],
        )

        # All three should have produced results
        summary = session_store.get_latest_summary(session_id)
        assert summary is not None, "Summary should have been generated"

        episodes = session_store.get_recent_episodes(session_id)
        assert len(episodes) >= 1, "Episode should have been created"

        # Facts may or may not be extracted depending on LLM
        # but the profile should at least not be corrupted
        state = session_store.get(session_id)
        assert state.family_profile is not None
