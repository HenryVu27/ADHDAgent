"""Tests for cross-session persistence: user profiles, longitudinal summaries,
user-level episodes/outcomes, profile seeding, and context injection."""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
import pytest_asyncio

from app.agent.sqlite_store import SQLiteSessionStore
from app.agent.memory import MemoryManager
from app.db import init_db_async
from app.models.schemas import (
    EpisodicMemory,
    FamilyProfile,
    SeedSessionRequest,
    UserSummary,
)


@pytest_asyncio.fixture
async def store():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = sqlite3.Row
    await conn.execute("PRAGMA foreign_keys=ON")
    await init_db_async(conn)
    s = SQLiteSessionStore(conn)
    yield s
    await conn.close()


@pytest_asyncio.fixture
async def store_with_user(store):
    """Store with a pre-created user (id=1)."""
    await store._conn.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        ("parent@test.com", "hash"),
    )
    await store._conn.commit()
    return store


# --- User profile CRUD ---


@pytest.mark.asyncio
async def test_get_user_profile_none(store_with_user):
    result = await store_with_user.get_user_profile(1)
    assert result is None


@pytest.mark.asyncio
async def test_update_user_profile_creates_and_returns(store_with_user):
    profile = await store_with_user.update_user_profile(
        1, child_name="Alex", child_age="8", challenge_areas=["homework", "mornings"],
    )
    assert profile.child_name == "Alex"
    assert profile.child_age == "8"
    assert profile.challenge_areas == ["homework", "mornings"]


@pytest.mark.asyncio
async def test_update_user_profile_merges_lists(store_with_user):
    await store_with_user.update_user_profile(1, challenge_areas=["homework"])
    profile = await store_with_user.update_user_profile(1, challenge_areas=["mornings"])
    assert profile.challenge_areas == ["homework", "mornings"]


@pytest.mark.asyncio
async def test_get_user_profile_returns_data(store_with_user):
    await store_with_user.update_user_profile(1, child_name="Alex", child_age="8")
    await store_with_user.commit()
    profile = await store_with_user.get_user_profile(1)
    assert profile is not None
    assert profile.child_name == "Alex"
    assert profile.child_age == "8"


# --- Profile write-through ---


@pytest.mark.asyncio
async def test_session_update_profile_writes_through_to_user(store_with_user):
    """When updating a session profile linked to a user, user profile is also updated."""
    await store_with_user._ensure_session("sess-1", user_id=1)
    await store_with_user.update_profile("sess-1", child_name="Alex", child_age="7")
    await store_with_user.commit()

    user_profile = await store_with_user.get_user_profile(1)
    assert user_profile is not None
    assert user_profile.child_name == "Alex"
    assert user_profile.child_age == "7"


@pytest.mark.asyncio
async def test_session_update_profile_no_writethrough_anonymous(store_with_user):
    """Anonymous sessions (no user_id) should not write-through."""
    await store_with_user._ensure_session("anon-sess")
    await store_with_user.update_profile("anon-sess", child_name="Bob")
    await store_with_user.commit()

    # No user profile should exist (user 1 was not linked)
    user_profile = await store_with_user.get_user_profile(1)
    assert user_profile is None


# --- Session seeding from user profile ---


@pytest.mark.asyncio
async def test_ensure_session_seeds_from_user_profile(store_with_user):
    """New session for a known user copies user profile into session."""
    await store_with_user.update_user_profile(
        1, child_name="Alex", child_age="8", challenge_areas=["homework"],
    )
    await store_with_user.commit()

    await store_with_user._ensure_session("new-sess", user_id=1)
    state = await store_with_user.get("new-sess")
    assert state.family_profile.child_name == "Alex"
    assert state.family_profile.child_age == "8"
    assert state.family_profile.challenge_areas == ["homework"]


@pytest.mark.asyncio
async def test_ensure_session_no_user_profile_creates_empty(store_with_user):
    """New session for user without a profile gets empty defaults."""
    await store_with_user._ensure_session("new-sess", user_id=1)
    state = await store_with_user.get("new-sess")
    assert state.family_profile.child_name is None
    assert state.family_profile.challenge_areas == []


@pytest.mark.asyncio
async def test_ensure_session_idempotent(store_with_user):
    """Calling _ensure_session twice doesn't overwrite existing session data."""
    await store_with_user._ensure_session("sess-1", user_id=1)
    await store_with_user.update_profile("sess-1", child_name="Original")
    await store_with_user.commit()

    # Second call should be a no-op (INSERT OR IGNORE)
    await store_with_user.update_user_profile(1, child_name="Updated")
    await store_with_user._ensure_session("sess-1", user_id=1)

    state = await store_with_user.get("sess-1")
    assert state.family_profile.child_name == "Original"


# --- User summary ---


@pytest.mark.asyncio
async def test_user_summary_roundtrip(store_with_user):
    summary = UserSummary(summary="Parent is making progress", covers_through_session="sess-1")
    await store_with_user.save_user_summary(1, summary)
    await store_with_user.commit()

    result = await store_with_user.get_user_summary(1)
    assert result is not None
    assert result.summary == "Parent is making progress"
    assert result.covers_through_session == "sess-1"


@pytest.mark.asyncio
async def test_user_summary_returns_latest(store_with_user):
    s1 = UserSummary(summary="First", covers_through_session="sess-1")
    s2 = UserSummary(summary="Second", covers_through_session="sess-2")
    await store_with_user.save_user_summary(1, s1)
    await store_with_user.save_user_summary(1, s2)
    await store_with_user.commit()

    result = await store_with_user.get_user_summary(1)
    assert result.summary == "Second"


@pytest.mark.asyncio
async def test_user_summary_none_when_empty(store_with_user):
    result = await store_with_user.get_user_summary(1)
    assert result is None


# --- Cross-session episodes and outcomes ---


@pytest.mark.asyncio
async def test_get_user_episodes_across_sessions(store_with_user):
    await store_with_user._ensure_session("sess-1", user_id=1)
    await store_with_user._ensure_session("sess-2", user_id=1)

    ep1 = EpisodicMemory(event_type="outcome_reported", summary="Timer worked well")
    ep2 = EpisodicMemory(event_type="goal_set", summary="Set bedtime goal")
    await store_with_user.add_episode("sess-1", ep1)
    await store_with_user.add_episode("sess-2", ep2)
    await store_with_user.commit()

    episodes = await store_with_user.get_user_episodes(1, limit=10)
    assert len(episodes) == 2
    summaries = [e.summary for e in episodes]
    assert "Timer worked well" in summaries
    assert "Set bedtime goal" in summaries


@pytest.mark.asyncio
async def test_get_user_outcomes_across_sessions(store_with_user):
    await store_with_user._ensure_session("sess-1", user_id=1)
    await store_with_user._ensure_session("sess-2", user_id=1)

    await store_with_user.add_outcome("sess-1", "Timer method", "positive", "Worked great")
    await store_with_user.add_outcome("sess-2", "Reward chart", "negative", "Lost interest")
    await store_with_user.commit()

    outcomes = await store_with_user.get_user_outcomes(1, limit=10)
    assert len(outcomes) == 2
    names = [o.strategy_name for o in outcomes]
    assert "Timer method" in names
    assert "Reward chart" in names


@pytest.mark.asyncio
async def test_get_user_episodes_respects_limit(store_with_user):
    await store_with_user._ensure_session("sess-1", user_id=1)
    for i in range(10):
        ep = EpisodicMemory(event_type="outcome_reported", summary=f"Episode {i}")
        await store_with_user.add_episode("sess-1", ep)
    await store_with_user.commit()

    episodes = await store_with_user.get_user_episodes(1, limit=3)
    assert len(episodes) == 3


# --- Longitudinal summary generation (mocked LLM) ---


@pytest.mark.asyncio
async def test_end_of_session_tasks_generates_summary(store_with_user):
    # Set up previous session with data
    await store_with_user._ensure_session("prev-sess", user_id=1)
    await store_with_user.add_outcome("prev-sess", "Timer", "positive", "Worked well")
    ep = EpisodicMemory(event_type="outcome_reported", summary="Timer strategy worked")
    await store_with_user.add_episode("prev-sess", ep)
    await store_with_user.commit()

    mock_gemini = AsyncMock()
    mock_gemini.generate = AsyncMock(return_value="Parent has been trying timer strategies with positive results.")

    manager = MemoryManager(store_with_user, mock_gemini)
    await manager.end_of_session_tasks(user_id=1, previous_session_id="prev-sess")

    # Verify summary was saved
    summary = await store_with_user.get_user_summary(1)
    assert summary is not None
    assert "timer" in summary.summary.lower()
    assert summary.covers_through_session == "prev-sess"


@pytest.mark.asyncio
async def test_end_of_session_tasks_skips_without_gemini(store_with_user):
    manager = MemoryManager(store_with_user, None)
    await manager.end_of_session_tasks(user_id=1, previous_session_id="prev-sess")

    summary = await store_with_user.get_user_summary(1)
    assert summary is None


# --- Context injection (prepare_context) ---


@pytest.mark.asyncio
async def test_prepare_context_injects_prior_sessions(store_with_user):
    from app.agent.hooks import create_prepare_context
    from langchain_core.messages import HumanMessage

    # Set up user data
    await store_with_user.update_user_profile(1, child_name="Alex")
    summary = UserSummary(summary="Parent has been working on timers", covers_through_session="prev-sess")
    await store_with_user.save_user_summary(1, summary)
    await store_with_user.commit()

    # Create new session for this user
    await store_with_user._ensure_session("new-sess", user_id=1)

    prepare_context = create_prepare_context(store_with_user)

    state = {
        "messages": [HumanMessage(content="Hello, we're back!")],
        "session_id": "new-sess",
        "user_id": 1,
    }

    result = await prepare_context(state)
    system_content = result["llm_input_messages"][0].content
    assert "<prior-sessions>" in system_content
    assert "Journey so far:" in system_content
    assert "timers" in system_content


@pytest.mark.asyncio
async def test_prepare_context_no_prior_sessions_for_new_user(store_with_user):
    from app.agent.hooks import create_prepare_context
    from langchain_core.messages import HumanMessage

    await store_with_user._ensure_session("first-sess", user_id=1)

    prepare_context = create_prepare_context(store_with_user)

    state = {
        "messages": [HumanMessage(content="Hello!")],
        "session_id": "first-sess",
        "user_id": 1,
    }

    result = await prepare_context(state)
    system_content = result["llm_input_messages"][0].content
    assert "<prior-sessions>" not in system_content


@pytest.mark.asyncio
async def test_prepare_context_skips_prior_sessions_anonymous(store_with_user):
    from app.agent.hooks import create_prepare_context
    from langchain_core.messages import HumanMessage

    await store_with_user._ensure_session("anon-sess")

    prepare_context = create_prepare_context(store_with_user)

    state = {
        "messages": [HumanMessage(content="Hello!")],
        "session_id": "anon-sess",
        "user_id": None,
    }

    result = await prepare_context(state)
    system_content = result["llm_input_messages"][0].content
    assert "<prior-sessions>" not in system_content


@pytest.mark.asyncio
async def test_prepare_context_skips_prior_sessions_after_turn_1(store_with_user):
    """Cross-session context only injected on turn 1, not later turns."""
    from app.agent.hooks import create_prepare_context
    from langchain_core.messages import HumanMessage

    await store_with_user.update_user_profile(1, child_name="Alex")
    summary = UserSummary(summary="Journey info", covers_through_session="prev")
    await store_with_user.save_user_summary(1, summary)
    await store_with_user.commit()

    await store_with_user._ensure_session("sess", user_id=1)
    # Simulate turn 2
    await store_with_user.increment_turn("sess")
    await store_with_user.increment_turn("sess")

    prepare_context = create_prepare_context(store_with_user)

    state = {
        "messages": [HumanMessage(content="Follow up question")],
        "session_id": "sess",
        "user_id": 1,
    }

    result = await prepare_context(state)
    system_content = result["llm_input_messages"][0].content
    assert "<prior-sessions>" not in system_content


# --- Seed session write-through ---


@pytest.mark.asyncio
async def test_seed_session_writes_through_to_user_profile(store_with_user):
    """AgentOrchestrator.seed_session should write-through to user profile."""
    from app.agent.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator(
        agent=AsyncMock(),
        session_store=store_with_user,
    )

    request = SeedSessionRequest(
        session_id="seed-sess",
        child_name="Alex",
        child_age="8",
        challenges=["homework"],
    )
    await orch.seed_session(request, user_id=1)

    user_profile = await store_with_user.get_user_profile(1)
    assert user_profile is not None
    assert user_profile.child_name == "Alex"
    assert user_profile.child_age == "8"
    assert user_profile.challenge_areas == ["homework"]
