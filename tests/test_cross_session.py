"""Tests for cross-session persistence: user profiles, longitudinal summaries,
user-level episodes/outcomes, profile seeding, and context injection."""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import aiosqlite
import pytest
import pytest_asyncio

from app.agent.sqlite_store import SQLiteSessionStore
from app.db import init_db_async
from app.models.schemas import (
    FamilyProfile,
    SeedSessionRequest,
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
