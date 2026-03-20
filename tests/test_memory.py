"""Tests for the Graphiti-based MemoryManager."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_graphiti():
    client = AsyncMock()
    client.add_episode = AsyncMock(return_value=None)
    client.search = AsyncMock(return_value=[])
    client.driver = MagicMock()
    client.driver.execute_query = AsyncMock(return_value=[])
    return client


@pytest.fixture
def mock_store():
    store = AsyncMock()
    state = MagicMock()
    state.family_profile.child_name = None
    state.family_profile.child_age = None
    state.family_profile.diagnosis_status = None
    state.family_profile.adhd_subtype = None
    store.get = AsyncMock(return_value=state)
    store.update_profile = AsyncMock()
    return store


@pytest.fixture
def mock_event_bus():
    bus = AsyncMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture
def memory_manager(mock_graphiti, mock_store, mock_event_bus):
    from app.agent.memory import MemoryManager
    return MemoryManager(
        session_store=mock_store,
        graphiti_client=mock_graphiti,
        event_bus=mock_event_bus,
    )


@pytest.mark.asyncio
async def test_post_turn_calls_add_episode(memory_manager, mock_graphiti):
    """post_turn_tasks should call graphiti.add_episode with formatted turn."""
    await memory_manager.post_turn_tasks(
        session_id="sess1",
        turn=3,
        user_message="My son keeps losing homework",
        assistant_response="That's a common challenge...",
        user_id=1,
    )
    mock_graphiti.add_episode.assert_called_once()
    call_kwargs = mock_graphiti.add_episode.call_args
    assert "Parent: My son keeps losing homework" in call_kwargs.kwargs.get("episode_body", "")
    assert call_kwargs.kwargs.get("group_id") == "1"


@pytest.mark.asyncio
async def test_post_turn_skips_when_no_graphiti(mock_store, mock_event_bus):
    """When graphiti_client is None, post_turn_tasks does nothing."""
    from app.agent.memory import MemoryManager
    mm = MemoryManager(session_store=mock_store, graphiti_client=None, event_bus=mock_event_bus)
    await mm.post_turn_tasks("sess1", 1, "hello", "hi there", user_id=1)
    # No error, no calls


@pytest.mark.asyncio
async def test_post_turn_handles_add_episode_failure(memory_manager, mock_graphiti, mock_event_bus):
    """add_episode failure should be logged, not raised."""
    mock_graphiti.add_episode.side_effect = Exception("Neo4j down")
    await memory_manager.post_turn_tasks("sess1", 1, "hello", "hi", user_id=1)
    mock_event_bus.emit.assert_any_call(
        "memory", "graphiti_ingestion_failed", "sess1", 1,
        detail={"error": "Neo4j down"},
    )


@pytest.mark.asyncio
async def test_sync_profile_fills_empty_fields(memory_manager, mock_graphiti, mock_store):
    """_sync_profile should update SQLite when graph has data and SQLite is empty."""
    record = {"child_name": "Alex", "child_age": "8", "diagnosis_status": None, "adhd_subtype": None}
    mock_graphiti.driver.execute_query = AsyncMock(return_value=[record])
    await memory_manager._sync_profile("sess1", 1)
    mock_store.update_profile.assert_called_once_with(
        "sess1", child_name="Alex", child_age="8",
    )
