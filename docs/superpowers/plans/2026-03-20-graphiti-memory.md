# Graphiti Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MemoryManager subsystem with Graphiti + Neo4j Aura for temporal knowledge graph memory with semantic retrieval.

**Architecture:** Graphiti ingests conversation turns, extracts entities/relationships into Neo4j, and provides hybrid search (vector + BM25 + graph traversal). SQLite retains session metadata, messages, traces, goals, outcomes, and profiles. Qdrant RAG pipeline is untouched.

**Tech Stack:** graphiti-core[google-genai], Neo4j Aura (free tier), Gemini Flash (LLM/embeddings), existing FastAPI + LangGraph + aiosqlite + Qdrant

**Spec:** `docs/superpowers/specs/2026-03-20-graphiti-memory-design.md`

**Testing policy:** Never run integration tests requiring real API keys (GEMINI_API_KEY, Neo4j credentials) without asking. All unit tests use mocks. Run tests with: `./adhd312/Scripts/python.exe -m pytest tests/<file> -v`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `app/agent/graphiti_client.py` | **NEW** -- Graphiti factory, custom entity/edge types, Cypher helpers |
| `app/agent/memory.py` | **REWRITE** -- Thin wrapper: `post_turn_tasks` calls `graphiti.add_episode()` + `_sync_profile` |
| `app/agent/hooks.py` | **MODIFY** -- Context assembly queries Graphiti instead of SQLite summaries/episodes |
| `app/agent/tools.py` | **MODIFY** -- Add `search_memory` tool, `_get_user_id` helper |
| `app/agent/prompts.py` | **MODIFY** -- Replace session summary template section with graph memory context, add `search_memory` tool description |
| `app/agent/graph.py` | **MODIFY** -- Pass `search_memory` to tool list |
| `app/agent/orchestrator.py` | **MODIFY** -- Add `user_id` to configurable, remove `end_of_session_tasks` call |
| `app/agent/store_protocol.py` | **MODIFY** -- Remove 12 abstract methods |
| `app/agent/sqlite_store.py` | **MODIFY** -- Remove implementations of deleted methods |
| `app/agent/session_store.py` | **MODIFY** -- Remove in-memory implementations of deleted methods |
| `app/config.py` | **MODIFY** -- Add NEO4J_* and GRAPHITI_* settings |
| `app/main.py` | **MODIFY** -- Initialize Graphiti, wire into MemoryManager + tools + hooks, shutdown handler |
| `tests/test_graphiti_client.py` | **NEW** -- Unit tests for graphiti_client.py |
| `tests/test_memory.py` | **REWRITE** -- Mock Graphiti instead of SQLite |
| `tests/test_agent_hooks.py` | **MODIFY** -- Update context assembly assertions |
| `tests/test_agent_tools.py` | **MODIFY** -- Add search_memory tests |

---

### Task 1: Install Dependencies and Add Config

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py:8-121`

- [ ] **Step 1: Install graphiti-core with Gemini support**

Run: `./adhd312/Scripts/pip.exe install "graphiti-core[google-genai]"`

- [ ] **Step 2: Add graphiti-core to requirements.txt**

Add `graphiti-core[google-genai]` to `requirements.txt`. Find where other ML/AI packages are listed and add it nearby.

- [ ] **Step 3: Add config fields to app/config.py**

Add these fields to the `Settings` class, after the existing `SQLITE_ENABLED` block (around line 98):

```python
    # Neo4j (for Graphiti memory graph)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # Graphiti memory
    GRAPHITI_ENABLED: bool = True
    GRAPHITI_LLM_MODEL: str = ""
    GRAPHITI_EMBEDDING_MODEL: str = ""
    GRAPHITI_CONTEXT_RESULTS: int = 10
    GRAPHITI_SEARCH_RESULTS: int = 10
    GRAPHITI_INGESTION_TIMEOUT_S: float = 30.0
```

- [ ] **Step 4: Add model defaults to _fill_model_defaults validator**

In the existing `_fill_model_defaults` method (line 111), add after the `GEMINI_FAST_MODEL` default:

```python
        if not self.GRAPHITI_LLM_MODEL:
            self.GRAPHITI_LLM_MODEL = self.GEMINI_UTILITY_MODEL
        if not self.GRAPHITI_EMBEDDING_MODEL:
            self.GRAPHITI_EMBEDDING_MODEL = self.GEMINI_EMBEDDING_MODEL
```

- [ ] **Step 5: Verify .env has Neo4j credentials**

Check that `.env` contains `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`. These were added during the design phase.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/config.py
git commit -m "Add Graphiti and Neo4j configuration"
```

---

### Task 2: Create Graphiti Client Module

**Files:**
- Create: `app/agent/graphiti_client.py`
- Test: `tests/test_graphiti_client.py`

- [ ] **Step 1: Write test for create_graphiti_client factory**

Create `tests/test_graphiti_client.py`:

```python
"""Tests for the Graphiti client factory and custom types."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graphiti_client import (
    create_graphiti_client,
    ADHDEntityTypes,
    ADHDEdgeTypes,
)


@pytest.mark.asyncio
async def test_create_graphiti_client_returns_none_when_disabled():
    """When GRAPHITI_ENABLED=False, factory returns None."""
    with patch("app.agent.graphiti_client.settings") as mock_settings:
        mock_settings.GRAPHITI_ENABLED = False
        mock_settings.NEO4J_PASSWORD = "test"
        result = await create_graphiti_client()
        assert result is None


@pytest.mark.asyncio
async def test_create_graphiti_client_returns_none_when_no_password():
    """When NEO4J_PASSWORD is empty, factory returns None."""
    with patch("app.agent.graphiti_client.settings") as mock_settings:
        mock_settings.GRAPHITI_ENABLED = True
        mock_settings.NEO4J_PASSWORD = ""
        result = await create_graphiti_client()
        assert result is None


def test_entity_types_defined():
    """Custom entity types are Pydantic models with expected fields."""
    assert hasattr(ADHDEntityTypes, "Child")
    assert hasattr(ADHDEntityTypes, "Strategy")
    assert hasattr(ADHDEntityTypes, "EmotionalState")


def test_edge_types_defined():
    """Custom edge types are Pydantic models with expected fields."""
    assert hasattr(ADHDEdgeTypes, "TriedStrategy")
    assert hasattr(ADHDEdgeTypes, "HasChallenge")
    assert hasattr(ADHDEdgeTypes, "ExperiencedEmotion")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_graphiti_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.agent.graphiti_client'"

- [ ] **Step 3: Implement graphiti_client.py**

Create `app/agent/graphiti_client.py`:

```python
"""Graphiti client factory and custom entity/edge type definitions.

Creates a configured Graphiti instance with Gemini LLM and embedder,
custom ADHD coaching domain types, and Neo4j connection.
"""

import logging

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# --- Custom Entity Types ---

class Child(BaseModel):
    """A child being discussed in coaching sessions."""
    name: str = ""
    age: str = ""
    diagnosis_status: str = ""
    adhd_subtype: str = ""


class Parent(BaseModel):
    """The parent participating in coaching."""
    name: str = ""


class Strategy(BaseModel):
    """An ADHD management strategy mentioned or tried."""
    name: str = ""
    source: str = ""
    evidence_level: str = ""


class Challenge(BaseModel):
    """A specific difficulty area the family faces."""
    name: str = ""
    severity: str = ""


class EmotionalState(BaseModel):
    """A detected emotional state of the parent."""
    emotion: str = ""
    intensity: str = ""


class Goal(BaseModel):
    """A goal the parent has set."""
    description: str = ""
    status: str = ""


# --- Custom Edge Types ---

class TriedStrategy(BaseModel):
    """Parent tried a strategy with an outcome."""
    outcome: str = ""
    notes: str = ""


class HasChallenge(BaseModel):
    """Child has a specific challenge area."""
    context: str = ""


class ExperiencedEmotion(BaseModel):
    """Parent experienced an emotional state."""
    trigger: str = ""


class AddressesChallenge(BaseModel):
    """Strategy addresses a specific challenge."""
    effectiveness: str = ""


class SetGoal(BaseModel):
    """Parent set a specific goal."""
    motivation: str = ""


# --- Namespace containers for easy import ---

class ADHDEntityTypes:
    Child = Child
    Parent = Parent
    Strategy = Strategy
    Challenge = Challenge
    EmotionalState = EmotionalState
    Goal = Goal


class ADHDEdgeTypes:
    TriedStrategy = TriedStrategy
    HasChallenge = HasChallenge
    ExperiencedEmotion = ExperiencedEmotion
    AddressesChallenge = AddressesChallenge
    SetGoal = SetGoal


async def create_graphiti_client():
    """Create and initialize a Graphiti client with Gemini and Neo4j.

    Returns None if GRAPHITI_ENABLED is False or Neo4j credentials are missing.
    Falls back to None on connection failure (logged as error).
    """
    if not settings.GRAPHITI_ENABLED or not settings.NEO4J_PASSWORD:
        logger.info("Graphiti disabled (enabled=%s, has_password=%s)",
                     settings.GRAPHITI_ENABLED, bool(settings.NEO4J_PASSWORD))
        return None

    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
        from graphiti_core.embedder.gemini_embedder import GeminiEmbedder, GeminiEmbedderConfig

        client = Graphiti(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD,
            llm_client=GeminiClient(
                config=LLMConfig(
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GRAPHITI_LLM_MODEL,
                )
            ),
            embedder=GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=settings.GEMINI_API_KEY,
                    embedding_model=settings.GRAPHITI_EMBEDDING_MODEL,
                )
            ),
        )
        await client.build_indices_and_constraints()
        logger.info(
            "Graphiti client initialized (neo4j=%s, llm=%s, embedder=%s)",
            settings.NEO4J_URI, settings.GRAPHITI_LLM_MODEL, settings.GRAPHITI_EMBEDDING_MODEL,
        )
        return client

    except Exception as e:
        logger.error("Graphiti initialization failed — falling back to SQLite memory: %s", e)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_graphiti_client.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/graphiti_client.py tests/test_graphiti_client.py
git commit -m "Add Graphiti client factory with custom ADHD entity and edge types"
```

---

### Task 3: Rewrite MemoryManager

**Files:**
- Modify: `app/agent/memory.py`
- Rewrite: `tests/test_memory.py`

- [ ] **Step 1: Write tests for the new MemoryManager**

Rewrite `tests/test_memory.py` to test the new Graphiti-based memory manager. Key tests:

```python
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
    client.driver.execute_query = AsyncMock(return_value=MagicMock(records=[]))
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
        detail=pytest.approx({"error": "Neo4j down"}, abs=1),
    )


@pytest.mark.asyncio
async def test_sync_profile_fills_empty_fields(memory_manager, mock_graphiti, mock_store):
    """_sync_profile should update SQLite when graph has data and SQLite is empty."""
    record = {"child_name": "Alex", "child_age": "8", "diagnosis_status": None, "adhd_subtype": None}
    mock_graphiti.driver.execute_query = AsyncMock(
        return_value=MagicMock(records=[record])
    )
    await memory_manager._sync_profile("sess1", 1)
    mock_store.update_profile.assert_called_once_with(
        "sess1", child_name="Alex", child_age="8",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py -v`
Expected: FAIL (MemoryManager constructor signature changed)

- [ ] **Step 3: Rewrite app/agent/memory.py**

Replace the entire file content with:

```python
"""MemoryManager -- ingests conversation turns into Graphiti knowledge graph.

Runs after the response is sent to the parent. Non-blocking.
Replaces the previous rolling summary / fact extraction / episodic memory system.
"""

import asyncio
import logging
import time

from langsmith import traceable

from app.agent.store_protocol import SessionStoreBase
from app.config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Ingests conversation turns into Graphiti and syncs profile to SQLite."""

    def __init__(self, session_store: SessionStoreBase, graphiti_client=None, event_bus=None):
        self._store = session_store
        self._graphiti = graphiti_client
        self._event_bus = event_bus

    @traceable(name="memory.post_turn_tasks", run_type="chain")
    async def post_turn_tasks(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        assistant_response: str,
        user_id: int | None = None,
        **kwargs,
    ) -> None:
        """Ingest the conversation turn into Graphiti.

        Formats the turn as "Parent: ... Coach: ..." and calls
        graphiti.add_episode(). On failure, logs and drops -- the
        conversation text is always in SQLite regardless.
        """
        if not self._graphiti:
            return

        from graphiti_core.nodes import EpisodeType
        from datetime import datetime

        episode_body = f"Parent: {user_message}\nCoach: {assistant_response}"
        group_id = str(user_id) if user_id is not None else session_id

        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                self._graphiti.add_episode(
                    name=f"session_{session_id}_turn_{turn}",
                    episode_body=episode_body,
                    source=EpisodeType.message,
                    source_description=f"ADHD coaching session {session_id}",
                    reference_time=datetime.now(),
                    group_id=group_id,
                ),
                timeout=settings.GRAPHITI_INGESTION_TIMEOUT_S,
            )
            duration_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Graphiti episode ingested: session=%s turn=%d (%.0fms)",
                session_id, turn, duration_ms,
            )
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", "graphiti_episode_ingested", session_id, turn,
                    duration_ms=duration_ms,
                )

            # Sync discovered entities to SQLite profile
            if user_id is not None:
                await self._sync_profile(session_id, user_id)

        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Graphiti ingestion failed: session=%s turn=%d error=%s (%.0fms)",
                session_id, turn, e, duration_ms,
            )
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", "graphiti_ingestion_failed", session_id, turn,
                    detail={"error": str(e)},
                )

    async def _sync_profile(self, session_id: str, user_id: int) -> None:
        """Sync Graphiti-discovered facts into SQLite FamilyProfile.

        One-way: Graphiti -> SQLite. Only fills empty fields.
        """
        try:
            # Try direct entity query first; fall back to episode traversal
            # if entity nodes don't carry group_id.
            query = """
            MATCH (e {group_id: $group_id})-[:MENTIONS]->(c:Entity)
            WHERE any(label IN labels(c) WHERE label = 'Child' OR c.name CONTAINS 'child')
            RETURN c.name AS child_name, c.summary AS summary
            ORDER BY c.created_at DESC LIMIT 1
            """
            result = await self._graphiti.driver.execute_query(
                query, {"group_id": str(user_id)}
            )
            if not result.records:
                return

            record = dict(result.records[0])
            current_profile = (await self._store.get(session_id)).family_profile

            updates = {}
            # Parse entity name/summary for profile fields
            child_name = record.get("child_name")
            if child_name and not current_profile.child_name:
                updates["child_name"] = child_name

            if updates:
                await self._store.update_profile(session_id, **updates)
                logger.info("Profile synced from graph: session=%s fields=%s",
                            session_id, list(updates.keys()))
                if self._event_bus:
                    await self._event_bus.emit(
                        "memory", "graphiti_profile_synced", session_id, 0,
                        detail={"fields_updated": list(updates.keys())},
                    )
        except Exception as e:
            logger.warning("Profile sync failed (non-critical): %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/memory.py tests/test_memory.py
git commit -m "Rewrite MemoryManager to ingest turns via Graphiti"
```

---

### Task 4: Remove Deleted Methods from Store Protocol and Implementations

**Files:**
- Modify: `app/agent/store_protocol.py`
- Modify: `app/agent/sqlite_store.py`
- Modify: `app/agent/session_store.py`

- [ ] **Step 1: Remove abstract methods from store_protocol.py**

Remove these abstract method definitions from `SessionStoreBase`:
- `get_latest_summary`
- `save_summary`
- `add_episode`
- `get_recent_episodes`
- `get_episodes_with_ids`
- `add_episode_link`
- `get_episode_links`
- `get_user_summary`
- `save_user_summary`
- `get_user_episodes`
- `get_user_outcomes`

Also remove the unused schema imports: `EpisodeLink`, `EpisodicMemory`, `SessionSummary`, `UserSummary`.

- [ ] **Step 2: Remove implementations from sqlite_store.py**

Remove the method implementations for all 12 methods listed above. Search for each method name and delete the entire method body. Also remove unused imports (`EpisodeLink`, `EpisodicMemory`, `SessionSummary`, `UserSummary`).

- [ ] **Step 3: Remove implementations from session_store.py**

Same removal for the in-memory store. Search for each method name and delete.

- [ ] **Step 4: Run existing tests to check for breakage**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_sqlite_store.py tests/test_session_store.py -v`

Fix any test failures caused by tests that call the removed methods (these tests should be deleted or updated since they test deleted functionality).

- [ ] **Step 5: Commit**

```bash
git add app/agent/store_protocol.py app/agent/sqlite_store.py app/agent/session_store.py
git commit -m "Remove summary, episode, and cross-session methods from store protocol"
```

---

### Task 5: Add search_memory Tool

**Files:**
- Modify: `app/agent/tools.py:16-161`
- Modify: `tests/test_agent_tools.py`

- [ ] **Step 1: Write test for search_memory tool**

Add to `tests/test_agent_tools.py`:

```python
@pytest.mark.asyncio
async def test_search_memory_returns_formatted_facts(mock_graphiti):
    """search_memory should format Graphiti edges as readable facts."""
    from datetime import datetime
    from unittest.mock import MagicMock

    edge = MagicMock()
    edge.fact = "Parent tried timer system for homework"
    edge.valid_at = datetime(2026, 1, 15)
    edge.invalid_at = None
    edge.source_node_uuid = "uuid1"
    edge.target_node_uuid = "uuid2"
    mock_graphiti.search = AsyncMock(return_value=[edge])

    # Create tools with graphiti
    tools = create_tools(
        retriever=mock_retriever,
        session_store=mock_store,
        graphiti_client=mock_graphiti,
    )
    search_mem = next(t for t in tools if t.name == "search_memory")

    config = {"configurable": {"session_id": "test", "user_id": 1}}
    result = await search_mem.ainvoke({"query": "homework strategies"}, config=config)
    assert "timer system" in result
    mock_graphiti.search.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_tools.py::test_search_memory_returns_formatted_facts -v`
Expected: FAIL

- [ ] **Step 3: Add _get_user_id helper and search_memory tool to tools.py**

In `app/agent/tools.py`, add after `_get_session_id` (line 18):

```python
def _get_user_id(config: RunnableConfig) -> int | None:
    """Extract user_id from LangGraph config."""
    return config.get("configurable", {}).get("user_id")
```

Update `create_tools` signature to accept `graphiti_client`:

```python
def create_tools(
    retriever: HybridRetriever,
    session_store: SessionStoreBase,
    graphiti_client=None,
) -> list:
```

Add `search_memory` tool inside the factory (after existing tools, before the return):

```python
    if graphiti_client is not None:
        @tool
        async def search_memory(
            query: str,
            config: RunnableConfig = None,
        ) -> str:
            """Search conversation memory for what this family has shared,
            tried, or experienced. Use when you need to recall past
            discussions, strategy outcomes, emotional patterns, or
            family context from previous turns or sessions."""
            user_id = _get_user_id(config)
            group_ids = [str(user_id)] if user_id is not None else None

            try:
                edges = await graphiti_client.search(
                    query,
                    group_ids=group_ids,
                    num_results=settings.GRAPHITI_SEARCH_RESULTS,
                )
            except Exception as e:
                logger.error("search_memory failed: %s", e)
                return "Memory search is temporarily unavailable."

            if not edges:
                return "No relevant memories found for this query."

            lines = []
            for i, edge in enumerate(edges, 1):
                fact = getattr(edge, "fact", str(edge))
                valid_at = getattr(edge, "valid_at", None)
                invalid_at = getattr(edge, "invalid_at", None)
                status = ""
                if invalid_at is not None:
                    status = " [no longer active]"
                date_str = ""
                if valid_at:
                    date_str = f" (since {valid_at.strftime('%b %Y')})"
                lines.append(f"[{i}] {fact}{date_str}{status}")

            return f"{len(edges)} memories found:\n\n" + "\n".join(lines)

        all_tools.append(search_memory)
```

Make sure `all_tools` is the list that gets returned (check the existing return pattern).

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_tools.py::test_search_memory_returns_formatted_facts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/tools.py tests/test_agent_tools.py
git commit -m "Add search_memory tool backed by Graphiti"
```

---

### Task 6: Update Context Assembly (hooks.py)

**Files:**
- Modify: `app/agent/hooks.py:53-228`
- Modify: `tests/test_agent_hooks.py`

- [ ] **Step 1: Write test for graph-sourced context assembly**

Add to `tests/test_agent_hooks.py`:

```python
@pytest.mark.asyncio
async def test_prepare_context_queries_graphiti(mock_graphiti):
    """When graphiti_client is provided, context assembly should query it."""
    from datetime import datetime
    from unittest.mock import MagicMock

    edge = MagicMock()
    edge.fact = "Alex struggles with homework initiation"
    edge.valid_at = datetime(2026, 1, 1)
    edge.invalid_at = None
    mock_graphiti.search = AsyncMock(return_value=[edge])

    prepare = create_prepare_context(
        session_store=mock_store,
        graphiti_client=mock_graphiti,
    )
    # ... invoke and assert "homework initiation" appears in system prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_hooks.py -v`
Expected: FAIL (create_prepare_context doesn't accept graphiti_client yet)

- [ ] **Step 3: Update create_prepare_context to accept graphiti_client**

Modify `create_prepare_context` in `hooks.py`:

```python
def create_prepare_context(
    session_store: SessionStoreBase,
    event_bus=None,
    graphiti_client=None,
):
```

Inside `prepare_context`, replace the summary/episode loading block (lines 101-115) with:

```python
        # Load memory context from Graphiti (or empty if unavailable)
        memory_context = ""
        if graphiti_client is not None:
            user_id = state.get("user_id")
            group_ids = [str(user_id)] if user_id is not None else None
            try:
                # Semantic search keyed to what the parent just said
                last_user_text = ""
                user_messages = [m for m in messages if isinstance(m, HumanMessage)]
                if user_messages:
                    content = user_messages[-1].content
                    last_user_text = content if isinstance(content, str) else str(content)

                if last_user_text.strip():
                    import time as _time
                    t0 = _time.monotonic()
                    edges = await graphiti_client.search(
                        last_user_text,
                        group_ids=group_ids,
                        num_results=settings.GRAPHITI_CONTEXT_RESULTS,
                    )
                    duration_ms = (_time.monotonic() - t0) * 1000

                    valid_facts = []
                    for edge in edges:
                        fact = getattr(edge, "fact", "")
                        if fact and getattr(edge, "invalid_at", None) is None:
                            valid_at = getattr(edge, "valid_at", None)
                            date_note = f" (since {valid_at.strftime('%b %Y')})" if valid_at else ""
                            valid_facts.append(f"- {fact}{date_note}")

                    if valid_facts:
                        memory_context = "\n".join(valid_facts)

                    if event_bus:
                        await event_bus.emit(
                            "memory", "graphiti_search_completed", session_id, turn_count,
                            duration_ms=duration_ms,
                            detail={"query_len": len(last_user_text), "result_count": len(edges)},
                        )
            except Exception as e:
                logger.warning("Graphiti context search failed: %s", e)

        summary_text = memory_context or "This is the beginning of the conversation."
```

Remove the `<prior-sessions>` block (lines 136-154) -- graph handles cross-session context naturally via `group_id`.

Remove the `get_latest_summary` and `get_recent_episodes` calls.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_hooks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/hooks.py tests/test_agent_hooks.py
git commit -m "Update context assembly to query Graphiti for memory context"
```

---

### Task 7: Update System Prompt and Agent Graph

**Files:**
- Modify: `app/agent/prompts.py:19-157`
- Modify: `app/agent/graph.py:23-157`

- [ ] **Step 1: Update search_memory tool description in prompts.py**

In the `<tools>` section of `SYSTEM_PROMPT_TEMPLATE` (around line 62), add after the existing tool descriptions:

```
- **search_memory**: Search conversation memory for what this family has shared, tried, or experienced across sessions. Use when you need to recall past discussions, strategy outcomes, emotional patterns, or whether something was already discussed. Do NOT use for general ADHD knowledge -- use search_knowledge_base for that.
```

- [ ] **Step 2: Rename "Session Summary" section to "Memory Context"**

In `SYSTEM_PROMPT_TEMPLATE`, change:

```
## Session Summary

{session_summary}
```

to:

```
## Memory Context

{session_summary}
```

This is cosmetic -- the variable name stays `session_summary` in the code for backward compatibility, but now contains graph-sourced facts.

- [ ] **Step 3: No changes needed to graph.py**

The tools list is already passed from `main.py` -> `create_tools` -> `build_agent`. Since `search_memory` is added to the tools list inside `create_tools`, it automatically gets included in both `pro_react_agent` and `flash_react_agent`. Verify this by reading `graph.py` -- `tools` parameter flows directly to `create_react_agent(tools=tools, ...)`.

- [ ] **Step 4: Run prompt tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_prompts.py -v`
Expected: PASS (adjust any tests that assert exact prompt content for "Session Summary" -> "Memory Context")

- [ ] **Step 5: Commit**

```bash
git add app/agent/prompts.py
git commit -m "Add search_memory tool description to system prompt"
```

---

### Task 8: Update Orchestrator

**Files:**
- Modify: `app/agent/orchestrator.py:382-534`

- [ ] **Step 1: Add user_id to RunnableConfig configurable**

In `process()` method (line 401), change:

```python
        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }
```

to:

```python
        config = {
            "configurable": {"session_id": session_id, "user_id": user_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }
```

Note: `process()` does not currently accept `user_id`. Check if it's available from state or add it as a parameter. The streaming path (`process_stream`) already has `user_id` as a parameter (line 486).

For `process()`, add `user_id: int | None = None` parameter:

```python
async def process(self, message: str, session_id: str, user_id: int | None = None, attachment_ids: list[str] | None = None) -> StreamDonePayload:
```

- [ ] **Step 2: Add user_id to configurable in process_stream()**

In `process_stream()` (line 526), change:

```python
        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }
```

to:

```python
        config = {
            "configurable": {"session_id": session_id, "user_id": user_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }
```

- [ ] **Step 3: Remove end_of_session_tasks call**

In `process_stream()` (lines 505-513), delete the longitudinal summary block:

```python
        # Trigger longitudinal summary for returning users on first turn
        if turn == 1 and user_id is not None and self._memory:
            user_sessions = await self._session_store.get_all_sessions(user_id=user_id)
            previous = [s for s in user_sessions if s.session_id != session_id]
            if previous:
                self._track_task(
                    self._memory.end_of_session_tasks(user_id, previous[0].session_id),
                    "longitudinal_summary",
                )
```

- [ ] **Step 4: Pass user_id to post_turn_tasks in _fire_background_tasks**

In `_fire_background_tasks` (line 893), update the `post_turn_tasks` call to pass `user_id`. First add `user_id` as a parameter to `_fire_background_tasks`:

```python
    async def _fire_background_tasks(
        self,
        session_id: str,
        turn: int,
        message: str,
        response_text: str,
        tool_calls_made: list[dict],
        enriched: EnrichedTrace,
        force_summary: bool,
        total_ms: float,
        user_id: int | None = None,
    ) -> None:
```

Then update the call:

```python
            self._track_task(
                self._memory.post_turn_tasks(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=response_text,
                    user_id=user_id,
                ),
                "memory",
            )
```

Remove the `tool_calls` and `force_summary` kwargs since the new MemoryManager doesn't use them.

Update all callers of `_fire_background_tasks` to pass `user_id`.

- [ ] **Step 5: Add search_memory status in the streaming tool handler**

In `process_stream()`, in the `on_tool_start` handler (around line 588), add a case for `search_memory`:

```python
                        elif tool_name == "search_memory":
                            query = tool_input.get("query", "")
                            logger.info("[agent] tool: search_memory(%.70s)", query)
                            status = f"Recalling '{query}'..."
```

- [ ] **Step 6: Run orchestrator tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_agent_orchestrator.py -v`
Expected: PASS (fix any failures from signature changes)

- [ ] **Step 7: Commit**

```bash
git add app/agent/orchestrator.py
git commit -m "Wire user_id through orchestrator, remove end_of_session_tasks"
```

---

### Task 9: Wire Graphiti into main.py

**Files:**
- Modify: `app/main.py:33-250`

- [ ] **Step 1: Initialize Graphiti client in build_dependencies**

After step 6 (event bus initialization, around line 145), add:

```python
    # 6b. Initialize Graphiti memory graph (optional)
    graphiti_client = None
    if settings.GRAPHITI_ENABLED and settings.NEO4J_PASSWORD and settings.GEMINI_API_KEY:
        from app.agent.graphiti_client import create_graphiti_client
        try:
            graphiti_client = await create_graphiti_client()
            if graphiti_client:
                logger.info("Graphiti memory graph initialized")
        except Exception as e:
            logger.error("Graphiti init failed — continuing without graph memory: %s", e)
```

- [ ] **Step 2: Pass graphiti_client to create_tools**

Change line 161:

```python
    tools = create_tools(retriever=retriever, session_store=session_store)
```

to:

```python
    tools = create_tools(
        retriever=retriever,
        session_store=session_store,
        graphiti_client=graphiti_client,
    )
```

- [ ] **Step 3: Pass graphiti_client to create_prepare_context**

Change lines 165-168:

```python
    prepare_context = create_prepare_context(
        session_store=session_store,
        event_bus=event_bus,
    )
```

to:

```python
    prepare_context = create_prepare_context(
        session_store=session_store,
        event_bus=event_bus,
        graphiti_client=graphiti_client,
    )
```

- [ ] **Step 4: Pass graphiti_client to MemoryManager**

Change lines 172-174:

```python
    memory_manager = MemoryManager(
        session_store=session_store, gemini_client=gemini, event_bus=event_bus,
    ) if gemini else None
```

to:

```python
    memory_manager = MemoryManager(
        session_store=session_store,
        graphiti_client=graphiti_client,
        event_bus=event_bus,
    ) if (graphiti_client or gemini) else None
```

- [ ] **Step 5: Add Graphiti shutdown in lifespan**

In the `lifespan` function (around line 244), add before the `db_conn` close:

```python
    graphiti_client = getattr(app.state, "graphiti_client", None)
    if graphiti_client:
        await graphiti_client.close()
```

Also store it in app.state by adding to the `deps` dict:

```python
    return {
        "orchestrator": orchestrator,
        "session_store": session_store,
        "event_bus": event_bus,
        "analyzer": analyzer,
        "db_conn": db_conn,
        "knowledge_base": store,
        "graphiti_client": graphiti_client,
    }
```

- [ ] **Step 6: Commit**

```bash
git add app/main.py
git commit -m "Wire Graphiti client into app initialization and shutdown"
```

---

### Task 10: Run Full Test Suite and Fix Breakage

**Files:**
- Various test files

- [ ] **Step 1: Run the full test suite (excluding integration tests)**

Run: `./adhd312/Scripts/python.exe -m pytest tests/ -v -m "not integration" --ignore=tests/test_e2e_conversations.py --ignore=tests/test_integration_guardrails.py --ignore=tests/test_integration_llm.py --ignore=tests/test_integration_memory.py --ignore=tests/test_integration_pipeline.py --ignore=tests/test_integration_rag.py`

- [ ] **Step 2: Fix any test failures**

Common expected failures:
- Tests that call removed store methods (`get_latest_summary`, `save_summary`, `add_episode`, etc.) -- update mocks or delete tests
- Tests that construct `MemoryManager` with old signature (`gemini_client=`) -- update to `graphiti_client=`
- Tests that assert old event bus events (`summary_updated`, `facts_extracted`) -- update to new events
- `test_cross_session.py` may reference `end_of_session_tasks` -- update or remove

- [ ] **Step 3: Verify all tests pass**

Run the full suite again. All non-integration tests should pass.

- [ ] **Step 4: Commit all fixes**

```bash
git add -A
git commit -m "Fix test suite for Graphiti memory migration"
```

---

### Task 11: Manual Smoke Test (Optional, Requires API Keys)

> **Do NOT run this without asking the user first.** This requires GEMINI_API_KEY and Neo4j credentials.

- [ ] **Step 1: Start the app**

Run: `./adhd312/Scripts/python.exe -m uvicorn app.main:app --reload`

Check logs for:
- "Graphiti client initialized" or "Graphiti disabled"
- "Graphiti memory graph initialized"
- No startup errors

- [ ] **Step 2: Send a test message and check logs**

Send a chat message via the frontend or curl. Verify:
- Response streams normally
- Logs show "Graphiti episode ingested" (background)
- No errors in the Graphiti ingestion path

- [ ] **Step 3: Verify search_memory works**

Send a message that would trigger the agent to use search_memory (e.g., "What have we discussed before?"). Check that the tool is called and returns results (or "No relevant memories" on a fresh graph).
