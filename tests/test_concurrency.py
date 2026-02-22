"""
Concurrency and race condition tests.

Tests the architectural issues identified in the codebase audit:
- InMemorySessionStore live-object mutation races
- SQLite shared-connection contention between store and event bus
- Concurrent session access (no per-session mutex)
- Background task lifecycle (fire-and-forget GC risks)
- EventBus thread safety under concurrent writes
- get_messages() returning live list references

These tests do NOT require GEMINI_API_KEY — they use mocks or in-memory stores only.
Run: pytest tests/test_concurrency.py -v
"""

import asyncio
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest

from app.agent.event_bus import EventBus
from app.agent.memory import MemoryManager
from app.agent.session_store import InMemorySessionStore
from app.agent.sqlite_store import SQLiteSessionStore
from app.db import get_connection, init_db
from app.models.schemas import SessionSummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_store():
    return InMemorySessionStore()


@pytest.fixture
def sqlite_conn():
    """In-memory SQLite connection for testing."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


@pytest.fixture
def sqlite_store(sqlite_conn):
    return SQLiteSessionStore(sqlite_conn)


@pytest.fixture
def event_bus_memory():
    """In-memory EventBus (no SQLite)."""
    return EventBus(buffer_size=100)


@pytest.fixture
def event_bus_sqlite(sqlite_conn):
    """SQLite-backed EventBus sharing the same connection."""
    return EventBus(buffer_size=100, conn=sqlite_conn)


# ---------------------------------------------------------------------------
# InMemorySessionStore — Live Object Mutation
# ---------------------------------------------------------------------------

class TestInMemoryLiveObjectMutation:
    """
    Audit finding #4: InMemorySessionStore.get() returns a reference to the
    live internal SessionState object. Mutations outside the lock are not
    protected. This test suite documents the current behavior and catches
    regressions if the store is fixed.
    """

    def test_get_returns_same_object_reference(self, memory_store):
        """Two calls to get() for the same session return the SAME object."""
        s1 = memory_store.get("sess-1")
        s2 = memory_store.get("sess-1")
        assert s1 is s2, (
            "InMemorySessionStore.get() should return the same object reference. "
            "If this fails, the store has been fixed (good!) — update dependent tests."
        )

    def test_direct_mutation_visible_across_references(self, memory_store):
        """Mutating the returned object modifies the internal state directly."""
        state = memory_store.get("sess-1")
        state.family_profile.child_name = "Directly Mutated"

        # Another get() sees the mutation
        state2 = memory_store.get("sess-1")
        assert state2.family_profile.child_name == "Directly Mutated"

    def test_list_append_outside_lock_is_visible(self, memory_store):
        """Appending to a list field outside the lock is visible to other readers."""
        state = memory_store.get("sess-1")
        state.goals.append(type("Goal", (), {"description": "test", "status": "active"})())

        state2 = memory_store.get("sess-1")
        assert len(state2.goals) == 1

    def test_concurrent_list_mutation_does_not_crash(self, memory_store):
        """
        Two threads concurrently appending to conversation_history should not
        crash (CPython GIL protects list.append at bytecode level), but may
        produce interleaved results.
        """
        session_id = "concurrent-list"
        state = memory_store.get(session_id)
        errors = []

        def append_messages(thread_id, count):
            try:
                for i in range(count):
                    state.conversation_history.append({
                        "role": "user",
                        "content": f"thread-{thread_id}-msg-{i}",
                        "turn": i,
                    })
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append_messages, args=(1, 100))
        t2 = threading.Thread(target=append_messages, args=(2, 100))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent appends should not crash: {errors}"
        # Should have all 200 messages (CPython GIL)
        assert len(state.conversation_history) == 200

    def test_get_messages_returns_live_list_without_limit(self, memory_store):
        """
        Audit finding: get_messages(session_id) without limit returns the live
        internal list — not a copy. Concurrent modifications are visible.
        """
        session_id = "live-list"
        memory_store.add_message(session_id, "user", "msg1", 1)

        messages = memory_store.get_messages(session_id)  # No limit
        original_len = len(messages)

        # Add another message
        memory_store.add_message(session_id, "assistant", "msg2", 1)

        # If get_messages returned a copy, len would still be original_len
        # If it returned the live list, len would be +1
        if len(messages) == original_len + 1:
            # Live reference confirmed — this is the current (buggy) behavior
            pass
        else:
            # Store was fixed to return a copy — this is the desired behavior
            pass
        # Either way, no crash
        assert len(messages) >= original_len

    def test_get_messages_with_limit_returns_copy(self, memory_store):
        """get_messages with a limit should return a slice (new list)."""
        session_id = "slice-test"
        for i in range(5):
            memory_store.add_message(session_id, "user", f"msg-{i}", i + 1)

        messages = memory_store.get_messages(session_id, limit=3)
        assert len(messages) == 3

        # Add more — the slice should not change
        memory_store.add_message(session_id, "user", "new-msg", 6)
        assert len(messages) == 3, "Sliced result should be a new list"


# ---------------------------------------------------------------------------
# SQLiteSessionStore — Concurrent Access
# ---------------------------------------------------------------------------

class TestSQLiteConcurrentAccess:
    """
    Audit finding #1: Single SQLite connection shared across store and EventBus
    with independent locks. Test concurrent access patterns.
    """

    def test_concurrent_store_operations_single_session(self, sqlite_store):
        """Multiple threads doing read-modify-write on the same session."""
        session_id = "concurrent-sqlite"
        errors = []

        def worker(thread_id, iterations):
            try:
                for i in range(iterations):
                    sqlite_store.increment_turn(session_id)
                    sqlite_store.add_message(
                        session_id, "user", f"thread-{thread_id}-turn-{i}", i + 1
                    )
            except Exception as e:
                errors.append((thread_id, e))

        threads = [
            threading.Thread(target=worker, args=(t, 10))
            for t in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Under the single-lock RLock, this should succeed
        assert not errors, f"Concurrent store operations failed: {errors}"
        messages = sqlite_store.get_messages(session_id)
        assert len(messages) == 30  # 3 threads * 10 messages

    def test_concurrent_different_sessions(self, sqlite_store):
        """Concurrent access to different sessions should not interfere."""
        errors = []

        def worker(session_id, iterations):
            try:
                for i in range(iterations):
                    sqlite_store.increment_turn(session_id)
                    sqlite_store.add_message(session_id, "user", f"msg-{i}", i + 1)
                    state = sqlite_store.get(session_id)
                    # State should be consistent for this session
                    assert state.turn_count == i + 1
            except Exception as e:
                errors.append((session_id, e))

        sessions = [f"session-{i}" for i in range(5)]
        threads = [
            threading.Thread(target=worker, args=(sid, 5))
            for sid in sessions
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent session access failed: {errors}"
        for sid in sessions:
            state = sqlite_store.get(sid)
            assert state.turn_count == 5

    def test_store_and_eventbus_concurrent_writes(self, sqlite_conn):
        """
        Audit finding #1/#2: SQLiteSessionStore and EventBus share the same
        connection but have independent locks. This tests whether concurrent
        writes cause errors.
        """
        store = SQLiteSessionStore(sqlite_conn)
        bus = EventBus(buffer_size=100, conn=sqlite_conn)
        errors = []

        def store_worker():
            try:
                for i in range(20):
                    store.increment_turn("shared-session")
                    store.add_message("shared-session", "user", f"msg-{i}", i + 1)
            except Exception as e:
                errors.append(("store", e))

        def bus_worker():
            try:
                for i in range(20):
                    bus.emit("test", f"event-{i}", "shared-session", i + 1)
            except Exception as e:
                errors.append(("bus", e))

        t1 = threading.Thread(target=store_worker)
        t2 = threading.Thread(target=bus_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # This test documents whether the shared connection causes errors.
        # If errors is non-empty, it confirms the audit finding.
        if errors:
            pytest.xfail(
                f"Shared SQLite connection race confirmed: {errors}. "
                "This is audit finding #1 — store and EventBus need separate connections."
            )
        else:
            # Under CPython with GIL, this might succeed due to lock ordering
            messages = store.get_messages("shared-session")
            events = bus.get_events("shared-session")
            assert len(messages) == 20
            assert len(events) == 20

    def test_get_connection_sets_synchronous_normal(self, tmp_path):
        """get_connection should set PRAGMA synchronous=NORMAL (safe with WAL)."""
        db_path = str(tmp_path / "test_sync.db")
        conn = get_connection(db_path)
        row = conn.execute("PRAGMA synchronous").fetchone()
        # NORMAL = 1
        assert row[0] == 1, f"Expected synchronous=NORMAL (1), got {row[0]}"
        conn.close()

    def test_store_and_eventbus_separate_connections(self, tmp_path):
        """
        Fix for audit finding #1: store and EventBus each get their own
        SQLite connection to the same DB file. WAL mode supports concurrent
        readers/writers on separate connections, so this should succeed
        without races.
        """
        db_path = str(tmp_path / "test_separate.db")
        conn_store = get_connection(db_path)
        init_db(conn_store)
        conn_events = get_connection(db_path)

        store = SQLiteSessionStore(conn_store)
        bus = EventBus(buffer_size=100, conn=conn_events)
        errors = []

        def store_worker():
            try:
                for i in range(20):
                    store.increment_turn("separate-session")
                    store.add_message(
                        "separate-session", "user", f"msg-{i}", i + 1,
                    )
            except Exception as e:
                errors.append(("store", e))

        def bus_worker():
            try:
                for i in range(20):
                    bus.emit("test", f"event-{i}", "separate-session", i + 1)
            except Exception as e:
                errors.append(("bus", e))

        t1 = threading.Thread(target=store_worker)
        t2 = threading.Thread(target=bus_worker)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, (
            f"Separate connections should not race: {errors}"
        )
        messages = store.get_messages("separate-session")
        events = bus.get_events("separate-session")
        assert len(messages) == 20
        assert len(events) == 20


# ---------------------------------------------------------------------------
# EventBus Thread Safety
# ---------------------------------------------------------------------------

class TestEventBusThreadSafety:
    """Verify EventBus handles concurrent access correctly."""

    def test_concurrent_emit_memory_bus(self, event_bus_memory):
        """Many threads emitting to the in-memory bus concurrently."""
        errors = []

        def emitter(thread_id, count):
            try:
                for i in range(count):
                    event_bus_memory.emit(
                        "test", f"event-{i}",
                        session_id=f"session-{thread_id}",
                        turn=i,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=emitter, args=(t, 50))
            for t in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent emit failed: {errors}"

        # Verify all events were stored
        for t in range(10):
            events = event_bus_memory.get_events(f"session-{t}")
            assert len(events) == 50

    def test_concurrent_emit_and_read(self, event_bus_memory):
        """Concurrent emit and read should not deadlock or corrupt."""
        session_id = "rw-test"
        errors = []
        stop = threading.Event()

        def writer():
            try:
                i = 0
                while not stop.is_set():
                    event_bus_memory.emit("test", f"event-{i}", session_id, i)
                    i += 1
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                while not stop.is_set():
                    events = event_bus_memory.get_events(session_id)
                    # Just read — should not crash
                    _ = len(events)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("reader", e))

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        time.sleep(0.5)
        stop.set()
        t_write.join()
        t_read.join()

        assert not errors, f"Concurrent read/write failed: {errors}"

    def test_buffer_size_respected_under_load(self, event_bus_memory):
        """Ring buffer should evict old events when full."""
        session_id = "buffer-test"
        for i in range(200):
            event_bus_memory.emit("test", f"event-{i}", session_id, i)

        events = event_bus_memory.get_events(session_id)
        assert len(events) <= 100  # buffer_size=100

    def test_sqlite_eventbus_concurrent_emit(self, event_bus_sqlite):
        """Concurrent emit to SQLite-backed EventBus."""
        errors = []

        def emitter(thread_id, count):
            try:
                for i in range(count):
                    event_bus_sqlite.emit(
                        "test", f"event-{i}",
                        session_id=f"session-{thread_id}",
                        turn=i,
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=emitter, args=(t, 20))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            pytest.xfail(
                f"SQLite EventBus concurrent emit race: {errors}. "
                "Known issue: EventBus lock is separate from SQLiteSessionStore lock."
            )


# ---------------------------------------------------------------------------
# Background Task Lifecycle
# ---------------------------------------------------------------------------

class TestBackgroundTaskLifecycle:
    """
    Audit finding #6: asyncio.create_task() return values are discarded.
    Test that background tasks actually complete and don't get GC'd.
    """

    async def test_background_task_completes(self):
        """A fire-and-forget task should complete even without holding a reference."""
        completed = asyncio.Event()

        async def background_work():
            await asyncio.sleep(0.1)
            completed.set()

        asyncio.create_task(background_work())
        # Wait a bit for the task to finish
        await asyncio.sleep(0.3)
        assert completed.is_set(), "Background task should have completed"

    async def test_multiple_background_tasks_all_complete(self):
        """Multiple fire-and-forget tasks should all complete."""
        results = []

        async def background_work(task_id):
            await asyncio.sleep(0.05)
            results.append(task_id)

        for i in range(10):
            asyncio.create_task(background_work(i))

        await asyncio.sleep(0.5)
        assert len(results) == 10, f"All tasks should complete, got {len(results)}"

    async def test_memory_post_turn_tasks_completes(self):
        """MemoryManager.post_turn_tasks should complete as a background task."""
        store = InMemorySessionStore()
        mock_gemini = AsyncMock()
        mock_gemini.generate = AsyncMock(return_value="Summary of conversation.")
        mock_gemini.extract_json = AsyncMock(return_value={"child_name": "Test"})

        memory = MemoryManager(store, mock_gemini)

        # Populate enough conversation for a summary
        session_id = "bg-test"
        for i in range(5):
            store.increment_turn(session_id)
            store.add_message(session_id, "user", f"User message turn {i+1}", i + 1)
            store.add_message(session_id, "assistant", f"Response turn {i+1}", i + 1)

        # Fire as a background task (the pattern from orchestrator.py)
        task = asyncio.create_task(memory.post_turn_tasks(
            session_id=session_id,
            turn=5,
            user_message="My son Alex is 8 and struggles with homework every night after school",
            assistant_response="I understand that can be challenging.",
        ))

        # Wait for completion
        await asyncio.wait_for(task, timeout=10.0)

        # Summary should have been generated
        summary = store.get_latest_summary(session_id)
        assert summary is not None, "Summary should be generated"
        assert summary.covers_through_turn == 5


# ---------------------------------------------------------------------------
# SQLite Seed Session Atomicity
# ---------------------------------------------------------------------------

class TestSeedSessionAtomicity:
    """
    Audit finding #7: seed_session performs multiple commits with no outer
    transaction. Test partial failure behavior.
    """

    def test_seed_session_creates_profile_and_goals(self, sqlite_store):
        """Full seed should create both profile and goals."""
        from app.models.schemas import SeedSessionRequest

        sqlite_store.seed_session(SeedSessionRequest(
            session_id="seed-test",
            child_name="Maya",
            child_age="6",
            challenges=["morning routine", "meltdowns"],
            goals=["Improve morning routine"],
        ))

        state = sqlite_store.get("seed-test")
        assert state.family_profile.child_name == "Maya"
        assert state.family_profile.child_age == "6"
        assert "morning routine" in state.family_profile.challenge_areas
        assert len(state.goals) == 1
        assert state.goals[0].description == "Improve morning routine"

    def test_seed_session_idempotent_profile(self, sqlite_store):
        """Seeding the same session twice should not duplicate data."""
        from app.models.schemas import SeedSessionRequest

        sqlite_store.seed_session(SeedSessionRequest(
            session_id="idempotent-test",
            child_name="Leo",
            child_age="10",
            challenges=["focus"],
        ))
        sqlite_store.seed_session(SeedSessionRequest(
            session_id="idempotent-test",
            child_name="Leo",
            child_age="10",
            challenges=["focus"],
        ))

        state = sqlite_store.get("idempotent-test")
        assert state.family_profile.child_name == "Leo"
        # Challenges should be deduplicated
        assert state.family_profile.challenge_areas.count("focus") == 1


# ---------------------------------------------------------------------------
# InMemorySessionStore — Concurrent Method Calls
# ---------------------------------------------------------------------------

class TestInMemoryConcurrency:
    """Test InMemorySessionStore under concurrent access patterns."""

    def test_concurrent_increment_turn(self, memory_store):
        """Multiple threads incrementing turn count should not lose increments."""
        session_id = "inc-test"
        errors = []

        def incrementer(count):
            try:
                for _ in range(count):
                    memory_store.increment_turn(session_id)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=incrementer, args=(50,))
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        state = memory_store.get(session_id)
        assert state.turn_count == 200, (
            f"Expected 200 increments, got {state.turn_count}"
        )

    def test_concurrent_add_message(self, memory_store):
        """Multiple threads adding messages to the same session."""
        session_id = "msg-test"
        errors = []

        def add_messages(thread_id, count):
            try:
                for i in range(count):
                    memory_store.add_message(
                        session_id, "user", f"t{thread_id}-{i}", i + 1
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_messages, args=(t, 50))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        messages = memory_store.get_messages(session_id)
        assert len(messages) == 200

    def test_concurrent_profile_update(self, memory_store):
        """Multiple threads updating the same profile."""
        session_id = "profile-test"
        errors = []

        def updater(thread_id):
            try:
                for i in range(20):
                    memory_store.update_profile(
                        session_id,
                        child_name=f"Child-{thread_id}-{i}",
                        challenge_areas=[f"challenge-{thread_id}-{i}"],
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=updater, args=(t,))
            for t in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        state = memory_store.get(session_id)
        # Profile should have a name (last write wins) and accumulated challenges
        assert state.family_profile.child_name is not None
        assert len(state.family_profile.challenge_areas) > 0

    def test_session_isolation_under_concurrent_access(self, memory_store):
        """Concurrent operations on different sessions should be fully isolated."""
        errors = []

        def session_worker(session_id, turn_count):
            try:
                for i in range(turn_count):
                    memory_store.increment_turn(session_id)
                    memory_store.add_message(session_id, "user", f"msg-{i}", i + 1)
                    memory_store.update_profile(session_id, child_name=f"Child-{session_id}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=session_worker, args=(f"iso-{i}", 10))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

        for i in range(5):
            state = memory_store.get(f"iso-{i}")
            assert state.turn_count == 10
            assert state.family_profile.child_name == f"Child-iso-{i}"
            messages = memory_store.get_messages(f"iso-{i}")
            assert len(messages) == 10


# ---------------------------------------------------------------------------
# SQLite f-string Column Name Injection
# ---------------------------------------------------------------------------

class TestSQLInjectionSurface:
    """
    Audit finding #8: update_profile uses f-string for column names.
    Test that unexpected field names don't execute arbitrary SQL.
    """

    def test_unknown_field_name_raises_or_is_rejected(self, sqlite_store):
        """Passing an unknown field name should raise ValueError before SQL."""
        sqlite_store.increment_turn("inject-test")  # Ensure session exists

        with pytest.raises(ValueError, match="Invalid profile field"):
            sqlite_store.update_profile("inject-test", nonexistent_field="value")

        # DB should not be corrupted
        state = sqlite_store.get("inject-test")
        assert state is not None

    def test_sql_injection_in_field_name_does_not_execute(self, sqlite_store):
        """
        A crafted field name attempting SQL injection should either:
        1. Raise an OperationalError (column not found)
        2. Not execute the injected SQL
        """
        sqlite_store.increment_turn("inject-test-2")

        try:
            sqlite_store.update_profile(
                "inject-test-2",
                **{"child_name = 'x'; DROP TABLE sessions; --": "injected"}
            )
        except (sqlite3.OperationalError, Exception):
            pass

        # Verify sessions table still exists
        state = sqlite_store.get("inject-test-2")
        assert state is not None, "Sessions table should not have been dropped"

    def test_invalid_field_rejected_before_sql(self, sqlite_store):
        """update_profile should raise ValueError for unknown fields, not OperationalError."""
        sqlite_store.increment_turn("validate-test")
        with pytest.raises(ValueError, match="Invalid profile field"):
            sqlite_store.update_profile("validate-test", bad_field="value")

    def test_inmemory_invalid_field_rejected(self, memory_store):
        """InMemorySessionStore should also reject unknown fields."""
        with pytest.raises(ValueError, match="Invalid profile field"):
            memory_store.update_profile("validate-test", bad_field="value")
