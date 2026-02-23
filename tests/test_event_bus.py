"""Tests for EventBus — async ring buffer, filtering, per-session isolation."""

import pytest

from app.agent.event_bus import EventBus


class TestEventBus:

    @pytest.mark.asyncio
    async def test_emit_and_get(self):
        bus = EventBus(buffer_size=10)
        await bus.emit("agent", "turn_start", "s1", turn=1)
        await bus.emit("guardrails", "input_check_passed", "s1", turn=1)

        events = await bus.get_events("s1")
        assert len(events) == 2
        assert events[0].category == "agent"
        assert events[1].category == "guardrails"

    @pytest.mark.asyncio
    async def test_ring_buffer_size_limit(self):
        bus = EventBus(buffer_size=3)
        for i in range(5):
            await bus.emit("agent", f"event_{i}", "s1")

        events = await bus.get_events("s1")
        assert len(events) == 3
        # Should keep the last 3 events
        assert events[0].event_type == "event_2"
        assert events[1].event_type == "event_3"
        assert events[2].event_type == "event_4"

    @pytest.mark.asyncio
    async def test_category_filtering(self):
        bus = EventBus()
        await bus.emit("agent", "start", "s1")
        await bus.emit("guardrails", "check", "s1")
        await bus.emit("memory", "summary", "s1")
        await bus.emit("guardrails", "output", "s1")

        guardrail_events = await bus.get_events("s1", category="guardrails")
        assert len(guardrail_events) == 2
        assert all(e.category == "guardrails" for e in guardrail_events)

    @pytest.mark.asyncio
    async def test_level_filtering(self):
        bus = EventBus()
        await bus.emit("agent", "info_event", "s1", level="info")
        await bus.emit("error", "error_event", "s1", level="error")
        await bus.emit("agent", "warning_event", "s1", level="warning")

        errors = await bus.get_events("s1", level="error")
        assert len(errors) == 1
        assert errors[0].event_type == "error_event"

    @pytest.mark.asyncio
    async def test_per_session_isolation(self):
        bus = EventBus()
        await bus.emit("agent", "event_a", "session_a")
        await bus.emit("agent", "event_b", "session_b")

        assert len(await bus.get_events("session_a")) == 1
        assert len(await bus.get_events("session_b")) == 1
        assert len(await bus.get_events("session_c")) == 0

    @pytest.mark.asyncio
    async def test_get_all_session_ids(self):
        bus = EventBus()
        await bus.emit("agent", "start", "s1")
        await bus.emit("agent", "start", "s2")
        await bus.emit("agent", "start", "s3")

        ids = await bus.get_all_session_ids()
        assert set(ids) == {"s1", "s2", "s3"}

    @pytest.mark.asyncio
    async def test_event_has_timestamp(self):
        bus = EventBus()
        await bus.emit("agent", "test", "s1")
        events = await bus.get_events("s1")
        assert events[0].timestamp != ""

    @pytest.mark.asyncio
    async def test_empty_session(self):
        bus = EventBus()
        assert await bus.get_events("nonexistent") == []

    @pytest.mark.asyncio
    async def test_detail_dict(self):
        bus = EventBus()
        await bus.emit("agent", "test", "s1", detail={"key": "value"})
        events = await bus.get_events("s1")
        assert events[0].detail == {"key": "value"}
