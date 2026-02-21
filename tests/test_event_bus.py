"""Tests for EventBus — ring buffer, filtering, thread safety."""

import threading

from app.agent.event_bus import EventBus


class TestEventBus:

    def test_emit_and_get(self):
        bus = EventBus(buffer_size=10)
        bus.emit("agent", "turn_start", "s1", turn=1)
        bus.emit("guardrails", "input_check_passed", "s1", turn=1)

        events = bus.get_events("s1")
        assert len(events) == 2
        assert events[0].category == "agent"
        assert events[1].category == "guardrails"

    def test_ring_buffer_size_limit(self):
        bus = EventBus(buffer_size=3)
        for i in range(5):
            bus.emit("agent", f"event_{i}", "s1")

        events = bus.get_events("s1")
        assert len(events) == 3
        # Should keep the last 3 events
        assert events[0].event_type == "event_2"
        assert events[1].event_type == "event_3"
        assert events[2].event_type == "event_4"

    def test_category_filtering(self):
        bus = EventBus()
        bus.emit("agent", "start", "s1")
        bus.emit("guardrails", "check", "s1")
        bus.emit("memory", "summary", "s1")
        bus.emit("guardrails", "output", "s1")

        guardrail_events = bus.get_events("s1", category="guardrails")
        assert len(guardrail_events) == 2
        assert all(e.category == "guardrails" for e in guardrail_events)

    def test_level_filtering(self):
        bus = EventBus()
        bus.emit("agent", "info_event", "s1", level="info")
        bus.emit("error", "error_event", "s1", level="error")
        bus.emit("agent", "warning_event", "s1", level="warning")

        errors = bus.get_events("s1", level="error")
        assert len(errors) == 1
        assert errors[0].event_type == "error_event"

    def test_per_session_isolation(self):
        bus = EventBus()
        bus.emit("agent", "event_a", "session_a")
        bus.emit("agent", "event_b", "session_b")

        assert len(bus.get_events("session_a")) == 1
        assert len(bus.get_events("session_b")) == 1
        assert len(bus.get_events("session_c")) == 0

    def test_get_all_session_ids(self):
        bus = EventBus()
        bus.emit("agent", "start", "s1")
        bus.emit("agent", "start", "s2")
        bus.emit("agent", "start", "s3")

        ids = bus.get_all_session_ids()
        assert set(ids) == {"s1", "s2", "s3"}

    def test_thread_safety(self):
        bus = EventBus(buffer_size=1000)
        errors = []

        def emit_events(thread_id):
            try:
                for i in range(100):
                    bus.emit("agent", f"event_{thread_id}_{i}", "shared")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_events, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        events = bus.get_events("shared")
        assert len(events) == 1000  # 10 threads * 100 events

    def test_event_has_timestamp(self):
        bus = EventBus()
        bus.emit("agent", "test", "s1")
        events = bus.get_events("s1")
        assert events[0].timestamp != ""

    def test_empty_session(self):
        bus = EventBus()
        assert bus.get_events("nonexistent") == []

    def test_detail_dict(self):
        bus = EventBus()
        bus.emit("agent", "test", "s1", detail={"key": "value"})
        events = bus.get_events("s1")
        assert events[0].detail == {"key": "value"}
