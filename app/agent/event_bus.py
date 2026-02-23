"""EventBus — async structured event store per session.

Uses aiosqlite for persistence when available, falls back to in-memory ring buffer.
Events survive server restarts when SQLite is enabled.
"""

import json
import logging
from collections import deque
from datetime import datetime, timezone

import aiosqlite

from app.models.schemas import ObservabilityEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Async per-session event store with optional aiosqlite persistence."""

    def __init__(self, buffer_size: int = 200, conn: aiosqlite.Connection | None = None):
        self._buffer_size = buffer_size
        self._conn = conn
        # In-memory buffer used as cache and fallback when no DB
        self._buffers: dict[str, deque[ObservabilityEvent]] = {}

    async def emit(
        self,
        category: str,
        event_type: str,
        session_id: str = "",
        turn: int = 0,
        duration_ms: float = 0.0,
        detail: dict | None = None,
        level: str = "info",
    ) -> None:
        """Emit a structured event. Persists to SQLite if available."""
        ts = datetime.now(timezone.utc).isoformat()
        event = ObservabilityEvent(
            category=category,
            event_type=event_type,
            session_id=session_id,
            turn=turn,
            timestamp=ts,
            duration_ms=duration_ms,
            detail=detail or {},
            level=level,
        )

        # Always keep in memory for fast reads
        if session_id not in self._buffers:
            self._buffers[session_id] = deque(maxlen=self._buffer_size)
        self._buffers[session_id].append(event)

        # Persist to SQLite
        if self._conn:
            try:
                await self._conn.execute(
                    "INSERT INTO observability_events (session_id, category, event_type, turn, timestamp, duration_ms, detail_json, level) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (session_id, category, event_type, turn, ts, duration_ms, json.dumps(detail or {}), level),
                )
            except Exception as e:
                logger.warning("Failed to persist event to SQLite: %s", e)

    async def get_events(
        self,
        session_id: str,
        category: str | None = None,
        level: str | None = None,
    ) -> list[ObservabilityEvent]:
        """Return events for a session, optionally filtered by category/level."""
        if self._conn:
            return await self._get_events_from_db(session_id, category, level)

        # Fallback to in-memory
        events = list(self._buffers.get(session_id, []))
        if category:
            events = [e for e in events if e.category == category]
        if level:
            events = [e for e in events if e.level == level]
        return events

    async def _get_events_from_db(
        self,
        session_id: str,
        category: str | None = None,
        level: str | None = None,
    ) -> list[ObservabilityEvent]:
        """Read events from SQLite with optional filters."""
        query = "SELECT category, event_type, session_id, turn, timestamp, duration_ms, detail_json, level FROM observability_events WHERE session_id = ?"
        params: list = [session_id]

        if category:
            query += " AND category = ?"
            params.append(category)
        if level:
            query += " AND level = ?"
            params.append(level)

        query += " ORDER BY id"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            ObservabilityEvent(
                category=r["category"],
                event_type=r["event_type"],
                session_id=r["session_id"],
                turn=r["turn"],
                timestamp=r["timestamp"],
                duration_ms=r["duration_ms"],
                detail=json.loads(r["detail_json"]),
                level=r["level"],
            )
            for r in rows
        ]

    async def get_all_session_ids(self) -> list[str]:
        """Return all session IDs that have events."""
        if self._conn:
            cursor = await self._conn.execute(
                "SELECT DISTINCT session_id FROM observability_events"
            )
            rows = await cursor.fetchall()
            return [r["session_id"] for r in rows]

        return list(self._buffers.keys())
