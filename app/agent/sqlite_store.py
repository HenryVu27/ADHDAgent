"""Async SQLite-backed session store implementing SessionStoreBase.

Uses aiosqlite for true async I/O. Each call to get() materializes a fresh
SessionState from DB rows. All mutations go through explicit store methods.

Commits are batched: individual write methods do NOT commit. Callers must
call await store.commit() at transaction boundaries. Exceptions: delete_session()
and seed_session() commit internally (standalone operations).
"""

import json
import logging

import aiosqlite

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import (
    EnrichedTrace,
    EpisodeLink,
    EpisodicMemory,
    FamilyProfile,
    Goal,
    Outcome,
    ProfileChange,
    SeedSessionRequest,
    SessionListItem,
    SessionState,
    SessionSummary,
    StoredToolResult,
    TurnAnalysis,
)

logger = logging.getLogger(__name__)


class SQLiteSessionStore(SessionStoreBase):
    """Async SQLite implementation of the session store."""

    _VALID_PROFILE_FIELDS = frozenset({
        "child_name", "child_age", "diagnosis_status", "adhd_subtype",
        "challenge_areas", "attempted_strategies",
        "good_day_description", "hardest_situations",
    })

    _SCALAR_PROFILE_FIELDS = frozenset({
        "child_name", "child_age", "diagnosis_status",
        "adhd_subtype", "good_day_description",
    })

    def __init__(self, conn: aiosqlite.Connection):
        self._conn = conn

    async def commit(self) -> None:
        """Flush pending writes to durable storage."""
        await self._conn.commit()

    async def _touch_updated(self, session_id: str) -> None:
        """Mark session as recently updated."""
        await self._conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?",
            (session_id,),
        )

    async def delete_session(self, session_id: str) -> None:
        """Delete all data for a session (cascading). Commits internally."""
        for table in (
            "tool_results", "turn_analyses", "traces", "episode_links", "episodes",
            "session_summaries", "active_strategies", "outcomes", "goals", "messages",
            "family_profiles", "profile_changelog", "sessions",
        ):
            await self._conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        await self._conn.commit()
        logger.info("Session %s deleted", session_id)

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists without creating it."""
        cursor = await self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row is not None

    async def _ensure_session(self, session_id: str, user_id: int | None = None) -> None:
        """Create session + profile rows if they don't exist.

        user_id is only used on first creation (INSERT OR IGNORE is a no-op
        if the session already exists, so internal callers can omit it).
        """
        await self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, user_id) VALUES (?, ?)",
            (session_id, user_id),
        )
        await self._conn.execute(
            "INSERT OR IGNORE INTO family_profiles (session_id) VALUES (?)",
            (session_id,),
        )

    async def get(self, session_id: str) -> SessionState:
        """Materialize a SessionState from DB rows.

        Does NOT load conversation_history (callers use get_messages() instead).
        """
        await self._ensure_session(session_id)

        # Session metadata
        cursor = await self._conn.execute(
            "SELECT turn_count, phase FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        turn_count = row["turn_count"]
        phase = row["phase"]

        # Family profile
        cursor = await self._conn.execute(
            "SELECT * FROM family_profiles WHERE session_id = ?",
            (session_id,),
        )
        prof_row = await cursor.fetchone()
        profile = FamilyProfile(
            child_name=prof_row["child_name"],
            child_age=prof_row["child_age"],
            diagnosis_status=prof_row["diagnosis_status"],
            adhd_subtype=prof_row["adhd_subtype"],
            challenge_areas=json.loads(prof_row["challenge_areas"]),
            attempted_strategies=json.loads(prof_row["attempted_strategies"]),
            good_day_description=prof_row["good_day_description"],
            hardest_situations=json.loads(prof_row["hardest_situations"]),
        )

        # Goals
        cursor = await self._conn.execute(
            "SELECT description, strategy_id, created_turn, status FROM goals WHERE session_id = ?",
            (session_id,),
        )
        goal_rows = await cursor.fetchall()
        goals = [
            Goal(
                description=r["description"],
                strategy_id=r["strategy_id"],
                created_turn=r["created_turn"],
                status=r["status"],
            )
            for r in goal_rows
        ]

        # Outcomes
        cursor = await self._conn.execute(
            "SELECT strategy_name, signal, detail, turn FROM outcomes WHERE session_id = ?",
            (session_id,),
        )
        outcome_rows = await cursor.fetchall()
        outcomes = [
            Outcome(
                strategy_name=r["strategy_name"],
                signal=r["signal"],
                detail=r["detail"],
                turn=r["turn"],
            )
            for r in outcome_rows
        ]

        # Active strategies
        cursor = await self._conn.execute(
            "SELECT strategy_name FROM active_strategies WHERE session_id = ?",
            (session_id,),
        )
        strat_rows = await cursor.fetchall()
        active_strategies = [r["strategy_name"] for r in strat_rows]

        return SessionState(
            session_id=session_id,
            phase=phase,
            turn_count=turn_count,
            family_profile=profile,
            active_strategies=active_strategies,
            goals=goals,
            outcomes=outcomes,
            conversation_history=[],
        )

    async def update_profile(self, session_id: str, **kwargs) -> FamilyProfile:
        """Update profile fields, return updated profile."""
        for field in kwargs:
            if field not in self._VALID_PROFILE_FIELDS:
                raise ValueError(f"Invalid profile field: {field!r}")

        await self._ensure_session(session_id)

        # Read current profile
        cursor = await self._conn.execute(
            "SELECT * FROM family_profiles WHERE session_id = ?",
            (session_id,),
        )
        prof_row = await cursor.fetchone()

        list_fields = ("challenge_areas", "attempted_strategies", "hardest_situations")

        # Get current turn for changelog entries
        cursor2 = await self._conn.execute(
            "SELECT turn_count FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        turn_row = await cursor2.fetchone()
        current_turn = turn_row["turn_count"] if turn_row else 0

        for field, value in kwargs.items():
            if value is None:
                continue
            if field in list_fields:
                current = json.loads(prof_row[field])
                if isinstance(value, list):
                    merged = list(dict.fromkeys(current + value))
                else:
                    merged = list(dict.fromkeys(current + [value]))
                await self._conn.execute(
                    f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                    (json.dumps(merged), session_id),
                )
            elif field in self._SCALAR_PROFILE_FIELDS:
                old_value = prof_row[field]
                if old_value is not None and old_value != value:
                    await self._conn.execute(
                        "INSERT INTO profile_changelog (session_id, field, old_value, new_value, turn) VALUES (?, ?, ?, ?, ?)",
                        (session_id, field, str(old_value), str(value), current_turn),
                    )
                await self._conn.execute(
                    f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                    (value, session_id),
                )
            else:
                await self._conn.execute(
                    f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                    (value, session_id),
                )

        await self._touch_updated(session_id)

        logger.info("Profile updated for session %s: %s", session_id, kwargs)
        # Re-read profile
        cursor = await self._conn.execute(
            "SELECT * FROM family_profiles WHERE session_id = ?",
            (session_id,),
        )
        prof_row = await cursor.fetchone()
        return FamilyProfile(
            child_name=prof_row["child_name"],
            child_age=prof_row["child_age"],
            diagnosis_status=prof_row["diagnosis_status"],
            adhd_subtype=prof_row["adhd_subtype"],
            challenge_areas=json.loads(prof_row["challenge_areas"]),
            attempted_strategies=json.loads(prof_row["attempted_strategies"]),
            good_day_description=prof_row["good_day_description"],
            hardest_situations=json.loads(prof_row["hardest_situations"]),
        )

    async def add_outcome(
        self,
        session_id: str,
        strategy_name: str,
        outcome: str,
        notes: str = "",
    ) -> Outcome:
        """Log an outcome for a strategy."""
        await self._ensure_session(session_id)
        cursor = await self._conn.execute(
            "SELECT turn_count FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        turn = row["turn_count"]

        await self._conn.execute(
            "INSERT INTO outcomes (session_id, strategy_name, signal, detail, turn) VALUES (?, ?, ?, ?, ?)",
            (session_id, strategy_name, outcome, notes, turn),
        )
        await self._touch_updated(session_id)

        entry = Outcome(
            strategy_name=strategy_name,
            signal=outcome,
            detail=notes,
            turn=turn,
        )
        logger.info("Outcome recorded for session %s: %s -> %s", session_id, strategy_name, outcome)
        return entry

    async def manage_goal(
        self,
        session_id: str,
        action: str,
        description: str = "",
    ) -> list[Goal]:
        """Add/complete/list goals."""
        await self._ensure_session(session_id)

        if action == "add" and description:
            cursor = await self._conn.execute(
                "SELECT turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            cursor = await self._conn.execute(
                "INSERT OR IGNORE INTO goals (session_id, description, created_turn) VALUES (?, ?, ?)",
                (session_id, description, row["turn_count"]),
            )
            if cursor.rowcount == 1:
                await self._touch_updated(session_id)
                logger.info("Goal added for session %s: %s", session_id, description)
        elif action == "complete" and description:
            await self._conn.execute(
                "UPDATE goals SET status = 'completed' WHERE session_id = ? AND LOWER(description) = LOWER(?) AND status = 'active'",
                (session_id, description),
            )
            await self._touch_updated(session_id)
            logger.info("Goal completed for session %s: %s", session_id, description)

        # Return all goals
        cursor = await self._conn.execute(
            "SELECT description, strategy_id, created_turn, status FROM goals WHERE session_id = ?",
            (session_id,),
        )
        goal_rows = await cursor.fetchall()
        return [
            Goal(
                description=r["description"],
                strategy_id=r["strategy_id"],
                created_turn=r["created_turn"],
                status=r["status"],
            )
            for r in goal_rows
        ]

    async def seed_session(self, request: SeedSessionRequest) -> None:
        """Pre-populate a session with onboarding data. Commits internally."""
        await self._ensure_session(request.session_id)

        list_fields = ("challenge_areas", "attempted_strategies", "hardest_situations")
        updates = {}
        if request.child_name:
            updates["child_name"] = request.child_name
        if request.child_age:
            updates["child_age"] = request.child_age
        if request.diagnosis_status:
            updates["diagnosis_status"] = request.diagnosis_status
        if request.adhd_subtype:
            updates["adhd_subtype"] = request.adhd_subtype
        if request.challenges:
            updates["challenge_areas"] = request.challenges
        if request.tried_strategies:
            updates["attempted_strategies"] = request.tried_strategies

        if updates:
            for field in updates:
                if field not in self._VALID_PROFILE_FIELDS:
                    raise ValueError(f"Invalid profile field: {field!r}")

            cursor = await self._conn.execute(
                "SELECT * FROM family_profiles WHERE session_id = ?",
                (request.session_id,),
            )
            prof_row = await cursor.fetchone()

            for field, value in updates.items():
                if field in list_fields:
                    current = json.loads(prof_row[field])
                    if isinstance(value, list):
                        merged = list(dict.fromkeys(current + value))
                    else:
                        merged = list(dict.fromkeys(current + [value]))
                    await self._conn.execute(
                        f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                        (json.dumps(merged), request.session_id),
                    )
                else:
                    await self._conn.execute(
                        f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                        (value, request.session_id),
                    )

        for goal_text in request.goals:
            await self._conn.execute(
                "INSERT OR IGNORE INTO goals (session_id, description) VALUES (?, ?)",
                (request.session_id, goal_text),
            )

        # Single commit for the entire seed operation
        await self._conn.commit()

        logger.info(
            "Session %s seeded: age=%s, challenges=%s",
            request.session_id,
            request.child_age,
            request.challenges,
        )

    async def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn count."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "UPDATE sessions SET turn_count = turn_count + 1, updated_at = datetime('now') WHERE session_id = ?",
            (session_id,),
        )
        cursor = await self._conn.execute(
            "SELECT turn_count FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row["turn_count"]

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        turn: int,
        blocked: bool = False,
        blocked_reason: str = "",
        tool_calls_summary: str = "",
    ) -> None:
        """Persist a single message."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT INTO messages (session_id, turn, role, content, blocked, blocked_reason, tool_calls_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn, role, content, int(blocked), blocked_reason, tool_calls_summary),
        )
        await self._touch_updated(session_id)

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Return message dicts ordered by insertion."""
        if limit:
            cursor = await self._conn.execute(
                "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            rows = await cursor.fetchall()
            rows = list(reversed(rows))
        else:
            cursor = await self._conn.execute(
                "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            )
            rows = await cursor.fetchall()

        return [
            {
                "role": r["role"],
                "content": r["content"],
                "turn": r["turn"],
                "blocked": bool(r["blocked"]),
                "blocked_reason": r["blocked_reason"],
                "tool_calls_summary": r["tool_calls_summary"],
            }
            for r in rows
        ]

    async def add_active_strategy(self, session_id: str, strategy_name: str) -> None:
        """Persist an active strategy (upsert)."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT OR IGNORE INTO active_strategies (session_id, strategy_name) VALUES (?, ?)",
            (session_id, strategy_name),
        )
        await self._touch_updated(session_id)

    async def get_active_strategies(self, session_id: str) -> list[str]:
        """Return list of active strategy names."""
        cursor = await self._conn.execute(
            "SELECT strategy_name FROM active_strategies WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [r["strategy_name"] for r in rows]

    async def get_latest_summary(self, session_id: str) -> SessionSummary | None:
        """Return the most recent session summary, or None."""
        cursor = await self._conn.execute(
            "SELECT summary, covers_through_turn FROM session_summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return SessionSummary(
            summary=row["summary"],
            covers_through_turn=row["covers_through_turn"],
        )

    async def save_summary(self, session_id: str, summary: SessionSummary) -> None:
        """Persist a rolling session summary."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT INTO session_summaries (session_id, summary, covers_through_turn) VALUES (?, ?, ?)",
            (session_id, summary.summary, summary.covers_through_turn),
        )
        await self._touch_updated(session_id)

    async def get_profile_changelog(self, session_id: str) -> list[ProfileChange]:
        """Return all profile field changes, oldest first."""
        cursor = await self._conn.execute(
            "SELECT field, old_value, new_value, turn, created_at "
            "FROM profile_changelog WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            ProfileChange(
                field=r["field"],
                old_value=r["old_value"],
                new_value=r["new_value"],
                turn=r["turn"],
                created_at=r["created_at"] or "",
            )
            for r in rows
        ]

    async def add_episode(self, session_id: str, episode: EpisodicMemory) -> int:
        """Persist an episodic memory. Returns the new episode's row ID."""
        await self._ensure_session(session_id)
        cursor = await self._conn.execute(
            "INSERT INTO episodes (session_id, event_type, summary, outcome, strategies_involved, emotional_context, turn_range_start, turn_range_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                episode.event_type,
                episode.summary,
                episode.outcome,
                json.dumps(episode.strategies_involved),
                episode.emotional_context,
                episode.turn_range_start,
                episode.turn_range_end,
            ),
        )
        await self._touch_updated(session_id)
        return cursor.lastrowid

    async def get_recent_episodes(self, session_id: str, limit: int = 5) -> list[EpisodicMemory]:
        """Return the most recent episodic memories."""
        cursor = await self._conn.execute(
            "SELECT event_type, summary, outcome, strategies_involved, emotional_context, turn_range_start, turn_range_end FROM episodes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            EpisodicMemory(
                event_type=r["event_type"],
                summary=r["summary"],
                outcome=r["outcome"],
                strategies_involved=json.loads(r["strategies_involved"]),
                emotional_context=r["emotional_context"],
                turn_range_start=r["turn_range_start"],
                turn_range_end=r["turn_range_end"],
            )
            for r in reversed(rows)
        ]

    async def get_episodes_with_ids(self, session_id: str, limit: int = 20) -> list[tuple[int, EpisodicMemory]]:
        """Return (id, EpisodicMemory) tuples for recent episodes, for linking purposes."""
        cursor = await self._conn.execute(
            "SELECT id, event_type, summary, outcome, strategies_involved, emotional_context, "
            "turn_range_start, turn_range_end FROM episodes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            (
                r["id"],
                EpisodicMemory(
                    event_type=r["event_type"],
                    summary=r["summary"],
                    outcome=r["outcome"],
                    strategies_involved=json.loads(r["strategies_involved"]),
                    emotional_context=r["emotional_context"],
                    turn_range_start=r["turn_range_start"],
                    turn_range_end=r["turn_range_end"],
                ),
            )
            for r in reversed(rows)
        ]

    async def add_episode_link(self, session_id: str, link: EpisodeLink) -> None:
        """Persist a link between two episodes."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT OR IGNORE INTO episode_links (session_id, source_id, target_id, link_type, link_reason) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, link.source_id, link.target_id, link.link_type, link.link_reason),
        )

    async def get_episode_links(self, session_id: str) -> list[EpisodeLink]:
        """Return all episode links for a session."""
        cursor = await self._conn.execute(
            "SELECT source_id, target_id, link_type, link_reason, created_at "
            "FROM episode_links WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            EpisodeLink(
                source_id=r["source_id"],
                target_id=r["target_id"],
                link_type=r["link_type"],
                link_reason=r["link_reason"],
                created_at=r["created_at"] or "",
            )
            for r in rows
        ]

    async def get_messages_range(
        self,
        session_id: str,
        start_turn: int,
        end_turn: int | None = None,
    ) -> list[dict]:
        """Return non-blocked messages in the given turn range (inclusive)."""
        if end_turn is not None:
            cursor = await self._conn.execute(
                "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? AND blocked = 0 AND turn >= ? AND turn <= ? ORDER BY id",
                (session_id, start_turn, end_turn),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? AND blocked = 0 AND turn >= ? ORDER BY id",
                (session_id, start_turn),
            )
        rows = await cursor.fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "turn": r["turn"],
                "blocked": bool(r["blocked"]),
                "blocked_reason": r["blocked_reason"],
                "tool_calls_summary": r["tool_calls_summary"],
            }
            for r in rows
        ]

    async def save_trace(self, session_id: str, trace: EnrichedTrace) -> None:
        """Persist an enriched trace for a turn."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT INTO traces (session_id, turn, trace_json) VALUES (?, ?, ?)",
            (session_id, trace.turn, trace.model_dump_json()),
        )
        await self._touch_updated(session_id)

    async def get_traces(self, session_id: str) -> list[EnrichedTrace]:
        """Return all enriched traces for a session."""
        cursor = await self._conn.execute(
            "SELECT trace_json FROM traces WHERE session_id = ? ORDER BY turn",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [EnrichedTrace.model_validate_json(r["trace_json"]) for r in rows]

    async def save_analysis(self, session_id: str, analysis: TurnAnalysis) -> None:
        """Persist a turn analysis."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT INTO turn_analyses (session_id, turn, analysis_json) VALUES (?, ?, ?)",
            (session_id, analysis.turn, analysis.model_dump_json()),
        )
        await self._touch_updated(session_id)

    async def get_analyses(self, session_id: str) -> list[TurnAnalysis]:
        """Return all turn analyses for a session."""
        cursor = await self._conn.execute(
            "SELECT analysis_json FROM turn_analyses WHERE session_id = ? ORDER BY turn",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [TurnAnalysis.model_validate_json(r["analysis_json"]) for r in rows]

    async def get_all_sessions(self, user_id: int | None = None) -> list[SessionListItem]:
        """Return summary info for sessions. Filters by user_id when provided."""
        if user_id is not None:
            cursor = await self._conn.execute(
                "SELECT session_id, turn_count, phase, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT session_id, turn_count, phase, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
            )
        rows = await cursor.fetchall()
        return [
            SessionListItem(
                session_id=r["session_id"],
                turn_count=r["turn_count"],
                phase=r["phase"],
                created_at=r["created_at"] or "",
                updated_at=r["updated_at"] or "",
            )
            for r in rows
        ]

    async def get_all_sessions_paginated(self, offset: int = 0, limit: int = 50, user_id: int | None = None) -> tuple[list[SessionListItem], int]:
        """Return paginated sessions and total count. Filters by user_id when provided."""
        if user_id is not None:
            count_row = await (await self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,)
            )).fetchone()
            total = count_row[0]
            cursor = await self._conn.execute(
                "SELECT session_id, turn_count, phase, created_at, updated_at FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset),
            )
        else:
            cursor = await self._conn.execute("SELECT COUNT(*) FROM sessions")
            total_row = await cursor.fetchone()
            total = total_row[0]

            cursor = await self._conn.execute(
                "SELECT session_id, turn_count, phase, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        items = [
            SessionListItem(
                session_id=r["session_id"],
                turn_count=r["turn_count"],
                phase=r["phase"],
                created_at=r["created_at"] or "",
                updated_at=r["updated_at"] or "",
            )
            for r in rows
        ]
        return items, total

    async def get_messages_paginated(self, session_id: str, offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
        """Return paginated messages and total count."""
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        )
        total_row = await cursor.fetchone()
        total = total_row[0]

        cursor = await self._conn.execute(
            "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? ORDER BY id LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        messages = [
            {
                "role": r["role"],
                "content": r["content"],
                "turn": r["turn"],
                "blocked": bool(r["blocked"]),
                "blocked_reason": r["blocked_reason"],
                "tool_calls_summary": r["tool_calls_summary"],
            }
            for r in rows
        ]
        return messages, total

    async def save_tool_result(self, session_id: str, tool_name: str, query: str, result_text: str, turn: int) -> None:
        """Persist a tool result for cross-turn retrieval."""
        await self._ensure_session(session_id)
        await self._conn.execute(
            "INSERT INTO tool_results (session_id, turn, tool_name, query, result_text) VALUES (?, ?, ?, ?, ?)",
            (session_id, turn, tool_name, query, result_text[:5000]),
        )
        await self._touch_updated(session_id)

    async def get_recent_tool_results(self, session_id: str, limit: int = 3) -> list[StoredToolResult]:
        """Return the most recent tool results."""
        cursor = await self._conn.execute(
            "SELECT tool_name, query, result_text, turn FROM tool_results WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [StoredToolResult(tool_name=r["tool_name"], query=r["query"], result_text=r["result_text"], turn=r["turn"]) for r in reversed(rows)]

    async def get_session_timestamps(self, session_id: str) -> tuple[str, str]:
        """Return (created_at, updated_at) from the sessions table."""
        cursor = await self._conn.execute(
            "SELECT created_at, updated_at FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row:
            return (row["created_at"] or "", row["updated_at"] or "")
        return ("", "")
