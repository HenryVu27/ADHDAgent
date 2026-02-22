"""SQLite-backed session store implementing SessionStoreBase.

Each call to get() materializes a fresh SessionState from DB rows.
All mutations go through explicit store methods — never via returned object mutation.
"""

import json
import logging
import sqlite3
import threading

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import (
    EnrichedTrace,
    EpisodicMemory,
    FamilyProfile,
    Goal,
    Outcome,
    SeedSessionRequest,
    SessionListItem,
    SessionState,
    SessionSummary,
    TurnAnalysis,
)

logger = logging.getLogger(__name__)


class SQLiteSessionStore(SessionStoreBase):
    """Full SQLite implementation of the session store."""

    _VALID_PROFILE_FIELDS = frozenset({
        "child_name", "child_age", "diagnosis_status",
        "challenge_areas", "attempted_strategies",
        "good_day_description", "hardest_situations",
    })

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = threading.RLock()

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists without creating it."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row is not None

    def _ensure_session(self, session_id: str) -> None:
        """Create session + profile rows if they don't exist."""
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
            (session_id,),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO family_profiles (session_id) VALUES (?)",
            (session_id,),
        )
        self._conn.commit()

    def get(self, session_id: str) -> SessionState:
        """Materialize a SessionState from DB rows."""
        with self._lock:
            self._ensure_session(session_id)

            # Session metadata
            row = self._conn.execute(
                "SELECT turn_count, phase FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn_count = row["turn_count"]
            phase = row["phase"]

            # Family profile
            prof_row = self._conn.execute(
                "SELECT * FROM family_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            profile = FamilyProfile(
                child_name=prof_row["child_name"],
                child_age=prof_row["child_age"],
                diagnosis_status=prof_row["diagnosis_status"],
                challenge_areas=json.loads(prof_row["challenge_areas"]),
                attempted_strategies=json.loads(prof_row["attempted_strategies"]),
                good_day_description=prof_row["good_day_description"],
                hardest_situations=json.loads(prof_row["hardest_situations"]),
            )

            # Goals
            goal_rows = self._conn.execute(
                "SELECT description, strategy_id, created_turn, status FROM goals WHERE session_id = ?",
                (session_id,),
            ).fetchall()
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
            outcome_rows = self._conn.execute(
                "SELECT goal_description, signal, detail, turn FROM outcomes WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            outcomes = [
                Outcome(
                    goal_description=r["goal_description"],
                    signal=r["signal"],
                    detail=r["detail"],
                    turn=r["turn"],
                )
                for r in outcome_rows
            ]

            # Active strategies
            strat_rows = self._conn.execute(
                "SELECT strategy_name FROM active_strategies WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            active_strategies = [r["strategy_name"] for r in strat_rows]

            # Conversation history (non-blocked messages for backward compat)
            msg_rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND blocked = 0 ORDER BY id",
                (session_id,),
            ).fetchall()
            conversation_history = [
                {"role": r["role"], "content": r["content"]} for r in msg_rows
            ]

            return SessionState(
                session_id=session_id,
                phase=phase,
                turn_count=turn_count,
                family_profile=profile,
                active_strategies=active_strategies,
                goals=goals,
                outcomes=outcomes,
                conversation_history=conversation_history,
            )

    def update_profile(self, session_id: str, **kwargs) -> FamilyProfile:
        """Update profile fields, return updated profile."""
        with self._lock:
            for field in kwargs:
                if field not in self._VALID_PROFILE_FIELDS:
                    raise ValueError(f"Invalid profile field: {field!r}")

            self._ensure_session(session_id)

            # Read current profile
            prof_row = self._conn.execute(
                "SELECT * FROM family_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            list_fields = ("challenge_areas", "attempted_strategies", "hardest_situations")

            for field, value in kwargs.items():
                if value is None:
                    continue
                if field in list_fields:
                    current = json.loads(prof_row[field])
                    if isinstance(value, list):
                        merged = list(dict.fromkeys(current + value))
                    else:
                        merged = list(dict.fromkeys(current + [value]))
                    self._conn.execute(
                        f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                        (json.dumps(merged), session_id),
                    )
                else:
                    self._conn.execute(
                        f"UPDATE family_profiles SET {field} = ? WHERE session_id = ?",
                        (value, session_id),
                    )

            self._conn.execute(
                "UPDATE sessions SET updated_at = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()

            logger.info("Profile updated for session %s: %s", session_id, kwargs)
            # Re-read profile within the lock to avoid calling self.get() which also acquires the lock
            prof_row = self._conn.execute(
                "SELECT * FROM family_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return FamilyProfile(
                child_name=prof_row["child_name"],
                child_age=prof_row["child_age"],
                diagnosis_status=prof_row["diagnosis_status"],
                challenge_areas=json.loads(prof_row["challenge_areas"]),
                attempted_strategies=json.loads(prof_row["attempted_strategies"]),
                good_day_description=prof_row["good_day_description"],
                hardest_situations=json.loads(prof_row["hardest_situations"]),
            )

    def add_outcome(
        self,
        session_id: str,
        strategy_name: str,
        outcome: str,
        notes: str = "",
    ) -> Outcome:
        """Log an outcome for a strategy."""
        with self._lock:
            self._ensure_session(session_id)
            row = self._conn.execute(
                "SELECT turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turn = row["turn_count"]

            self._conn.execute(
                "INSERT INTO outcomes (session_id, goal_description, signal, detail, turn) VALUES (?, ?, ?, ?, ?)",
                (session_id, strategy_name, outcome, notes, turn),
            )
            self._conn.commit()

            entry = Outcome(
                goal_description=strategy_name,
                signal=outcome,
                detail=notes,
                turn=turn,
            )
            logger.info("Outcome recorded for session %s: %s -> %s", session_id, strategy_name, outcome)
            return entry

    def manage_goal(
        self,
        session_id: str,
        action: str,
        description: str = "",
    ) -> list[Goal]:
        """Add/complete/list goals."""
        with self._lock:
            self._ensure_session(session_id)

            if action == "add" and description:
                row = self._conn.execute(
                    "SELECT turn_count FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                self._conn.execute(
                    "INSERT INTO goals (session_id, description, created_turn) VALUES (?, ?, ?)",
                    (session_id, description, row["turn_count"]),
                )
                self._conn.commit()
                logger.info("Goal added for session %s: %s", session_id, description)
            elif action == "complete" and description:
                self._conn.execute(
                    "UPDATE goals SET status = 'completed' WHERE session_id = ? AND LOWER(description) = LOWER(?) AND status = 'active'",
                    (session_id, description),
                )
                self._conn.commit()
                logger.info("Goal completed for session %s: %s", session_id, description)

            # Return all goals
            goal_rows = self._conn.execute(
                "SELECT description, strategy_id, created_turn, status FROM goals WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return [
                Goal(
                    description=r["description"],
                    strategy_id=r["strategy_id"],
                    created_turn=r["created_turn"],
                    status=r["status"],
                )
                for r in goal_rows
            ]

    def seed_session(self, request: SeedSessionRequest) -> None:
        """Pre-populate a session with onboarding data."""
        with self._lock:
            self._ensure_session(request.session_id)

            updates = {}
            if request.child_name:
                updates["child_name"] = request.child_name
            if request.child_age:
                updates["child_age"] = request.child_age
            if request.challenges:
                updates["challenge_areas"] = request.challenges
            if request.tried_strategies:
                updates["attempted_strategies"] = request.tried_strategies

            if updates:
                self.update_profile(request.session_id, **updates)

            for goal_text in request.goals:
                self._conn.execute(
                    "INSERT INTO goals (session_id, description) VALUES (?, ?)",
                    (request.session_id, goal_text),
                )
            self._conn.commit()

            logger.info(
                "Session %s seeded: age=%s, challenges=%s",
                request.session_id,
                request.child_age,
                request.challenges,
            )

    def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn count."""
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1, updated_at = datetime('now') WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row["turn_count"]

    def add_message(
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
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
                "INSERT INTO messages (session_id, turn, role, content, blocked, blocked_reason, tool_calls_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, turn, role, content, int(blocked), blocked_reason, tool_calls_summary),
            )
            self._conn.commit()

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Return message dicts ordered by insertion."""
        with self._lock:
            if limit:
                rows = self._conn.execute(
                    "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = self._conn.execute(
                    "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()

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

    def add_active_strategy(self, session_id: str, strategy_name: str) -> None:
        """Persist an active strategy (upsert)."""
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
                "INSERT OR IGNORE INTO active_strategies (session_id, strategy_name) VALUES (?, ?)",
                (session_id, strategy_name),
            )
            self._conn.commit()

    def get_active_strategies(self, session_id: str) -> list[str]:
        """Return list of active strategy names."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT strategy_name FROM active_strategies WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return [r["strategy_name"] for r in rows]

    def get_latest_summary(self, session_id: str) -> SessionSummary | None:
        """Return the most recent session summary, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT summary, covers_through_turn FROM session_summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return SessionSummary(
                summary=row["summary"],
                covers_through_turn=row["covers_through_turn"],
            )

    def save_summary(self, session_id: str, summary: SessionSummary) -> None:
        """Persist a rolling session summary."""
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
                "INSERT INTO session_summaries (session_id, summary, covers_through_turn) VALUES (?, ?, ?)",
                (session_id, summary.summary, summary.covers_through_turn),
            )
            self._conn.commit()

    def add_episode(self, session_id: str, episode: EpisodicMemory) -> None:
        """Persist an episodic memory."""
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
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
            self._conn.commit()

    def get_recent_episodes(self, session_id: str, limit: int = 5) -> list[EpisodicMemory]:
        """Return the most recent episodic memories."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_type, summary, outcome, strategies_involved, emotional_context, turn_range_start, turn_range_end FROM episodes WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
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

    def get_messages_range(
        self,
        session_id: str,
        start_turn: int,
        end_turn: int | None = None,
    ) -> list[dict]:
        """Return non-blocked messages in the given turn range (inclusive)."""
        with self._lock:
            if end_turn is not None:
                rows = self._conn.execute(
                    "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? AND blocked = 0 AND turn >= ? AND turn <= ? ORDER BY id",
                    (session_id, start_turn, end_turn),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? AND blocked = 0 AND turn >= ? ORDER BY id",
                    (session_id, start_turn),
                ).fetchall()
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

    def save_trace(self, session_id: str, trace: EnrichedTrace) -> None:
        """Persist an enriched trace for a turn."""
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
                "INSERT INTO traces (session_id, turn, trace_json) VALUES (?, ?, ?)",
                (session_id, trace.turn, trace.model_dump_json()),
            )
            self._conn.commit()

    def get_traces(self, session_id: str) -> list[EnrichedTrace]:
        """Return all enriched traces for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT trace_json FROM traces WHERE session_id = ? ORDER BY turn",
                (session_id,),
            ).fetchall()
            return [EnrichedTrace.model_validate_json(r["trace_json"]) for r in rows]

    def save_analysis(self, session_id: str, analysis: TurnAnalysis) -> None:
        """Persist a turn analysis."""
        with self._lock:
            self._ensure_session(session_id)
            self._conn.execute(
                "INSERT INTO turn_analyses (session_id, turn, analysis_json) VALUES (?, ?, ?)",
                (session_id, analysis.turn, analysis.model_dump_json()),
            )
            self._conn.commit()

    def get_analyses(self, session_id: str) -> list[TurnAnalysis]:
        """Return all turn analyses for a session."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT analysis_json FROM turn_analyses WHERE session_id = ? ORDER BY turn",
                (session_id,),
            ).fetchall()
            return [TurnAnalysis.model_validate_json(r["analysis_json"]) for r in rows]

    def get_all_sessions(self) -> list[SessionListItem]:
        """Return summary info for all sessions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, turn_count, phase, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
            ).fetchall()
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

    def get_all_sessions_paginated(self, offset: int = 0, limit: int = 50) -> tuple[list[SessionListItem], int]:
        """Return paginated sessions and total count."""
        with self._lock:
            total_row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            total = total_row[0]

            rows = self._conn.execute(
                "SELECT session_id, turn_count, phase, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
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

    def get_messages_paginated(self, session_id: str, offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
        """Return paginated messages and total count."""
        with self._lock:
            total_row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            total = total_row[0]

            rows = self._conn.execute(
                "SELECT role, content, turn, blocked, blocked_reason, tool_calls_summary FROM messages WHERE session_id = ? ORDER BY id LIMIT ? OFFSET ?",
                (session_id, limit, offset),
            ).fetchall()
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

    def get_session_timestamps(self, session_id: str) -> tuple[str, str]:
        """Return (created_at, updated_at) from the sessions table."""
        with self._lock:
            row = self._conn.execute(
                "SELECT created_at, updated_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return (row["created_at"] or "", row["updated_at"] or "")
            return ("", "")
