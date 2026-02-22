"""In-memory session state store implementing SessionStoreBase."""

import logging
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


class InMemorySessionStore(SessionStoreBase):
    """In-memory session state, accessed by tools via session_id."""

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}
        self._summaries: dict[str, list[SessionSummary]] = {}
        self._episodes: dict[str, list[EpisodicMemory]] = {}
        self._traces: dict[str, list[EnrichedTrace]] = {}
        self._analyses: dict[str, list[TurnAnalysis]] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> SessionState:
        """Get or create session state."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def update_profile(self, session_id: str, **kwargs) -> FamilyProfile:
        """Update profile fields, return updated profile."""
        with self._lock:
            state = self.get(session_id)
            profile = state.family_profile

            for field, value in kwargs.items():
                if value is None:
                    continue
                if field in ("challenge_areas", "attempted_strategies", "hardest_situations"):
                    # Append to lists, dedup
                    current = getattr(profile, field)
                    if isinstance(value, list):
                        merged = list(dict.fromkeys(current + value))
                    else:
                        merged = list(dict.fromkeys(current + [value]))
                    setattr(profile, field, merged)
                else:
                    setattr(profile, field, value)

            logger.info("Profile updated for session %s: %s", session_id, kwargs)
            return profile

    def add_outcome(
        self,
        session_id: str,
        strategy_name: str,
        outcome: str,
        notes: str = "",
    ) -> Outcome:
        """Log an outcome for a strategy."""
        with self._lock:
            state = self.get(session_id)
            entry = Outcome(
                goal_description=strategy_name,
                signal=outcome,
                detail=notes,
                turn=state.turn_count,
            )
            state.outcomes.append(entry)
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
            state = self.get(session_id)

            if action == "add" and description:
                state.goals.append(Goal(
                    description=description,
                    created_turn=state.turn_count,
                ))
                logger.info("Goal added for session %s: %s", session_id, description)
            elif action == "complete" and description:
                for goal in state.goals:
                    if goal.description.lower() == description.lower() and goal.status == "active":
                        goal.status = "completed"
                        logger.info("Goal completed for session %s: %s", session_id, description)
                        break
            # "list" action just returns current goals

            return state.goals

    def seed_session(self, request: SeedSessionRequest) -> None:
        """Pre-populate a session with onboarding data."""
        with self._lock:
            state = self.get(request.session_id)
            state.family_profile.child_name = request.child_name or None
            state.family_profile.child_age = request.child_age or None
            state.family_profile.challenge_areas = request.challenges
            state.family_profile.attempted_strategies = request.tried_strategies
            for goal_text in request.goals:
                state.goals.append(Goal(description=goal_text))
            logger.info(
                "Session %s seeded: age=%s, challenges=%s",
                request.session_id,
                request.child_age,
                request.challenges,
            )

    def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn count."""
        with self._lock:
            state = self.get(session_id)
            state.turn_count += 1
            return state.turn_count

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
        """Persist a single message in conversation_history."""
        with self._lock:
            state = self.get(session_id)
            state.conversation_history.append({
                "role": role,
                "content": content,
                "turn": turn,
                "blocked": blocked,
                "blocked_reason": blocked_reason,
                "tool_calls_summary": tool_calls_summary,
            })

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Return message dicts."""
        with self._lock:
            state = self.get(session_id)
            messages = state.conversation_history
            if limit:
                messages = messages[-limit:]
            return messages

    def add_active_strategy(self, session_id: str, strategy_name: str) -> None:
        """Add an active strategy (dedup)."""
        with self._lock:
            state = self.get(session_id)
            if strategy_name not in state.active_strategies:
                state.active_strategies.append(strategy_name)

    def get_active_strategies(self, session_id: str) -> list[str]:
        """Return list of active strategy names."""
        with self._lock:
            return self.get(session_id).active_strategies

    def get_latest_summary(self, session_id: str) -> SessionSummary | None:
        """Return the most recent session summary, or None."""
        with self._lock:
            summaries = self._summaries.get(session_id, [])
            return summaries[-1] if summaries else None

    def save_summary(self, session_id: str, summary: SessionSummary) -> None:
        """Persist a rolling session summary."""
        with self._lock:
            if session_id not in self._summaries:
                self._summaries[session_id] = []
            self._summaries[session_id].append(summary)

    def add_episode(self, session_id: str, episode: EpisodicMemory) -> None:
        """Persist an episodic memory."""
        with self._lock:
            if session_id not in self._episodes:
                self._episodes[session_id] = []
            self._episodes[session_id].append(episode)

    def get_recent_episodes(self, session_id: str, limit: int = 5) -> list[EpisodicMemory]:
        """Return the most recent episodic memories."""
        with self._lock:
            episodes = self._episodes.get(session_id, [])
            return episodes[-limit:]

    def get_messages_range(
        self,
        session_id: str,
        start_turn: int,
        end_turn: int | None = None,
    ) -> list[dict]:
        """Return non-blocked messages in the given turn range (inclusive)."""
        with self._lock:
            state = self.get(session_id)
            result = []
            for msg in state.conversation_history:
                turn = msg.get("turn", 0)
                if msg.get("blocked"):
                    continue
                if turn < start_turn:
                    continue
                if end_turn is not None and turn > end_turn:
                    continue
                result.append(msg)
            return result

    def save_trace(self, session_id: str, trace: EnrichedTrace) -> None:
        """Persist an enriched trace for a turn."""
        with self._lock:
            if session_id not in self._traces:
                self._traces[session_id] = []
            self._traces[session_id].append(trace)

    def get_traces(self, session_id: str) -> list[EnrichedTrace]:
        """Return all enriched traces for a session."""
        with self._lock:
            return list(self._traces.get(session_id, []))

    def save_analysis(self, session_id: str, analysis: TurnAnalysis) -> None:
        """Persist a turn analysis."""
        with self._lock:
            if session_id not in self._analyses:
                self._analyses[session_id] = []
            self._analyses[session_id].append(analysis)

    def get_analyses(self, session_id: str) -> list[TurnAnalysis]:
        """Return all turn analyses for a session."""
        with self._lock:
            return list(self._analyses.get(session_id, []))

    def get_all_sessions(self) -> list[SessionListItem]:
        """Return summary info for all sessions."""
        with self._lock:
            return [
                SessionListItem(
                    session_id=s.session_id,
                    turn_count=s.turn_count,
                    phase=s.phase.value if hasattr(s.phase, "value") else str(s.phase),
                )
                for s in self._sessions.values()
            ]

    def get_session_timestamps(self, session_id: str) -> tuple[str, str]:
        """In-memory store has no timestamps."""
        return ("", "")


# Backward-compatibility alias
SessionStateStore = InMemorySessionStore
