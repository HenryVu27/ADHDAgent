"""Abstract base class for session stores.

Both InMemorySessionStore and SQLiteSessionStore implement this protocol,
making them interchangeable throughout the application.
"""

from abc import ABC, abstractmethod

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
    StoredToolResult,
    TurnAnalysis,
)


class SessionStoreBase(ABC):
    """Protocol for session state persistence."""

    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists without creating it."""

    @abstractmethod
    def get(self, session_id: str) -> SessionState:
        """Get or create session state."""

    @abstractmethod
    def update_profile(self, session_id: str, **kwargs) -> FamilyProfile:
        """Update profile fields, return updated profile."""

    @abstractmethod
    def add_outcome(
        self,
        session_id: str,
        strategy_name: str,
        outcome: str,
        notes: str = "",
    ) -> Outcome:
        """Log an outcome for a strategy."""

    @abstractmethod
    def manage_goal(
        self,
        session_id: str,
        action: str,
        description: str = "",
    ) -> list[Goal]:
        """Add/complete/list goals."""

    @abstractmethod
    def seed_session(self, request: SeedSessionRequest) -> None:
        """Pre-populate a session with onboarding data."""

    @abstractmethod
    def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn count."""

    @abstractmethod
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
        """Persist a single message (user or assistant). Blocked turns included."""

    @abstractmethod
    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Return message dicts with keys: role, content, turn, blocked, blocked_reason."""

    @abstractmethod
    def add_active_strategy(self, session_id: str, strategy_name: str) -> None:
        """Persist an active strategy (avoids direct state mutation)."""

    @abstractmethod
    def get_active_strategies(self, session_id: str) -> list[str]:
        """Return list of active strategy names for the session."""

    @abstractmethod
    def get_latest_summary(self, session_id: str) -> SessionSummary | None:
        """Return the most recent session summary, or None."""

    @abstractmethod
    def save_summary(self, session_id: str, summary: SessionSummary) -> None:
        """Persist a rolling session summary."""

    @abstractmethod
    def add_episode(self, session_id: str, episode: EpisodicMemory) -> None:
        """Persist an episodic memory."""

    @abstractmethod
    def get_recent_episodes(self, session_id: str, limit: int = 5) -> list[EpisodicMemory]:
        """Return the most recent episodic memories."""

    @abstractmethod
    def get_messages_range(
        self,
        session_id: str,
        start_turn: int,
        end_turn: int | None = None,
    ) -> list[dict]:
        """Return non-blocked messages in the given turn range (inclusive)."""

    @abstractmethod
    def save_trace(self, session_id: str, trace: EnrichedTrace) -> None:
        """Persist an enriched trace for a turn."""

    @abstractmethod
    def get_traces(self, session_id: str) -> list[EnrichedTrace]:
        """Return all enriched traces for a session."""

    @abstractmethod
    def save_analysis(self, session_id: str, analysis: TurnAnalysis) -> None:
        """Persist a turn analysis."""

    @abstractmethod
    def get_analyses(self, session_id: str) -> list[TurnAnalysis]:
        """Return all turn analyses for a session."""

    @abstractmethod
    def get_all_sessions(self) -> list[SessionListItem]:
        """Return summary info for all sessions."""

    @abstractmethod
    def get_all_sessions_paginated(self, offset: int = 0, limit: int = 50) -> tuple[list[SessionListItem], int]:
        """Return paginated sessions and total count."""

    @abstractmethod
    def get_messages_paginated(self, session_id: str, offset: int = 0, limit: int = 50) -> tuple[list[dict], int]:
        """Return paginated messages and total count."""

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete all data for a session (right-to-erasure)."""

    @abstractmethod
    def save_tool_result(self, session_id: str, tool_name: str, query: str, result_text: str, turn: int) -> None:
        """Persist a tool result for cross-turn retrieval."""

    @abstractmethod
    def get_recent_tool_results(self, session_id: str, limit: int = 3) -> list[StoredToolResult]:
        """Return the most recent tool results."""

    @abstractmethod
    def get_session_timestamps(self, session_id: str) -> tuple[str, str]:
        """Return (created_at, updated_at) for a session. Empty strings if unavailable."""
