"""Abstract base class for session stores.

Both InMemorySessionStore and SQLiteSessionStore implement this protocol,
making them interchangeable throughout the application.
"""

from abc import ABC, abstractmethod

from app.models.schemas import (
    EpisodicMemory,
    FamilyProfile,
    Goal,
    Outcome,
    SeedSessionRequest,
    SessionState,
    SessionSummary,
)


class SessionStoreBase(ABC):
    """Protocol for session state persistence."""

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
