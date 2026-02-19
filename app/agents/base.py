from abc import ABC, abstractmethod

from app.models.schemas import PhaseDecision, SessionState


class BaseAgent(ABC):
    """Base class for all coaching agents."""

    name: str = "base"
    description: str = "Base agent"

    @abstractmethod
    async def process(
        self,
        message: str,
        decision: PhaseDecision,
        state: SessionState,
        rag_context: str,
    ) -> str:
        """Process a parent message and return a response."""
        ...
