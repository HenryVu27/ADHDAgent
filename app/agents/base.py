from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all coaching agents."""

    name: str = "base"
    description: str = "Base agent"

    @abstractmethod
    def process(self, message: str, context: dict) -> str:
        """Process a parent message and return a response."""
        pass

    def can_handle(self, asp_directives: list[str]) -> bool:
        """Check if this agent should handle the current turn based on ASP directives."""
        return any(self.name in d for d in asp_directives)
