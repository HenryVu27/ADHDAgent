import json
from pathlib import Path

from app.agents.base import BaseAgent

KNOWLEDGE_PATH = Path(__file__).parent.parent / "knowledge" / "adhd_strategies.json"


class StrategyAgent(BaseAgent):
    """
    Recommends evidence-based parenting strategies matched to the family's
    specific situation and challenges.
    """

    name = "strategy"
    description = "Recommends evidence-based ADHD parenting strategies"

    def __init__(self):
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> list[dict]:
        if KNOWLEDGE_PATH.exists():
            with open(KNOWLEDGE_PATH) as f:
                return json.load(f)
        return []

    def process(self, message: str, context: dict) -> str:
        predicates = context.get("predicates", [])
        matched = self._match_strategies(predicates)

        if not matched:
            return (
                "I hear you. Let me think about this situation carefully. "
                "Could you tell me a bit more about when this typically happens "
                "and what you've already tried?"
            )

        strategy = matched[0]
        return (
            f"Based on what you're describing, here's a strategy that many families "
            f"find helpful:\n\n"
            f"**{strategy['name']}**\n\n"
            f"{strategy['description']}\n\n"
            f"**How to try it:**\n"
            + "\n".join(f"- {step}" for step in strategy.get("steps", []))
            + "\n\nWould you like to explore this further, or would you prefer "
            "to hear about a different approach?"
        )

    def _match_strategies(self, predicates: list[dict]) -> list[dict]:
        """Match predicates to relevant strategies from the knowledge base."""
        if not self.strategies or not predicates:
            return []

        matched = []
        for strategy in self.strategies:
            tags = set(strategy.get("tags", []))
            for pred in predicates:
                pred_terms = {pred.get("subject", ""), pred.get("category", "")}
                if tags & pred_terms:
                    matched.append(strategy)
                    break

        return matched

    def can_handle(self, asp_directives: list[str]) -> bool:
        return any("strategy" in d or "recommend" in d for d in asp_directives)
