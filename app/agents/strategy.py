"""
Strategy agent with RAG + Gemini response generation.

Receives RAG retrieval results, formats as context for Gemini,
generates personalized strategy recommendations. Tracks which
strategies are recommended for outcome measurement.
"""

import logging

from app.agents.base import BaseAgent
from app.agents.context import format_conversation_window
from app.config import settings
from app.llm.prompts import RESPONSE_GENERATION_PROMPT
from app.models.schemas import PhaseDecision, SessionState

logger = logging.getLogger(__name__)


class StrategyAgent(BaseAgent):
    """Recommends evidence-based ADHD parenting strategies using RAG + Gemini."""

    name = "strategy"
    description = "Recommends evidence-based ADHD parenting strategies"

    def __init__(self, gemini_client=None):
        self._gemini = gemini_client

    async def process(
        self,
        message: str,
        decision: PhaseDecision,
        state: SessionState,
        rag_context: str,
    ) -> str:
        if self._gemini and settings.USE_LLM_RESPONSES:
            try:
                return await self._generate_with_gemini(
                    message, decision, state, rag_context
                )
            except Exception as e:
                logger.warning(f"Gemini strategy generation failed: {e}")

        return self._fallback_response(rag_context)

    async def _generate_with_gemini(
        self,
        message: str,
        decision: PhaseDecision,
        state: SessionState,
        rag_context: str,
    ) -> str:
        """Generate strategy response using Gemini with RAG context."""
        profile = state.family_profile
        family_str = (
            f"Child name: {profile.child_name or 'your child'}, "
            f"Child age: {profile.child_age or 'unknown'}, "
            f"Challenges: {profile.challenge_areas or ['not yet identified']}, "
            f"Tried: {profile.attempted_strategies or ['not yet discussed']}"
        )

        conversation_history = format_conversation_window(state.conversation_history)

        rag_section = ""
        if rag_context:
            rag_section = f"Retrieved knowledge base context:\n{rag_context}"

        prompt = RESPONSE_GENERATION_PROMPT.format(
            message=message,
            conversation_history=conversation_history or "No prior conversation.",
            phase=decision.phase.value,
            family_profile=family_str,
            agent="strategy",
            directives=decision.directives,
            constraints=decision.constraints,
            rag_context=rag_section,
            active_strategies=state.active_strategies or ["none yet"],
        )

        return await self._gemini.generate(prompt, temperature=0.7)

    def _fallback_response(self, rag_context: str) -> str:
        """Template response when Gemini is unavailable."""
        if rag_context:
            return (
                "Based on what you're describing, here are some approaches "
                "that many families find helpful:\n\n"
                f"{rag_context}\n\n"
                "Would you like to explore any of these further, or would you "
                "prefer to hear about a different approach?"
            )

        return (
            "I hear you. Let me think about this situation carefully. "
            "Could you tell me a bit more about when this typically happens "
            "and what you've already tried?"
        )
