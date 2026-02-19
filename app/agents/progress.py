"""
Progress agent for goal tracking and outcome measurement.

Tracks goals, check-ins, and outcomes. This is Bryan's "close the loop" —
measuring whether strategies are actually working for the family.
"""

import logging

from app.agents.base import BaseAgent
from app.agents.context import format_conversation_window
from app.config import settings
from app.llm.prompts import PROGRESS_CHECK_PROMPT
from app.models.schemas import (
    ExtractionResult,
    Goal,
    Outcome,
    RulesDecision,
    SessionState,
)

logger = logging.getLogger(__name__)

# Signals that indicate positive or negative outcomes from predicates
POSITIVE_SIGNALS = {"better", "improved", "working", "helped", "easier", "success", "good"}
NEGATIVE_SIGNALS = {"worse", "harder", "failed", "didn't work", "struggling", "setback"}


class ProgressAgent(BaseAgent):
    """Tracks goals, check-ins, and outcomes for strategy effectiveness."""

    name = "progress"
    description = "Tracks goals and progress, celebrates wins, adjusts plans"

    def __init__(self, gemini_client=None):
        self._gemini = gemini_client

    async def process(
        self,
        message: str,
        extraction: ExtractionResult,
        decision: RulesDecision,
        state: SessionState,
        rag_context: str,
    ) -> str:
        # Track outcome signals from the message
        self._track_outcomes(message, extraction, state)

        if self._gemini and settings.USE_LLM_RESPONSES:
            try:
                return await self._generate_with_gemini(message, state)
            except Exception as e:
                logger.warning(f"Gemini progress generation failed: {e}")

        return self._fallback_response(state)

    async def _generate_with_gemini(self, message: str, state: SessionState) -> str:
        """Generate progress check-in using Gemini."""
        conversation_history = format_conversation_window(state.conversation_history)

        profile = state.family_profile
        family_context = (
            f"Child name: {profile.child_name or 'your child'}, "
            f"Child age: {profile.child_age or 'unknown'}, "
            f"Challenges: {profile.challenge_areas or ['not yet identified']}"
        )

        goals_str = "\n".join(
            f"- {g.description} (status: {g.status})" for g in state.goals
        ) or "No goals set yet"

        outcomes_str = "\n".join(
            f"- {o.goal_description}: {o.signal} - {o.detail}" for o in state.outcomes[-5:]
        ) or "No outcomes recorded yet"

        prompt = PROGRESS_CHECK_PROMPT.format(
            message=message,
            conversation_history=conversation_history or "No prior conversation.",
            family_context=family_context,
            goals=goals_str,
            active_strategies=state.active_strategies or ["none yet"],
            outcomes=outcomes_str,
        )

        return await self._gemini.generate(prompt, temperature=0.7)

    def _fallback_response(self, state: SessionState) -> str:
        """Template response when Gemini is unavailable."""
        if not state.goals:
            return (
                "It sounds like you're ready to set some goals! Let's start small "
                "and specific. What's one thing you'd like to see improve this week? "
                "For example: 'Start homework within 10 minutes of being asked' or "
                "'Complete the bedtime routine without a meltdown 3 times this week.'"
            )

        positive = [o for o in state.outcomes if o.signal == "positive"]
        if positive:
            return (
                "I can see some progress happening! Let's check in on how things "
                "have been going. Which of your goals would you like to talk about?"
            )

        return (
            "Let's check in on how things have been going. "
            "Remember, progress isn't always linear, and noticing what's "
            "happening is itself a big step."
        )

    def _track_outcomes(self, message: str, extraction: ExtractionResult, state: SessionState):
        """Track positive/negative outcome signals from the conversation."""
        message_lower = message.lower()

        for signal_word in POSITIVE_SIGNALS:
            if signal_word in message_lower:
                state.outcomes.append(Outcome(
                    goal_description=state.goals[-1].description if state.goals else "general",
                    signal="positive",
                    detail=signal_word,
                    turn=state.turn_count,
                ))
                break

        for signal_word in NEGATIVE_SIGNALS:
            if signal_word in message_lower:
                state.outcomes.append(Outcome(
                    goal_description=state.goals[-1].description if state.goals else "general",
                    signal="negative",
                    detail=signal_word,
                    turn=state.turn_count,
                ))
                break
