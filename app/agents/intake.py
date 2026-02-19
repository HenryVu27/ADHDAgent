"""
Intake agent with Gemini-powered empathetic acknowledgments.

Gathers family context through 5 structured questions, using Gemini
to generate warm acknowledgments of what the parent shared.
Extracts family profile data from predicates as they come in.
"""

import logging

from app.agents.base import BaseAgent
from app.config import settings
from app.llm.prompts import INTAKE_ACKNOWLEDGMENT_PROMPT
from app.models.schemas import ExtractionResult, RulesDecision, SessionState

logger = logging.getLogger(__name__)

INTAKE_QUESTIONS = [
    "How old is your child, and when were they diagnosed with ADHD?",
    "What are the biggest daily challenges you face related to your child's ADHD?",
    "What strategies have you tried so far? What has worked or not worked?",
    "Are there specific situations (homework, bedtime, transitions) that are hardest?",
    "What does a good day look like for your family?",
]

# Template acknowledgments when Gemini is unavailable
FALLBACK_ACKS = [
    "Welcome! I'm here to help you build strategies that work for your family. Let's start by getting to know your situation.\n\n",
    "Thank you for sharing that. It helps me understand your family better. ",
    "That's really helpful context. I appreciate you being so open. ",
    "I can hear how important this is to you. Thank you for sharing. ",
    "That gives me a good picture of what things look like day to day. ",
]


class IntakeAgent(BaseAgent):
    """Gathers family context through structured intake with empathetic acknowledgments."""

    name = "intake"
    description = "Gathers family context and child profile through guided intake"

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
        progress = state.intake_question_index

        # Update family profile from predicates
        self._update_profile(extraction, state)

        # First turn: welcome + first question
        if progress == 0:
            state.intake_question_index = 1
            return (
                "Welcome! I'm here to help you build strategies that work for your "
                "family. I'm not a doctor or therapist, but I can share evidence-based "
                "parenting approaches that many families find helpful.\n\n"
                "Let's start by getting to know your situation.\n\n"
                f"{INTAKE_QUESTIONS[0]}"
            )

        # Generate acknowledgment + next question
        if progress < len(INTAKE_QUESTIONS):
            state.intake_question_index = progress + 1
            ack = await self._generate_acknowledgment(message, state, progress)
            return f"{ack}{INTAKE_QUESTIONS[progress]}"

        # Intake complete
        return (
            "Thank you for walking me through all of that. I have a much better "
            "picture of your family's situation now.\n\n"
            "Based on what you've shared, I'd like to suggest some strategies "
            "that other families in similar situations have found helpful. "
            "Would you like to start with the area that feels most challenging right now?"
        )

    async def _generate_acknowledgment(self, message: str, state: SessionState, progress: int) -> str:
        """Generate an empathetic acknowledgment using Gemini or fallback."""
        if self._gemini and settings.USE_LLM_RESPONSES:
            try:
                context = f"Child age: {state.family_profile.child_age or 'unknown'}, Challenges: {state.family_profile.challenge_areas}"
                prompt = INTAKE_ACKNOWLEDGMENT_PROMPT.format(
                    parent_message=message,
                    child_name=state.family_profile.child_name or "not yet shared",
                    context=context,
                    next_question=INTAKE_QUESTIONS[progress],
                )
                return await self._gemini.generate(prompt, temperature=0.7) + "\n\n"
            except Exception as e:
                logger.warning(f"Gemini ack generation failed: {e}")

        return FALLBACK_ACKS[min(progress, len(FALLBACK_ACKS) - 1)]

    def _update_profile(self, extraction: ExtractionResult, state: SessionState):
        """Extract family profile data from predicates."""
        for pred in extraction.predicates:
            if pred.predicate == "child_age" and pred.subject:
                state.family_profile.child_age = pred.subject

            if pred.predicate == "child_behavior" and pred.category:
                if pred.category not in state.family_profile.challenge_areas:
                    state.family_profile.challenge_areas.append(pred.category)

            if pred.predicate == "situation" and pred.subject:
                if pred.subject not in state.family_profile.hardest_situations:
                    state.family_profile.hardest_situations.append(pred.subject)
