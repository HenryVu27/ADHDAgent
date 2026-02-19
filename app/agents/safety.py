"""
Safety monitor using Gemini semantic classification.

All safety decisions go through the LLM for contextual understanding,
avoiding false positives from naive keyword matching.
"""

import logging

from app.config import settings
from app.llm.prompts import SAFETY_CHECK_PROMPT
from app.models.schemas import SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

CRISIS_RESPONSE = (
    "I want to make sure you and your family are safe. What you're describing "
    "sounds like it needs immediate professional support.\n\n"
    "**If there is an immediate safety concern:**\n"
    "- Call 911 for emergencies\n"
    "- Call or text 988 for the Suicide & Crisis Lifeline\n"
    "- Text HOME to 741741 for the Crisis Text Line\n\n"
    "Please reach out to a professional who can help right away. "
    "I'll be here whenever you're ready to continue our conversation."
)

OUT_OF_SCOPE_RESPONSE = (
    "That's an important topic, but it's outside what I'm able to help with. "
    "Questions about {topic} should be directed to your child's healthcare "
    "provider, who knows your family's specific situation.\n\n"
    "I'm here to help with behavioral strategies, daily routines, "
    "and practical parenting approaches. What would you like to work on?"
)


class SafetyMonitor:
    """LLM-based safety enforcement."""

    def __init__(self, gemini_client=None):
        self._gemini = gemini_client

    async def check(self, message: str, conversation_context: str = "") -> SafetyResult:
        """Run Gemini semantic safety check."""
        if self._gemini and settings.USE_LLM_SAFETY:
            try:
                result = await self._gemini_check(message, conversation_context)
                if result.level != SafetyLevel.safe:
                    logger.info(f"Safety triggered: {result.level} - {result.detected_topic}")
                return result
            except Exception as e:
                logger.warning(f"Safety check failed, defaulting to safe: {e}")

        return SafetyResult(level=SafetyLevel.safe)

    async def _gemini_check(self, message: str, conversation_context: str = "") -> SafetyResult:
        """Gemini semantic classification."""
        prompt = SAFETY_CHECK_PROMPT.format(
            message=message,
            recent_context=conversation_context or "No prior conversation.",
        )
        result = await self._gemini.extract_json(prompt)

        if isinstance(result, dict):
            level_str = result.get("level", "safe")
            topic = result.get("detected_topic")

            if level_str == "crisis":
                return SafetyResult(
                    level=SafetyLevel.crisis,
                    detected_topic=topic,
                    response_override=CRISIS_RESPONSE,
                )
            elif level_str == "out_of_scope":
                return SafetyResult(
                    level=SafetyLevel.out_of_scope,
                    detected_topic=topic,
                    response_override=OUT_OF_SCOPE_RESPONSE.format(topic=topic or "that"),
                )

        return SafetyResult(level=SafetyLevel.safe)
