"""
Predicate Extraction Module (Layer 1).

Extracts structured predicates from parent natural language utterances.
Primary: Gemini structured JSON extraction.
Fallback: keyword-based extraction for offline/demo/testing.

Example:
    Input:  "My 7-year-old son won't do his homework and keeps getting distracted"
    Output: ExtractionResult with predicates:
        - Predicate(predicate="child_behavior", subject="avoidance", category="homework")
        - Predicate(predicate="child_behavior", subject="distraction", category="attention")
        - Predicate(predicate="child_age", subject="7", category="demographics")
"""

import logging
import re
import time

from app.config import settings
from app.llm.prompts import PREDICATE_EXTRACTION_PROMPT
from app.models.schemas import (
    ExtractionMethod,
    ExtractionResult,
    Predicate,
)

logger = logging.getLogger(__name__)

# Keyword-to-predicate mappings for fallback extraction
BEHAVIOR_KEYWORDS: dict[str, tuple[str, str]] = {
    "won't": ("avoidance", "task"),
    "refuses": ("avoidance", "task"),
    "distracted": ("distraction", "attention"),
    "can't focus": ("distraction", "attention"),
    "hyperactive": ("hyperactivity", "movement"),
    "can't sit still": ("hyperactivity", "movement"),
    "meltdown": ("emotional_dysregulation", "emotion"),
    "tantrum": ("emotional_dysregulation", "emotion"),
    "angry": ("emotional_dysregulation", "emotion"),
    "frustrated": ("emotional_dysregulation", "emotion"),
    "homework": ("academic_concern", "homework"),
    "school": ("academic_concern", "school"),
    "bedtime": ("routine_difficulty", "bedtime"),
    "morning": ("routine_difficulty", "morning"),
    "routine": ("routine_difficulty", "daily_routine"),
    "friends": ("social_difficulty", "peers"),
    "transition": ("transition_difficulty", "transitions"),
    "screen": ("transition_difficulty", "screen_time"),
}

CONCERN_KEYWORDS: dict[str, str] = {
    "worried": "parent_worry",
    "exhausted": "parent_burnout",
    "frustrated": "parent_frustration",
    "help": "seeking_help",
    "don't know": "uncertainty",
    "tried everything": "exhausted_options",
}


class PredicateExtractor:
    """Extracts structured predicates from parent utterances."""

    def __init__(self, gemini_client=None):
        self._gemini = gemini_client

    async def extract(self, message: str, conversation_context: str = "") -> ExtractionResult:
        """
        Extract predicates from a parent's message.
        Uses Gemini when available, falls back to keywords.
        """
        start = time.time()

        if self._gemini and settings.USE_LLM_EXTRACTION:
            try:
                result = await self._extract_with_gemini(message, conversation_context)
                result.raw_text = message
                logger.info(
                    f"Gemini extraction: {len(result.predicates)} predicates "
                    f"in {(time.time() - start) * 1000:.0f}ms"
                )
                return result
            except Exception as e:
                logger.warning(f"Gemini extraction failed, using fallback: {e}")

        result = self._extract_with_keywords(message)
        result.raw_text = message
        return result

    async def _extract_with_gemini(self, message: str, conversation_context: str = "") -> ExtractionResult:
        """Extract predicates using Gemini structured JSON output."""
        prompt = PREDICATE_EXTRACTION_PROMPT.format(
            message=message,
            recent_context=conversation_context or "No prior conversation.",
        )
        raw = await self._gemini.extract_json(prompt)

        predicates = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and "predicate" in item:
                    predicates.append(Predicate(
                        predicate=item.get("predicate", "unknown"),
                        subject=item.get("subject", ""),
                        category=item.get("category", ""),
                        confidence=float(item.get("confidence", 0.9)),
                    ))

        return ExtractionResult(
            predicates=predicates,
            method=ExtractionMethod.gemini,
        )

    def _extract_with_keywords(self, message: str) -> ExtractionResult:
        """Fallback extraction using keyword matching."""
        predicates: list[Predicate] = []
        message_lower = message.lower()

        for keyword, (subject, category) in BEHAVIOR_KEYWORDS.items():
            if keyword in message_lower:
                predicates.append(Predicate(
                    predicate="child_behavior",
                    subject=subject,
                    category=category,
                    confidence=0.7,
                ))

        for keyword, concern_type in CONCERN_KEYWORDS.items():
            if keyword in message_lower:
                predicates.append(Predicate(
                    predicate="parent_concern",
                    subject=concern_type,
                    category="parent_state",
                    confidence=0.7,
                ))

        age_match = re.search(r"(\d{1,2})\s*(?:year|yr|-year)", message_lower)
        if age_match:
            predicates.append(Predicate(
                predicate="child_age",
                subject=age_match.group(1),
                category="demographics",
                confidence=0.95,
            ))

        return ExtractionResult(
            predicates=predicates,
            method=ExtractionMethod.keyword_fallback,
        )
