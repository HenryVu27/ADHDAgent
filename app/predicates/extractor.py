"""
Predicate Extraction Module.

Extracts structured predicates from parent natural language utterances.
These predicates are then fed into the ASP reasoning engine to determine
valid conversation moves.

Example:
    Input:  "My son won't do his homework and keeps getting distracted"
    Output: [
        {"predicate": "child_behavior", "subject": "avoidance", "category": "homework"},
        {"predicate": "child_behavior", "subject": "distraction", "category": "homework"},
        {"predicate": "parent_concern", "subject": "academic_performance", "category": "homework"},
    ]

In production, this uses an LLM with structured output to extract predicates.
The fallback uses keyword matching for offline/demo use.
"""

import json
import re


# Keyword-to-predicate mappings for fallback extraction
BEHAVIOR_KEYWORDS = {
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

CONCERN_KEYWORDS = {
    "worried": "parent_worry",
    "exhausted": "parent_burnout",
    "frustrated": "parent_frustration",
    "help": "seeking_help",
    "don't know": "uncertainty",
    "tried everything": "exhausted_options",
}

# Prompt template for LLM-based extraction
EXTRACTION_PROMPT = """You are a predicate extraction system for an ADHD parenting coach.
Given a parent's message, extract structured predicates that capture:
1. Child behaviors being described
2. Situations or contexts mentioned
3. Parent emotions or concerns
4. Specific ADHD-related challenges

Return a JSON array of predicates, each with:
- "predicate": the type (child_behavior, parent_concern, situation, challenge)
- "subject": the specific thing described
- "category": broader category it falls under

Parent message: {message}

Return ONLY valid JSON array, no other text."""


class PredicateExtractor:
    """Extracts structured predicates from parent utterances."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def extract(self, message: str) -> list[dict]:
        """
        Extract predicates from a parent's message.
        Uses keyword matching as fallback, LLM for production.
        """
        if self.use_llm:
            return self._extract_with_llm(message)
        return self._extract_with_keywords(message)

    def _extract_with_keywords(self, message: str) -> list[dict]:
        """Fallback extraction using keyword matching."""
        predicates = []
        message_lower = message.lower()

        # Extract behavior predicates
        for keyword, (subject, category) in BEHAVIOR_KEYWORDS.items():
            if keyword in message_lower:
                predicates.append({
                    "predicate": "child_behavior",
                    "subject": subject,
                    "category": category,
                })

        # Extract concern predicates
        for keyword, concern_type in CONCERN_KEYWORDS.items():
            if keyword in message_lower:
                predicates.append({
                    "predicate": "parent_concern",
                    "subject": concern_type,
                    "category": "parent_state",
                })

        # Extract age if mentioned
        age_match = re.search(r"(\d{1,2})\s*(?:year|yr)", message_lower)
        if age_match:
            predicates.append({
                "predicate": "child_age",
                "subject": age_match.group(1),
                "category": "demographics",
            })

        return predicates

    def _extract_with_llm(self, message: str) -> list[dict]:
        """Extract predicates using LLM with structured output."""
        try:
            from openai import OpenAI
            from app.config import settings

            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=[
                    {"role": "user", "content": EXTRACTION_PROMPT.format(message=message)}
                ],
                temperature=0,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception:
            # Fall back to keyword extraction
            return self._extract_with_keywords(message)
