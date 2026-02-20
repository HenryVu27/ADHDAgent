"""Rule-based model routing for per-turn complexity classification.

Classifies each turn as fast/standard/complex and returns a model selector
callable for create_react_agent's `model` parameter.
"""

import logging
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent.state import CoachingState
from app.config import settings

logger = logging.getLogger(__name__)

# Patterns for trivial messages (fast tier)
TRIVIAL_PATTERNS = re.compile(
    r"^(yes|no|ok|okay|sure|thanks|thank you|yep|nope|yeah|nah|got it|cool|great|right|hmm|hm)\.?!?$",
    re.IGNORECASE,
)

# Keywords that indicate safety-adjacent content (complex tier)
SAFETY_KEYWORDS = {
    # Medication names
    "adderall", "ritalin", "concerta", "vyvanse", "strattera", "focalin",
    "methylphenidate", "amphetamine", "dexedrine", "guanfacine", "intuniv",
    "clonidine", "wellbutrin", "medication", "meds", "dosage", "prescri",
    # Diagnosis
    "diagnos", "adhd test", "evaluation", "assess",
    # Safety
    "harm", "hurt", "suicide", "kill", "abuse", "neglect", "danger",
    # Legal
    "legal", "lawsuit", "iep rights", "section 504",
}


def classify_complexity(state: CoachingState) -> str:
    """Classify the current turn's complexity tier.

    Rules evaluated in order:
    1. Short trivial pattern + turn > 1 -> "fast"
    2. Safety-adjacent keywords -> "complex"
    3. Turn count > 10 -> "complex"
    4. Everything else -> "standard"

    Returns:
        "fast", "standard", or "complex"
    """
    messages = state.get("messages", [])

    # Find latest human message
    latest_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_human = msg
            break

    if not latest_human:
        return "standard"

    text = latest_human.content.strip()
    turn_count = state.get("turn_count", 0)

    # Rule 1: Short trivial message in active conversation
    if len(text) < 20 and turn_count > 1 and TRIVIAL_PATTERNS.match(text):
        logger.debug("Model tier: fast (trivial pattern)")
        return "fast"

    # Rule 2: Safety-adjacent keywords
    text_lower = text.lower()
    if any(kw in text_lower for kw in SAFETY_KEYWORDS):
        logger.debug("Model tier: complex (safety keywords)")
        return "complex"

    # Rule 3: Long conversation
    if turn_count > 10:
        logger.debug("Model tier: complex (long conversation)")
        return "complex"

    # Rule 4: Default
    return "standard"


def create_model_selector():
    """Create a callable that returns the appropriate ChatGoogleGenerativeAI
    based on the model_tier set in state.

    Returns a function compatible with create_react_agent's `model` parameter:
        (state, config) -> BaseChatModel
    """
    # Cache model instances to avoid re-creating them each call
    _cache: dict[str, ChatGoogleGenerativeAI] = {}

    tier_to_model = {
        "fast": settings.GEMINI_MODEL_FAST,
        "standard": settings.GEMINI_MODEL_STANDARD,
        "complex": settings.GEMINI_MODEL_COMPLEX,
    }

    def selector(state, config=None):
        tier = state.get("model_tier", "standard")
        model_name = tier_to_model.get(tier, settings.GEMINI_MODEL_STANDARD)

        if model_name not in _cache:
            _cache[model_name] = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.7,
                max_output_tokens=1024,
            )
            logger.info("Cached model instance for tier=%s model=%s", tier, model_name)

        return _cache[model_name]

    return selector
