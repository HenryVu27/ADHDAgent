"""Message decomposition for multi-concern parent messages.

Splits compound messages ("homework AND bedtime AND mornings") into
individual concerns so the agent can address each one. Simple messages
and greetings bypass the LLM call entirely.
"""

import logging
import re

from app.models.schemas import Concern, DecomposedMessage

logger = logging.getLogger(__name__)

_SHORT_MESSAGE_THRESHOLD = 40

_SIMPLE_PATTERNS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|got it|sounds good)",
    re.IGNORECASE,
)

DECOMPOSE_PROMPT = """You are analyzing a parent's message to an ADHD coaching chatbot.
Identify each distinct concern or topic the parent is raising.

Parent message: {message}

Rules:
- A "concern" is a separate topic the parent wants help with
- Greetings, thanks, and brief acknowledgments are a single "greeting" concern
- Venting or expressing emotions is a single "venting" concern
- "My kid won't do homework and has meltdowns at bedtime" = 2 concerns
- "Tell me about homework strategies" = 1 concern
- Outcome reports ("we tried X and it worked/didn't work") are "outcome_report"

Return ONLY JSON:
{{"concerns": [{{"description": "short description", "intent": "strategy_request|venting|outcome_report|question|greeting", "search_query": "keywords for knowledge base search if needed"}}], "is_multi_concern": true/false}}"""


async def decompose_message(
    message: str,
    gemini_client,
) -> DecomposedMessage:
    stripped = message.strip()
    if len(stripped) < _SHORT_MESSAGE_THRESHOLD or _SIMPLE_PATTERNS.match(stripped):
        intent = "greeting" if _SIMPLE_PATTERNS.match(stripped) else "strategy_request"
        return DecomposedMessage(
            concerns=[Concern(description=stripped, intent=intent)],
            is_multi_concern=False,
            original_message=message,
        )

    if gemini_client is None:
        return DecomposedMessage(
            concerns=[Concern(description=stripped, intent="strategy_request")],
            is_multi_concern=False,
            original_message=message,
        )

    try:
        prompt = DECOMPOSE_PROMPT.format(message=message[:1000])
        raw = await gemini_client.extract_json(
            prompt,
            temperature=0.0,
            max_output_tokens=512,
            timeout=5.0,
            disable_thinking=True,
        )
        if isinstance(raw, dict):
            result = DecomposedMessage(**raw, original_message=message)
            if len(result.concerns) > 4:
                result.concerns = result.concerns[:4]
            return result
    except Exception as e:
        logger.debug("Decomposition failed, using single concern: %s", e)

    return DecomposedMessage(
        concerns=[Concern(description=stripped, intent="strategy_request")],
        is_multi_concern=False,
        original_message=message,
    )
