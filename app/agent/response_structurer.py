"""Post-agent response structuring via lightweight Flash call.

Extracts structured metadata from the agent's free-text response
without disrupting the ReAct loop or streaming.
"""

import logging

from app.models.schemas import AgentResponse

logger = logging.getLogger(__name__)

STRUCTURE_PROMPT = """Extract structured metadata from this coaching response.
The parent's message and the coach's response are provided.

Parent message: {user_message}

Coach response: {response_text}

Return a JSON object with:
- "response_text": the coach's response text (unchanged)
- "concerns_addressed": list of parent concerns this response addresses (short phrases)
- "strategies_referenced": list of strategy/document IDs mentioned or used
- "follow_up_question": the follow-up question asked, if any (empty string if none)
- "needs_more_info": true if the coach is asking for more information

Return ONLY JSON, no markdown."""


async def structure_response(
    response_text: str,
    user_message: str,
    gemini_client,
) -> AgentResponse:
    """Extract structured metadata from the agent's free-text response.

    On any error, returns AgentResponse with just the raw text (graceful degradation).
    """
    try:
        prompt = STRUCTURE_PROMPT.format(
            user_message=user_message[:500],
            response_text=response_text[:2000],
        )
        raw = await gemini_client.extract_json(
            prompt,
            temperature=0.0,
            max_output_tokens=512,
            timeout=5.0,
            disable_thinking=True,
        )
        if isinstance(raw, dict):
            raw["response_text"] = response_text
            return AgentResponse(**raw)
        return AgentResponse(response_text=response_text)
    except Exception as e:
        logger.debug("Response structuring failed: %s", e)
        return AgentResponse(response_text=response_text)
