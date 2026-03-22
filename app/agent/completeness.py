"""Completeness checker for multi-concern responses.

After the ReAct agent produces a response, this module checks whether
all decomposed concerns from the parent's message were addressed.
"""

import logging

from pydantic import BaseModel, Field

from app.models.schemas import Concern

logger = logging.getLogger(__name__)


class CompletenessResult(BaseModel):
    all_addressed: bool = True
    addressed: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    supplement: str = ""


COMPLETENESS_PROMPT = """Check if this coaching response addresses all the parent's concerns.

Parent's concerns:
{concerns_list}

Coach's response:
{response_text}

For each concern, determine if the response meaningfully addresses it
(even briefly). A concern is "addressed" if the response acknowledges it,
gives advice about it, or asks a follow-up about it.

Return ONLY JSON:
{{"addressed": ["concern 1 description", ...], "missed": ["concern 2 description", ...]}}"""


SUPPLEMENT_TEMPLATE = (
    "\n\nYou also mentioned {missed_summary}. "
    "I want to make sure I help with that too -- "
    "could we talk about that next?"
)


async def check_completeness(
    response_text: str,
    concerns: list[Concern],
    gemini_client,
) -> CompletenessResult:
    if len(concerns) <= 1:
        return CompletenessResult(all_addressed=True)

    if gemini_client is None:
        return CompletenessResult(all_addressed=True)

    try:
        concerns_list = "\n".join(f"- {c.description}" for c in concerns)
        prompt = COMPLETENESS_PROMPT.format(
            concerns_list=concerns_list,
            response_text=response_text[:2000],
        )
        raw = await gemini_client.extract_json(
            prompt,
            temperature=0.0,
            max_output_tokens=256,
            timeout=5.0,
            disable_thinking=True,
        )
        if isinstance(raw, dict):
            missed = raw.get("missed", [])
            addressed = raw.get("addressed", [])
            supplement = ""
            if missed:
                missed_summary = " and ".join(missed[:2])
                supplement = SUPPLEMENT_TEMPLATE.format(missed_summary=missed_summary)
            return CompletenessResult(
                all_addressed=len(missed) == 0,
                addressed=addressed,
                missed=missed,
                supplement=supplement,
            )
    except Exception as e:
        logger.debug("Completeness check failed: %s", e)

    return CompletenessResult(all_addressed=True)
