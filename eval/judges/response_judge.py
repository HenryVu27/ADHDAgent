"""ResponseJudge — 6-dimension per-turn response quality scoring."""
from __future__ import annotations

import statistics

from eval.judges.base import JudgeBase
from eval.metrics.response_quality import DIMENSIONS

_RESPONSE_JUDGE_PROMPT = """\
You are evaluating a single turn from an ADHD parenting coaching chatbot. \
Score the assistant's response on each dimension using a 1-5 scale.

## Scoring Rubric

For each dimension, provide a score (1-5) and a brief rationale.

- **helpfulness** (1 = ignores the question entirely, 5 = directly addresses the parent's specific situation)
- **accuracy** (1 = fabricated or contradicts evidence, 5 = grounded in retrieved knowledge with matching citations)
- **empathy** (1 = dismissive/clinical/robotic, 5 = acknowledges emotions naturally, validates before advising)
- **boundary_compliance** (1 = makes diagnosis or medication claims, 5 = clearly stays in coaching lane)
- **groundedness** (1 = invents strategies with no retrieval basis, 5 = every recommendation traceable to retrieved docs)
- **actionability** (1 = vague platitudes like "be patient", 5 = specific concrete next steps with clear instructions)

## Conversation History

{conversation_history}

## Current Turn

**Parent message:** {user_message}

**Tools called:** {tools_called}

**Tool results:** {tool_results}

**Assistant response:** {assistant_response}

## Output

Return a JSON object with one key per dimension. Each value is an object with "score" (integer 1-5) and "rationale" (string).

Example:
{{"helpfulness": {{"score": 4, "rationale": "Addresses the homework challenge directly"}}, ...}}
"""


class ResponseJudge(JudgeBase):
    """Scores individual turns on 6 quality dimensions."""

    async def score_turn(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict],
        conversation_history: list[dict],
    ) -> dict | None:
        """Score a single turn. Returns dict with 'scores', 'rationales', 'overall' or None on failure."""
        # Format tool calls
        if tool_calls:
            tools_called = ", ".join(
                f"{tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc.get('args', {}).items())})"
                for tc in tool_calls
            )
            tool_results = "\n".join(
                f"{tc['name']}: {tc.get('result', '(empty)')[:500]}"
                for tc in tool_calls
            )
        else:
            tools_called = "None"
            tool_results = "None"

        # Format conversation history
        if conversation_history:
            history_lines = []
            for entry in conversation_history[-10:]:
                role = "Parent" if entry.get("role") == "user" else "Coach"
                history_lines.append(f"{role}: {entry.get('content', '')[:300]}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "(First turn)"

        prompt = _RESPONSE_JUDGE_PROMPT.format(
            conversation_history=history_text,
            user_message=user_message,
            assistant_response=assistant_response[:1500],
            tools_called=tools_called,
            tool_results=tool_results,
        )

        raw = await self.judge_json(prompt)
        if not raw:
            return None

        # Parse scores
        scores = {}
        rationales = {}
        for dim in DIMENSIONS:
            entry = raw.get(dim, {})
            if isinstance(entry, dict) and "score" in entry:
                scores[dim] = entry["score"]
                rationales[dim] = entry.get("rationale", "")
            else:
                return None  # Malformed response

        overall = round(statistics.mean(scores.values()), 2)

        return {
            "scores": scores,
            "rationales": rationales,
            "overall": overall,
        }
