"""CoherenceJudge — conversation-level multi-turn coherence scoring."""
from __future__ import annotations

import statistics

from eval.judges.base import JudgeBase
from eval.metrics.coherence import COHERENCE_DIMENSIONS

MIN_TURNS = 5

_COHERENCE_JUDGE_PROMPT = """\
You are evaluating a full multi-turn conversation from an ADHD parenting coaching chatbot.
Score the conversation on four coherence dimensions (1-5 each).

## Scoring Rubric

- **progressive_profiling** (1 = never asks about the family / interrogates upfront, 5 = learns naturally over turns)
- **repetition_avoidance** (1 = repeats questions or advice, 5 = no repetition at all)
- **follow_up** (1 = never checks on recommended strategies, 5 = appropriately follows up)
- **topic_management** (1 = forces topic / ignores shifts, 5 = handles changes gracefully)

## Full Conversation ({n_turns} turns)

{transcript}

## Family Profile at End

{family_profile}

## Output

Return a JSON object with:
- "scores": object with one key per dimension, integer 1-5
- "rationale": object with one key per dimension, brief explanation
- "notable_moments": array of objects with "turn" (int), "type" (string), "detail" (string) for specific good/bad moments
"""


class CoherenceJudge(JudgeBase):
    """Scores conversation-level coherence across 4 dimensions."""

    async def score_conversation(
        self,
        turns: list[dict],
        family_profile: dict,
    ) -> dict | None:
        """Score a full conversation. Returns None if too short or on failure."""
        if len(turns) < MIN_TURNS:
            return None

        # Build transcript
        lines = []
        for t in turns:
            lines.append(f"[Turn {t['turn']}] Parent: {t['user_message']}")
            if t.get("tool_calls"):
                tool_names = ", ".join(tc["name"] for tc in t["tool_calls"])
                lines.append(f"  (Tools used: {tool_names})")
            lines.append(f"  Coach: {t['assistant_response'][:500]}")
        transcript = "\n".join(lines)

        profile_text = "\n".join(f"- {k}: {v}" for k, v in family_profile.items() if v) or "(Empty)"

        prompt = _COHERENCE_JUDGE_PROMPT.format(
            n_turns=len(turns),
            transcript=transcript,
            family_profile=profile_text,
        )

        raw = await self.judge_json(prompt, max_tokens=2048)
        if not raw or "scores" not in raw:
            return None

        scores = raw["scores"]
        for dim in COHERENCE_DIMENSIONS:
            if dim not in scores or not isinstance(scores[dim], (int, float)):
                return None

        overall = round(statistics.mean(scores[dim] for dim in COHERENCE_DIMENSIONS), 2)

        return {
            "scores": scores,
            "rationale": raw.get("rationale", {}),
            "overall": overall,
            "notable_moments": raw.get("notable_moments", []),
        }
