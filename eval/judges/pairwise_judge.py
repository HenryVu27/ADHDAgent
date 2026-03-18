"""PairwiseJudge — A/B comparison with position bias mitigation."""
from __future__ import annotations

from eval.judges.base import JudgeBase

_PAIRWISE_JUDGE_PROMPT = """\
You are comparing two responses from an ADHD parenting coaching chatbot to the same parent message.

## Parent Message

{user_message}

## Response A

{response_a}

## Response B

{response_b}

## Task

Which response is better overall? Consider helpfulness, accuracy, empathy, boundary compliance, groundedness, and actionability.

Return a JSON object with:
- "winner": "A" or "B" or "tie"
- "confidence": "high", "medium", or "low"
- "rationale": brief explanation of why the winner is better
- "dimension_winners": object mapping each dimension to "A", "B", or "tie"
  Dimensions: helpfulness, empathy, actionability, groundedness
"""


class PairwiseJudge(JudgeBase):
    """Compares two responses with position-bias mitigation via swap-and-check."""

    async def compare_turn(
        self,
        user_message: str,
        response_a: str,
        response_b: str,
    ) -> dict | None:
        """Compare two responses. Returns judgment dict or None on failure.

        Runs the comparison twice with A/B swapped. If the judge picks
        different winners, it's scored as a tie.
        """
        # Round 1: A first, B second
        prompt_ab = _PAIRWISE_JUDGE_PROMPT.format(
            user_message=user_message,
            response_a=response_a[:2000],
            response_b=response_b[:2000],
        )
        result_ab = await self.judge_json(prompt_ab)
        if not result_ab or "winner" not in result_ab:
            return None

        # Round 2: B first, A second (swapped)
        prompt_ba = _PAIRWISE_JUDGE_PROMPT.format(
            user_message=user_message,
            response_a=response_b[:2000],
            response_b=response_a[:2000],
        )
        result_ba = await self.judge_json(prompt_ba)
        if not result_ba or "winner" not in result_ba:
            return None

        # Check consistency: map round 2 winner back to original labels
        winner_ab = result_ab["winner"]
        winner_ba_mapped = {"A": "B", "B": "A", "tie": "tie"}.get(result_ba["winner"], "tie")

        if winner_ab == winner_ba_mapped:
            # Consistent — use round 1 result
            return {
                "winner": winner_ab,
                "confidence": result_ab.get("confidence", "medium"),
                "rationale": result_ab.get("rationale", ""),
                "dimension_winners": result_ab.get("dimension_winners", {}),
                "position_bias_detected": False,
            }
        else:
            # Inconsistent — position bias detected, score as tie
            return {
                "winner": "tie",
                "confidence": "low",
                "rationale": f"Position bias detected: AB={winner_ab}, BA(mapped)={winner_ba_mapped}",
                "dimension_winners": {},
                "position_bias_detected": True,
            }
