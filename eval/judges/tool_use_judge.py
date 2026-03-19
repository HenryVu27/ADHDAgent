"""ToolUseJudge — heuristic + LLM evaluation of agent tool usage."""
from __future__ import annotations

import re
import statistics

from eval.judges.base import JudgeBase
from eval.metrics.tool_use import compute_tool_f1

# Words that suggest a specific problem needing knowledge base search
_CHALLENGE_PATTERNS = re.compile(
    r"(meltdown|tantrum|homework|focus|distract|angry|aggressive|sleep|morning routine|"
    r"medication|school|teacher|impulse|hyperactiv|organiz|forget|losing things|"
    r"emotional|overwhelm|anxiety|frustrat|struggle|difficult|challenging|problem)",
    re.IGNORECASE,
)

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|good morning|good evening|thanks|thank you|ok|okay|sure|yes|no)\b",
    re.IGNORECASE,
)


def run_heuristics(
    user_message: str,
    assistant_response: str,
    tool_calls: list[dict],
) -> list[dict]:
    """Run rule-based heuristic checks. Returns list of flag dicts."""
    flags = []

    # Check result utilization: tool was called but result keywords not in response
    for tc in tool_calls:
        result_text = tc.get("result", "")
        if not result_text:
            continue
        # Extract key phrases from result (first 3 significant words)
        result_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", result_text[:300])
        )
        response_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", assistant_response)
        )
        overlap = result_words & response_words
        if len(overlap) < 2 and len(result_words) > 3:
            flags.append({
                "type": "low_result_utilization",
                "tool": tc["name"],
                "detail": f"Tool returned content but response shares <2 key words with result",
            })

    # Check missed search: user describes a challenge but no search was called
    if not any(tc["name"] == "search_knowledge_base" for tc in tool_calls):
        if _CHALLENGE_PATTERNS.search(user_message) and not _GREETING_PATTERNS.match(user_message.strip()):
            flags.append({
                "type": "likely_missed_search",
                "detail": "User message contains challenge keywords but search_knowledge_base was not called",
            })

    return flags


_TOOL_USE_JUDGE_PROMPT = """\
You are evaluating tool usage in a single turn of an ADHD parenting coaching chatbot.

The chatbot has these tools available:
- search_knowledge_base(query): Search for evidence-based parenting strategies
- update_family_profile(updates): Save new info about the family
- get_family_profile(): Retrieve current family profile
- track_outcome(strategy_name, signal, detail): Record strategy results
- manage_goals(action, description, ...): Create/update/complete goals

## Conversation History

{conversation_history}

## Current Turn

**Parent message:** {user_message}

**Tools called:** {tools_called}

**Tool results:** {tool_results}

**Assistant response:** {assistant_response}

## Heuristic Flags (pre-computed)

{heuristic_flags}

## Task

Evaluate each tool call and identify any tools that should have been called but weren't.

Return a JSON object with:
- "tool_verdicts": array of objects, one per tool that was called, each with:
  - "name": tool name
  - "verdict": "appropriate" or "unnecessary"
  - "argument_quality": integer 1-5 (was the query/args well-formed?)
  - "result_utilization": integer 1-5 (did the response use the tool's output?)
- "missed_tools": array of tool names that should have been called but weren't

If no tools were called and none should have been, return: {{"tool_verdicts": [], "missed_tools": []}}
"""


class ToolUseJudge(JudgeBase):
    """Two-layer tool use evaluation: heuristics + LLM judge."""

    async def evaluate_turn(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict],
        conversation_history: list[dict],
    ) -> dict | None:
        """Evaluate tool use for a single turn. Returns metrics dict or None on failure."""
        # Layer 1: heuristics
        heuristic_flags = run_heuristics(user_message, assistant_response, tool_calls)

        # Format inputs for LLM judge
        if tool_calls:
            tools_called = ", ".join(
                f"{tc['name']}({', '.join(f'{k}={v!r}' for k, v in tc.get('args', {}).items())})"
                for tc in tool_calls
            )
            tool_results = "\n".join(
                f"{tc['name']}: {tc.get('result', '(empty)')[:500]}" for tc in tool_calls
            )
        else:
            tools_called = "None"
            tool_results = "None"

        if conversation_history:
            history_lines = []
            for entry in conversation_history[-10:]:
                role = "Parent" if entry.get("role") == "user" else "Coach"
                history_lines.append(f"{role}: {entry.get('content', '')[:300]}")
            history_text = "\n".join(history_lines)
        else:
            history_text = "(First turn)"

        flags_text = "\n".join(f"- {f['type']}: {f.get('detail', '')}" for f in heuristic_flags) or "None"

        prompt = _TOOL_USE_JUDGE_PROMPT.format(
            conversation_history=history_text,
            user_message=user_message,
            assistant_response=assistant_response[:1500],
            tools_called=tools_called,
            tool_results=tool_results,
            heuristic_flags=flags_text,
        )

        raw = await self.judge_json(prompt)
        if not raw:
            return None

        verdicts = raw.get("tool_verdicts", [])
        missed = raw.get("missed_tools", [])

        # Compute precision/recall/F1
        f1_scores = compute_tool_f1(verdicts, missed)

        # Compute mean argument accuracy and result utilization
        arg_scores = [v.get("argument_quality", 3) for v in verdicts if "argument_quality" in v]
        util_scores = [v.get("result_utilization", 3) for v in verdicts if "result_utilization" in v]

        by_tool = {}
        for v in verdicts:
            by_tool[v["name"]] = {
                "verdict": v["verdict"],
                "argument_quality": v.get("argument_quality", 3),
                "result_utilization": v.get("result_utilization", 3),
            }

        return {
            **f1_scores,
            "argument_accuracy": round(statistics.mean(arg_scores), 2) if arg_scores else 5,
            "result_utilization": round(statistics.mean(util_scores), 2) if util_scores else 5,
            "by_tool": by_tool,
            "heuristic_flags": heuristic_flags,
        }
