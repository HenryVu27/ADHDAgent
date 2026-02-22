"""ConversationAnalyzer — async post-turn quality analysis.

Follows the MemoryManager pattern: fires after the response is sent,
never blocks the response pipeline. Flags quality issues like broken
promises, missed tool calls, and tone problems.
"""

import logging
from datetime import datetime, timezone

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import AnalysisFlag, EnrichedTrace, TurnAnalysis

logger = logging.getLogger(__name__)

ANALYZER_PROMPT = """You are an AI conversation quality analyzer for an ADHD parenting coaching chatbot.

Analyze this single turn and check for these issues:

1. **broken_promise**: The assistant says "I'll search/find/look for" something but the tool results show no relevant content was delivered, OR no tool was called at all.
2. **missed_tool_call**: The parent describes a specific problem/challenge that would benefit from searching the knowledge base, but search_knowledge_base was NOT called.
3. **incomplete_response**: The response trails off, changes topic mid-sentence, or fails to address the parent's actual question.
4. **circular**: The assistant repeats a question already answered or gives advice that was already given in the recent conversation history below.
5. **tone_issue**: The response is too clinical, dismissive, uses jargon, or fails to validate the parent's feelings when they express frustration/worry.

## Recent Conversation History

{conversation_history}

## Current Turn

**User message:** {user_message}

**Assistant response:** {assistant_response}

**Tools called:** {tools_called}

**Tool results summary:** {tool_results}

## Output

Return a JSON object with:
- "flags": array of objects, each with "flag_type" (one of the 5 types above), "severity" ("info"|"warning"|"error"), "description" (brief explanation), "evidence" (quote from the message/response)
- "quality_score": float 0.0-1.0 (1.0 = perfect, 0.0 = completely broken)
- "summary": one-sentence assessment
- "tool_call_assessment": "appropriate" | "missed" | "unnecessary"

If no issues found, return: {{"flags": [], "quality_score": 1.0, "summary": "No issues detected.", "tool_call_assessment": "appropriate"}}
"""


class ConversationAnalyzer:
    """Async post-turn conversation quality analyzer."""

    def __init__(self, session_store: SessionStoreBase, gemini_client):
        self._store = session_store
        self._gemini = gemini_client

    async def analyze_turn(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        assistant_response: str,
        enriched_trace: EnrichedTrace,
    ) -> TurnAnalysis | None:
        """Analyze a single turn for quality issues. Returns None on error."""
        if not self._gemini:
            return None

        try:
            # Fetch recent conversation history for circular detection
            state = self._store.get(session_id)
            recent_history = state.conversation_history[-5:]
            if recent_history:
                history_lines = []
                for entry in recent_history:
                    if entry.get("role") == "user":
                        history_lines.append(f"Parent: {entry['content']}")
                    elif entry.get("role") == "assistant":
                        history_lines.append(f"Coach: {entry['content']}")
                conversation_history = "\n".join(history_lines)
            else:
                conversation_history = "(No prior turns)"

            # Summarize tool calls and results for the prompt
            tools_called = "None"
            tool_results = "None"
            if enriched_trace.tool_calls:
                tools_called = ", ".join(
                    f"{tc.name}({', '.join(f'{k}={v!r}' for k, v in tc.args.items())})"
                    for tc in enriched_trace.tool_calls
                )
                results_parts = []
                for tc in enriched_trace.tool_calls:
                    result_preview = tc.result[:300] if tc.result else "(empty)"
                    results_parts.append(f"{tc.name}: {result_preview}")
                tool_results = "\n".join(results_parts)

            # Truncate response at sentence boundary instead of mid-text
            response_text = assistant_response
            if len(response_text) > 1200:
                truncated = response_text[:1200]
                last_period = truncated.rfind(".")
                if last_period > 800:
                    response_text = truncated[:last_period + 1]
                else:
                    response_text = truncated

            prompt = ANALYZER_PROMPT.format(
                conversation_history=conversation_history,
                user_message=user_message,
                assistant_response=response_text,
                tools_called=tools_called,
                tool_results=tool_results,
            )

            result = await self._gemini.extract_json(prompt, temperature=0.0, max_output_tokens=1024)

            if not isinstance(result, dict):
                logger.warning("Analyzer returned non-dict for session %s turn %d", session_id, turn)
                return None

            flags = []
            for f in result.get("flags", []):
                if isinstance(f, dict) and "flag_type" in f:
                    flags.append(AnalysisFlag(
                        flag_type=f["flag_type"],
                        severity=f.get("severity", "warning"),
                        description=f.get("description", ""),
                        evidence=f.get("evidence", ""),
                    ))

            analysis = TurnAnalysis(
                session_id=session_id,
                turn=turn,
                flags=flags,
                quality_score=float(result.get("quality_score", 1.0)),
                summary=result.get("summary", ""),
                tool_call_assessment=result.get("tool_call_assessment", ""),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            self._store.save_analysis(session_id, analysis)

            if flags:
                logger.info(
                    "Analyzer flagged %d issues for session %s turn %d: %s",
                    len(flags), session_id, turn,
                    [f.flag_type for f in flags],
                )

            return analysis

        except Exception as e:
            logger.error("Analyzer failed for session %s turn %d: %s", session_id, turn, e)
            return None
