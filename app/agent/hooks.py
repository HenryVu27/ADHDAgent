"""Context assembly for the ReAct agent.

prepare_context: runs as the pre_model_hook in the ReAct agent.
  - Builds system prompt from session state (family profile, goals, summary, episodes).
  - Trims conversation history to recent turns.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import build_conversation_state, build_system_prompt
from app.agent.store_protocol import SessionStoreBase
from app.agent.state import CoachingState
from app.config import settings

logger = logging.getLogger(__name__)


def _estimate_chars(messages) -> int:
    """Sum the character length of all message contents."""
    total = 0
    for m in messages:
        content = m.content if hasattr(m, "content") else ""
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(
                len(part.get("text", "")) if isinstance(part, dict) else len(str(part))
                for part in content
            )
    return total


def create_prepare_context(
    session_store: SessionStoreBase,
    event_bus=None,
):
    """Create the prepare_context closure used as pre_model_hook."""

    async def prepare_context(state: CoachingState):
        """Build system prompt and trim conversation for the LLM."""
        messages = state["messages"]
        session_id = state.get("session_id", "default")

        # Build system prompt via context assembly helpers
        session_state = session_store.get(session_id)

        # Load rolling summary if available
        latest_summary = session_store.get_latest_summary(session_id)
        summary_text = latest_summary.summary if latest_summary else ""

        # Load recent episodes into summary context
        recent_episodes = session_store.get_recent_episodes(session_id, limit=5)
        if recent_episodes:
            episode_lines = []
            for ep in recent_episodes:
                line = f"- [{ep.event_type}] {ep.summary}"
                if ep.emotional_context:
                    line += f" (mood: {ep.emotional_context})"
                episode_lines.append(line)
            episodes_text = "\n\nKey moments:\n" + "\n".join(episode_lines)
            summary_text = (summary_text + episodes_text) if summary_text else episodes_text

        # Load recent tool results for cross-turn evidence
        recent_tool_results = session_store.get_recent_tool_results(session_id, limit=3)

        system_prompt = build_system_prompt(
            profile=session_state.family_profile,
            active_strategies=session_state.active_strategies,
            goals=session_state.goals,
            outcomes=session_state.outcomes,
            session_summary=summary_text,
        )

        # Build conversation state block
        turn_count = session_state.turn_count
        phase = session_state.phase.value if hasattr(session_state.phase, "value") else str(session_state.phase)

        # Get tool names from the last trace, if available
        recent_tool_names: list[str] | None = None
        try:
            traces = session_store.get_traces(session_id)
            if traces:
                last_trace = traces[-1]
                if last_trace.tool_calls:
                    recent_tool_names = [tc.name for tc in last_trace.tool_calls]
        except Exception:
            pass

        # Derive active topic from the last user message
        active_topic = ""
        user_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if user_messages:
            last_user = user_messages[-1].content
            if isinstance(last_user, str) and last_user.strip():
                active_topic = last_user.strip()[:60]

        state_block = build_conversation_state(
            turn=turn_count,
            phase=phase,
            recent_tool_calls=recent_tool_names,
            active_topic=active_topic,
        )
        system_prompt = state_block + "\n\n" + system_prompt

        if recent_tool_results:
            evidence_lines = []
            for tr in recent_tool_results:
                evidence_lines.append(f"[Turn {tr.turn}, query: \"{tr.query}\"]:\n{tr.result_text[:2000]}")
            evidence_block = "\n\n<prior-search-evidence>\n" + "\n---\n".join(evidence_lines) + "\n</prior-search-evidence>"
            system_prompt += evidence_block

        # Trim conversation: message count cap first, then character budget
        max_messages = settings.CONTEXT_WINDOW_TURNS * 2
        conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        if len(conversation_messages) > max_messages:
            conversation_messages = conversation_messages[-max_messages:]

        # Character budget trimming (drops oldest messages until within budget)
        remaining_budget = settings.CONTEXT_MAX_CHARS - len(system_prompt)
        while (
            _estimate_chars(conversation_messages) > remaining_budget
            and len(conversation_messages) > 2
        ):
            conversation_messages = conversation_messages[1:]

        # Build augmented message list for the LLM
        llm_messages = [SystemMessage(content=system_prompt)] + conversation_messages
        return {"llm_input_messages": llm_messages}

    return prepare_context
