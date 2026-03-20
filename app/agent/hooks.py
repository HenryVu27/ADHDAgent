"""Context assembly for the ReAct agent.

prepare_context: runs as the pre_model_hook in the ReAct agent.
  - Builds system prompt from session state (family profile, goals, summary, episodes).
  - Trims conversation history to recent turns.
  - Caches the system prompt within a single turn's ReAct loop to avoid redundant rebuilds.
"""

import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.prompts import build_conversation_state, build_system_prompt
from app.agent.store_protocol import SessionStoreBase
from app.agent.state import CoachingState
from app.config import settings

logger = logging.getLogger(__name__)

# Module-level prompt cache keyed by (session_id, turn_count)
_prompt_cache: dict[tuple[str, int], str] = {}

# Tools that mutate session state and require cache invalidation
_MUTATING_TOOLS = {"update_family_profile", "manage_goals", "track_outcome"}


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


def _has_mutating_tool(messages) -> bool:
    """Check if a state-mutating tool ran since the last HumanMessage."""
    for msg in reversed(messages):
        if hasattr(msg, "name") and getattr(msg, "name", None) in _MUTATING_TOOLS:
            return True
        if isinstance(msg, HumanMessage):
            break  # Stop at the user message
    return False


def create_prepare_context(
    session_store: SessionStoreBase,
    event_bus=None,
):
    """Create the prepare_context closure used as pre_model_hook."""

    async def prepare_context(state: CoachingState):
        """Build system prompt and trim conversation for the LLM."""
        messages = state["messages"]
        session_id = state.get("session_id", "default")

        # Get session state (cheap read) for turn count and cache key
        session_state = await session_store.get(session_id)
        turn_count = session_state.turn_count
        cache_key = (session_id, turn_count)

        # Check if a state-mutating tool ran in this iteration
        has_mutation = _has_mutating_tool(messages)

        # Always build the per-turn state block (attached to HumanMessage, not system prompt).
        phase = session_state.phase.value if hasattr(session_state.phase, "value") else str(session_state.phase)
        recent_tool_names: list[str] | None = None
        try:
            traces = await session_store.get_traces(session_id)
            if traces:
                last_trace = traces[-1]
                if last_trace.tool_calls:
                    recent_tool_names = [tc.name for tc in last_trace.tool_calls]
        except Exception:
            pass
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
            current_datetime=datetime.now(),
        )

        if cache_key in _prompt_cache and not has_mutation:
            system_prompt = _prompt_cache[cache_key]
        else:
            # Full prompt build
            # Load rolling summary if available
            latest_summary = await session_store.get_latest_summary(session_id)
            summary_text = latest_summary.summary if latest_summary else ""

            # Load recent episodes into summary context
            recent_episodes = await session_store.get_recent_episodes(session_id, limit=5)
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
            recent_tool_results = await session_store.get_recent_tool_results(session_id, limit=3)

            system_prompt = build_system_prompt(
                profile=session_state.family_profile,
                active_strategies=session_state.active_strategies,
                goals=session_state.goals,
                outcomes=session_state.outcomes,
                session_summary=summary_text,
            )

            if recent_tool_results:
                evidence_lines = []
                for tr in recent_tool_results:
                    evidence_lines.append(f"[Turn {tr.turn}, query: \"{tr.query}\"]:\n{tr.result_text[:2000]}")
                evidence_block = "\n\n<prior-search-evidence>\n" + "\n---\n".join(evidence_lines) + "\n</prior-search-evidence>"
                system_prompt += evidence_block

            # Cross-session context for returning users (first turn only)
            user_id = state.get("user_id")
            if user_id is not None and turn_count <= 1:
                user_summary = await session_store.get_user_summary(user_id)
                user_episodes = await session_store.get_user_episodes(user_id, limit=5)
                user_outcomes = await session_store.get_user_outcomes(user_id, limit=5)

                prior_parts = []
                if user_summary:
                    prior_parts.append(f"Journey so far:\n{user_summary.summary}")
                if user_outcomes:
                    out_lines = [f"- {o.strategy_name}: {o.signal} — {o.detail}" for o in user_outcomes]
                    prior_parts.append("Recent outcomes across sessions:\n" + "\n".join(out_lines))
                if user_episodes:
                    ep_lines = [f"- [{ep.event_type}] {ep.summary}" for ep in user_episodes]
                    prior_parts.append("Recent key moments:\n" + "\n".join(ep_lines))

                if prior_parts:
                    prior_block = "\n\n<prior-sessions>\n" + "\n\n".join(prior_parts) + "\n</prior-sessions>"
                    system_prompt += prior_block

            # Cache the prompt
            _prompt_cache[cache_key] = system_prompt
            # Evict old entries (keep max 10)
            if len(_prompt_cache) > 10:
                oldest = next(iter(_prompt_cache))
                del _prompt_cache[oldest]

            # Emit memory usage event for utility tracking
            if event_bus:
                injected_fields = []
                profile = session_state.family_profile
                for field in ("child_name", "child_age", "diagnosis_status", "adhd_subtype",
                              "good_day_description"):
                    if getattr(profile, field, None):
                        injected_fields.append(field)
                for field in ("challenge_areas", "attempted_strategies", "hardest_situations"):
                    if getattr(profile, field, []):
                        injected_fields.append(field)
                if session_state.goals:
                    injected_fields.append("goals")
                if session_state.outcomes:
                    injected_fields.append("outcomes")
                try:
                    await event_bus.emit(
                        "memory_usage", "profile_injected", session_id, turn_count,
                        detail={"fields": injected_fields, "episode_count": len(recent_episodes)},
                    )
                except Exception:
                    pass  # Never block context assembly for observability

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

        # Build augmented message list for the LLM.
        # conversation_state is appended to the last HumanMessage (not the system prompt)
        # so the system message stays stable for KV-cache reuse.
        if conversation_messages:
            last_msg = conversation_messages[-1]
            if isinstance(last_msg, HumanMessage):
                state_suffix = "\n\n" + state_block
                if isinstance(last_msg.content, str):
                    new_content = last_msg.content + state_suffix
                elif isinstance(last_msg.content, list):
                    # Multipart content: append state block to first text part
                    new_content = list(last_msg.content)  # shallow copy
                    for i, part in enumerate(new_content):
                        if isinstance(part, dict) and part.get("type") == "text":
                            new_content[i] = {**part, "text": part["text"] + state_suffix}
                            break
                    else:
                        # No text part found -- prepend one
                        new_content.insert(0, {"type": "text", "text": state_suffix})
                else:
                    new_content = str(last_msg.content) + state_suffix
                conversation_messages = conversation_messages[:-1] + [
                    HumanMessage(content=new_content)
                ]

        llm_messages = [SystemMessage(content=system_prompt)] + conversation_messages
        return {"llm_input_messages": llm_messages}

    return prepare_context
