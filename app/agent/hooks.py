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
    graphiti_client=None,
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
            # Load memory context from Graphiti (or empty if unavailable)
            memory_context = ""
            if graphiti_client is not None:
                user_id = state.get("user_id")
                group_ids = [str(user_id)] if user_id is not None else None
                try:
                    last_user_text = ""
                    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
                    if user_msgs:
                        content = user_msgs[-1].content
                        last_user_text = content if isinstance(content, str) else str(content)

                    if last_user_text.strip():
                        import time as _time
                        t0 = _time.monotonic()
                        edges = await graphiti_client.search(
                            last_user_text,
                            group_ids=group_ids,
                            num_results=settings.GRAPHITI_CONTEXT_RESULTS,
                        )
                        duration_ms = (_time.monotonic() - t0) * 1000

                        valid_facts = []
                        for edge in edges:
                            fact = getattr(edge, "fact", "")
                            if fact and getattr(edge, "invalid_at", None) is None:
                                valid_at = getattr(edge, "valid_at", None)
                                date_note = f" (since {valid_at.strftime('%b %Y')})" if valid_at else ""
                                valid_facts.append(f"- {fact}{date_note}")

                        if valid_facts:
                            memory_context = "\n".join(valid_facts)

                        if event_bus:
                            await event_bus.emit(
                                "memory", "graphiti_search_completed", session_id, turn_count,
                                duration_ms=duration_ms,
                                detail={"query_len": len(last_user_text), "result_count": len(edges)},
                            )
                except Exception as e:
                    logger.warning("Graphiti context search failed: %s", e)

            summary_text = memory_context or "This is the beginning of the conversation."

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

            concerns = state.get("concerns", [])
            if concerns and state.get("is_multi_concern"):
                concern_lines = []
                for i, c in enumerate(concerns, 1):
                    concern_lines.append(f"{i}. {c.get('description', '')} (intent: {c.get('intent', '')})")
                concern_block = (
                    "\n\n<parent-concerns>\n"
                    "The parent's message contains multiple concerns. Address ALL of them:\n"
                    + "\n".join(concern_lines)
                    + "\n\nStructure your response with a clear section for each concern. "
                    "Do not skip any concern.\n</parent-concerns>"
                )
                system_prompt += concern_block

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
                        detail={"fields": injected_fields, "has_memory_context": bool(memory_context)},
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
