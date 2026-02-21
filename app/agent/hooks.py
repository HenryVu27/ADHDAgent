"""Pre- and post-model hooks for the ReAct agent.

pre_model_hook: runs before every LLM call in the ReAct loop.
  - First iteration: run NeMo input rails, inject session context.
  - Subsequent iterations (tool call results): pass through.

post_model_hook: runs after the agent produces a final text response.
  - Run output guardrails on the response.
  - Replace with safe fallback if violation detected.
"""

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.agent.prompts import SAFE_OUTPUT_FALLBACK, build_system_prompt
from app.agent.store_protocol import SessionStoreBase
from app.agent.state import CoachingState
from app.config import settings
from app.guardrails.validator import GuardrailsValidator
from app.models.schemas import GuardrailsError

logger = logging.getLogger(__name__)


def create_hooks(
    guardrails: GuardrailsValidator,
    session_store: SessionStoreBase,
    event_bus=None,
):
    """Create pre_model_hook and post_model_hook closures."""

    async def pre_model_hook(state: CoachingState):
        """Run before each LLM call. Handles guardrails + context injection."""
        messages = state["messages"]
        session_id = state.get("session_id", "default")

        # Find the latest human message
        latest_human = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_human = msg
                break

        # Only run input guardrails on the first iteration (when last message is HumanMessage)
        # If last message is a tool result, we're mid-loop — skip guardrails
        if latest_human and isinstance(messages[-1], HumanMessage):
            start = time.time()
            # Pass recent conversation history so NeMo can understand short
            # replies like "Yes please" in context (prevents false off-topic blocks).
            session_state_for_rails = session_store.get(session_id)
            recent_history = session_state_for_rails.conversation_history[-6:]
            try:
                check = await guardrails.check_input(
                    latest_human.content,
                    context={"conversation_history": recent_history},
                )
                duration_ms = (time.time() - start) * 1000

                trace_step = {
                    "name": "input_guardrails",
                    "duration_ms": duration_ms,
                    "detail": {
                        "is_allowed": check.is_allowed,
                        "blocked_reason": check.blocked_reason,
                    },
                }

                if not check.is_allowed:
                    # NeMo's Colang checks each message in isolation (via
                    # $user_message). This causes false off_topic blocks in
                    # two scenarios:
                    # 1. Short continuations ("Yes please") in active conversations
                    # 2. On-topic messages with casual phrasing or greetings that
                    #    confuse the flash-lite classifier
                    _ON_TOPIC_KEYWORDS = {
                        "child", "kid", "son", "daughter", "adhd", "focus",
                        "homework", "school", "routine", "behavior", "behaviour",
                        "meltdown", "tantrum", "attention", "distract", "strategy",
                        "help", "parent", "morning", "bedtime", "transition",
                        "emotion", "frustrat", "calm", "overwhelm", "goal",
                    }
                    msg_lower = latest_human.content.strip().lower()
                    has_on_topic_keyword = any(kw in msg_lower for kw in _ON_TOPIC_KEYWORDS)

                    is_false_positive = (
                        check.blocked_reason == "off_topic"
                        and (
                            # Short continuation in active conversation
                            (len(msg_lower) < 40 and len(recent_history) > 0)
                            # Or message contains ADHD/parenting keywords
                            or has_on_topic_keyword
                        )
                    )
                    if is_false_positive:
                        logger.info(
                            "Overriding off_topic block (keywords=%s): %r",
                            has_on_topic_keyword, latest_human.content,
                        )
                    else:
                        logger.info("Input blocked: %s", check.blocked_reason)
                        if event_bus:
                            event_bus.emit("guardrails", "input_blocked", session_id, duration_ms=duration_ms,
                                           detail={"reason": check.blocked_reason})
                        # Return a Command that sets the block and goes to END
                        return Command(
                            goto="__end__",
                            update={
                                "input_blocked": True,
                                "block_response": check.override_response or "",
                                "trace_steps": [trace_step],
                                "messages": [AIMessage(content=check.override_response or "")],
                            },
                        )
                else:
                    if event_bus:
                        event_bus.emit("guardrails", "input_check_passed", session_id, duration_ms=duration_ms)

            except GuardrailsError as e:
                logger.error("Guardrails check failed: %s", e)
                # On guardrails failure, let the message through but log it
                trace_step = {
                    "name": "input_guardrails",
                    "duration_ms": (time.time() - start) * 1000,
                    "detail": {"error": str(e)},
                }

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

        system_prompt = build_system_prompt(
            profile=session_state.family_profile,
            active_strategies=session_state.active_strategies,
            goals=session_state.goals,
            outcomes=session_state.outcomes,
            session_summary=summary_text,
        )

        # Trim conversation to recent turns (keep system prompt + last N*2 messages)
        max_messages = settings.CONTEXT_WINDOW_TURNS * 2
        conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        if len(conversation_messages) > max_messages:
            conversation_messages = conversation_messages[-max_messages:]

        # Build augmented message list for the LLM
        llm_messages = [SystemMessage(content=system_prompt)] + conversation_messages
        result = {"llm_input_messages": llm_messages}

        # Model routing: classify complexity and set tier in state
        if settings.MODEL_ROUTING_ENABLED:
            from app.agent.model_router import classify_complexity
            tier = classify_complexity(state)
            result["model_tier"] = tier
            logger.info("Model tier classified: %s", tier)
            if event_bus:
                event_bus.emit("model_routing", "tier_classified", session_id, detail={"tier": tier})

        return result

    async def post_model_hook(state: CoachingState):
        """Run after the agent produces a final response. Checks output guardrails."""
        messages = state["messages"]

        # Find the last AI message (the agent's response)
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                last_ai = msg
                break

        if not last_ai:
            return {"messages": messages}

        start = time.time()
        try:
            check = await guardrails.check_output(last_ai.content)
            duration_ms = (time.time() - start) * 1000

            trace_step = {
                "name": "output_guardrails",
                "duration_ms": duration_ms,
                "detail": {
                    "is_valid": check.is_valid,
                    "violation_type": check.violation_type,
                },
            }

            if not check.is_valid:
                logger.info("Output guardrail triggered: %s", check.violation_type)
                if event_bus:
                    out_session_id = state.get("session_id", "")
                    event_bus.emit("guardrails", "output_violation", out_session_id, duration_ms=duration_ms,
                                   detail={"violation_type": check.violation_type})
                # Replace the response with safe fallback
                safe_msg = AIMessage(content=SAFE_OUTPUT_FALLBACK)
                new_messages = [m for m in messages if m is not last_ai] + [safe_msg]
                return {
                    "messages": new_messages,
                    "trace_steps": [trace_step],
                }

            if event_bus:
                out_session_id = state.get("session_id", "")
                event_bus.emit("guardrails", "output_check_passed", out_session_id, duration_ms=duration_ms)
            return {"trace_steps": [trace_step]}

        except GuardrailsError as e:
            logger.error("Output guardrails failed: %s", e)
            # On failure, let the response through
            return {"trace_steps": [{
                "name": "output_guardrails",
                "duration_ms": (time.time() - start) * 1000,
                "detail": {"error": str(e)},
            }]}

    return pre_model_hook, post_model_hook
