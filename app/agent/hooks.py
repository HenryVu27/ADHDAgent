"""Context assembly for the ReAct agent.

prepare_context: runs as the pre_model_hook in the ReAct agent.
  - Builds system prompt from session state (family profile, goals, summary, episodes).
  - Trims conversation history to recent turns.
  - Optionally classifies message complexity for model routing.
"""

import logging

from langchain_core.messages import SystemMessage

from app.agent.prompts import build_system_prompt
from app.agent.store_protocol import SessionStoreBase
from app.agent.state import CoachingState
from app.config import settings

logger = logging.getLogger(__name__)


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

    return prepare_context
