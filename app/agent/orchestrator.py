"""AgentOrchestrator — thin wrapper: manages sessions and invokes the ReAct agent."""

import asyncio
import logging
import time

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import (
    ChatResponse,
    ConversationPhase,
    PipelineStep,
    PipelineTrace,
    SeedSessionRequest,
    SessionState,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Manages sessions and runs the compiled ReAct agent."""

    def __init__(self, agent, session_store: SessionStoreBase, memory_manager=None):
        self._agent = agent
        self._session_store = session_store
        self._memory = memory_manager

    def get_session(self, session_id: str) -> SessionState:
        return self._session_store.get(session_id)

    def seed_session(self, request: SeedSessionRequest) -> None:
        self._session_store.seed_session(request)

    async def process(self, message: str, session_id: str) -> ChatResponse:
        """Run the ReAct agent for a single parent message."""
        turn = self._session_store.increment_turn(session_id)
        logger.info(
            "[agent] === START === session=%s, turn=%d, message=%.80s",
            session_id, turn, message,
        )

        from app.config import settings

        # Build full message list from conversation history so the agent
        # has multi-turn context (previous turns were stored but never passed back).
        stored_messages = self._session_store.get_messages(session_id)
        history_messages = []
        for entry in stored_messages:
            if entry.get("blocked"):
                continue  # Skip blocked turns
            if entry["role"] == "user":
                history_messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                history_messages.append(AIMessage(content=entry["content"]))

        start = time.time()
        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 1,
        }

        result = await self._agent.ainvoke(
            {
                "messages": history_messages + [HumanMessage(content=message)],
                "session_id": session_id,
            },
            config=config,
        )

        total_ms = (time.time() - start) * 1000

        # Check if input was blocked
        if result.get("input_blocked"):
            response_text = result.get("block_response", "")
            # Record blocked turn
            blocked_reason = ""
            for step in result.get("trace_steps", []):
                if step.get("name") == "input_guardrails":
                    blocked_reason = step.get("detail", {}).get("blocked_reason", "")
            self._session_store.add_message(
                session_id, "user", message, turn,
                blocked=True, blocked_reason=blocked_reason,
            )
            self._session_store.add_message(
                session_id, "assistant", response_text, turn,
                blocked=True, blocked_reason=blocked_reason,
            )
            trace = self._build_trace(result, total_ms)
            logger.info("[agent] === BLOCKED === session=%s, reason=input_guardrails", session_id)
            return ChatResponse(
                response=response_text,
                agent_used="guardrails",
                phase=self._infer_phase(session_id),
                pipeline_trace=trace,
                session_id=session_id,
            )

        # Extract the final AI response from NEW messages only (skip history).
        # Anchor on the last HumanMessage (our current message) rather than
        # counting, since LangGraph may modify the message list internally.
        all_messages = result.get("messages", [])
        last_human_idx = -1
        for i in range(len(all_messages) - 1, -1, -1):
            if isinstance(all_messages[i], HumanMessage):
                last_human_idx = i
                break
        new_messages = all_messages[last_human_idx + 1:] if last_human_idx >= 0 else []

        response_text = ""
        tool_calls_made = []
        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls_made.extend(msg.tool_calls)
                # Use `if` not `elif` — Gemini can return content AND tool_calls
                # in the same message. We want the last message's content.
                if msg.content:
                    response_text = msg.content

        # Fallback if the agent returned an empty response (can happen when
        # Gemini flash-lite calls a tool but generates empty text afterward,
        # typically for short/ambiguous user messages like "Yes please").
        if not response_text.strip():
            logger.warning("[agent] Empty response from agent — using fallback")
            response_text = (
                "I want to make sure I give you the best help. "
                "Could you tell me a bit more about what you'd like to focus on?"
            )

        # Record turn in conversation history
        self._session_store.add_message(session_id, "user", message, turn)
        self._session_store.add_message(session_id, "assistant", response_text, turn)

        # Fire background memory tasks (non-blocking)
        if self._memory:
            asyncio.create_task(self._memory.post_turn_tasks(
                session_id=session_id,
                turn=turn,
                user_message=message,
                assistant_response=response_text,
                tool_calls=[tc for tc in tool_calls_made],
            ))

        trace = self._build_trace(result, total_ms, tool_calls_made)

        logger.info(
            "[agent] === END === session=%s, tools=%d, duration=%.0fms",
            session_id, len(tool_calls_made), total_ms,
        )

        return ChatResponse(
            response=response_text,
            agent_used="react_agent",
            phase=self._infer_phase(session_id),
            pipeline_trace=trace,
            session_id=session_id,
        )

    def _infer_phase(self, session_id: str) -> ConversationPhase:
        """Infer a phase label from session state for API compatibility."""
        state = self._session_store.get(session_id)
        profile = state.family_profile

        # If we have outcomes, we're in progress tracking
        if state.outcomes:
            return ConversationPhase.progress

        # If we have active strategies, we're in strategy mode
        if state.active_strategies or state.recommended_strategies:
            return ConversationPhase.strategy

        # If profile has meaningful data, we're past intake
        has_profile = bool(
            profile.child_age
            or profile.challenge_areas
            or profile.hardest_situations
        )
        if has_profile:
            return ConversationPhase.strategy

        return ConversationPhase.intake

    @staticmethod
    def _build_trace(
        result: dict,
        total_ms: float,
        tool_calls: list | None = None,
    ) -> PipelineTrace:
        """Build a PipelineTrace from the agent result."""
        steps = []

        # Add trace steps recorded by hooks
        for step_dict in result.get("trace_steps", []):
            steps.append(PipelineStep(**step_dict))

        # Add tool call steps
        for tc in (tool_calls or []):
            steps.append(PipelineStep(
                name=f"tool:{tc.get('name', 'unknown')}",
                detail={"args": tc.get("args", {})},
            ))

        # Add the agent step
        steps.append(PipelineStep(
            name="react_agent",
            duration_ms=total_ms,
        ))

        return PipelineTrace(
            steps=steps,
            total_duration_ms=total_ms,
            agent_used="react_agent",
        )
