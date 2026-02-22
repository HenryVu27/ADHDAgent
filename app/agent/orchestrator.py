"""AgentOrchestrator — thin wrapper: manages sessions and invokes the ReAct agent."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import (
    AgentReasoningStep,
    ChatResponse,
    ConversationPhase,
    EnrichedTrace,
    PipelineStep,
    PipelineTrace,
    SeedSessionRequest,
    SessionState,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Extract text from Gemini content (may be str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


class AgentOrchestrator:
    """Manages sessions and runs the compiled ReAct agent."""

    def __init__(self, agent, session_store: SessionStoreBase, memory_manager=None, analyzer=None, event_bus=None):
        self._agent = agent
        self._session_store = session_store
        self._memory = memory_manager
        self._analyzer = analyzer
        self._event_bus = event_bus
        self._pending_tasks: set[asyncio.Task] = set()

    def get_session(self, session_id: str) -> SessionState:
        return self._session_store.get(session_id)

    def get_session_store(self) -> SessionStoreBase:
        """Public accessor for the session store."""
        return self._session_store

    def infer_phase(self, session_id: str) -> ConversationPhase:
        """Infer a phase label from session state."""
        return self._infer_phase(session_id)

    def seed_session(self, request: SeedSessionRequest) -> None:
        self._session_store.seed_session(request)

    async def process(self, message: str, session_id: str) -> ChatResponse:
        """Run the ReAct agent for a single parent message."""
        turn = self._session_store.increment_turn(session_id)
        logger.info(
            "[agent] === START === session=%s, turn=%d, message=%.80s",
            session_id, turn, message,
        )

        if self._event_bus:
            self._event_bus.emit("agent", "turn_start", session_id, turn, detail={"message_preview": message[:80]})

        from app.config import settings

        # Build full message list from conversation history so the agent
        # has multi-turn context (previous turns were stored but never passed back).
        stored_messages = self._session_store.get_messages(session_id)

        # Skip messages already captured by the rolling summary
        latest_summary = self._session_store.get_latest_summary(session_id)
        summary_through_turn = latest_summary.covers_through_turn if latest_summary else 0

        history_messages = []
        for entry in stored_messages:
            if entry.get("blocked"):
                continue  # Skip blocked turns
            # Skip messages from turns already covered by the rolling summary
            msg_turn = entry.get("turn", 0)
            if summary_through_turn > 0 and msg_turn <= summary_through_turn:
                continue
            if entry["role"] == "user":
                history_messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                content = entry["content"]
                tcs = entry.get("tool_calls_summary", "")
                if tcs:
                    content = f"[Tools used: {tcs}]\n{content}"
                history_messages.append(AIMessage(content=content))

        start = time.time()
        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 1,
        }

        try:
            result = await self._agent.ainvoke(
                {
                    "messages": history_messages + [HumanMessage(content=message)],
                    "session_id": session_id,
                },
                config=config,
            )
        except Exception as e:
            total_ms = (time.time() - start) * 1000
            logger.warning("[agent] Agent invocation failed (%s) — using static fallback", type(e).__name__)
            response_text = (
                "I want to make sure I give you the best help. "
                "Could you tell me a bit more about what you'd like to focus on?"
            )
            self._session_store.add_message(session_id, "user", message, turn)
            self._session_store.add_message(session_id, "assistant", response_text, turn)
            trace = PipelineTrace(
                steps=[PipelineStep(name="react_agent", duration_ms=total_ms, detail={"error": str(e)})],
                total_duration_ms=total_ms,
                agent_used="react_agent_fallback",
            )
            return ChatResponse(
                response=response_text,
                agent_used="react_agent_fallback",
                phase=self._infer_phase(session_id),
                pipeline_trace=trace,
                session_id=session_id,
            )

        total_ms = (time.time() - start) * 1000

        # Check if input was blocked
        if result.get("input_blocked"):
            response_text = result.get("block_response", "")
            # Record blocked turn
            blocked_reason = ""
            for step in result.get("trace_steps", []):
                if step.get("name") == "input_gate":
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

            # Persist enriched trace for blocked turns
            enriched = EnrichedTrace(
                session_id=session_id,
                turn=turn,
                timestamp=datetime.now(timezone.utc).isoformat(),
                pipeline_steps=[PipelineStep(**s) for s in result.get("trace_steps", [])],
                total_duration_ms=total_ms,
                input_blocked=True,
                blocked_reason=blocked_reason,
                agent_used="input_gate",
            )
            self._session_store.save_trace(session_id, enriched)

            if self._event_bus:
                self._event_bus.emit("agent", "turn_blocked", session_id, turn, total_ms, detail={"reason": blocked_reason})

            logger.info("[agent] === BLOCKED === session=%s, reason=input_gate", session_id)
            return ChatResponse(
                response=response_text,
                agent_used="input_gate",
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
                elif msg.content:
                    # Only non-tool-calling AI messages count as final response
                    response_text = _extract_text(msg.content)

        # Fallback if agent produced no final text response
        is_empty = not response_text.strip() if isinstance(response_text, str) else not response_text
        if is_empty:
            logger.warning("[agent] Empty response from agent — using static fallback")
            response_text = (
                "I want to make sure I give you the best help. "
                "Could you tell me a bit more about what you'd like to focus on?"
            )

        # Record turn in conversation history
        tool_summary = self._build_tool_calls_summary(tool_calls_made)
        self._session_store.add_message(session_id, "user", message, turn)
        self._session_store.add_message(session_id, "assistant", response_text, turn, tool_calls_summary=tool_summary)

        # Persist tool results for cross-turn evidence
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                tc_name = ""
                tc_query = ""
                for tc in tool_calls_made:
                    if tc.get("id") == msg.tool_call_id:
                        tc_name = tc.get("name", "")
                        tc_query = str(tc.get("args", {}).get("query", ""))
                        break
                if tc_name == "search_knowledge_base":
                    result_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    self._session_store.save_tool_result(session_id, tc_name, tc_query, result_text, turn)

        # Build and persist enriched trace
        enriched = self._build_enriched_trace(
            session_id, turn, result, new_messages, tool_calls_made, total_ms,
        )
        self._session_store.save_trace(session_id, enriched)

        # Fire background memory tasks (non-blocking)
        if self._memory:
            self._track_task(
                self._memory.post_turn_tasks(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=response_text,
                    tool_calls=[tc for tc in tool_calls_made],
                ),
                "memory",
            )

        # Fire background analyzer (non-blocking)
        if self._analyzer:
            self._track_task(
                self._analyzer.analyze_turn(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=response_text,
                    enriched_trace=enriched,
                ),
                "analyzer",
            )

        trace = self._build_trace(result, total_ms, tool_calls_made)

        if self._event_bus:
            self._event_bus.emit(
                "agent", "turn_end", session_id, turn, total_ms,
                detail={"tools": len(tool_calls_made), "model_tier": enriched.model_tier},
            )

        logger.info(
            "[agent] === END === session=%s, tools=%d, duration=%.0fms",
            session_id, len(tool_calls_made), total_ms,
        )

        route = result.get("route", "pro")
        agent_label = "flash_react_agent" if route == "flash" else "react_agent"

        return ChatResponse(
            response=response_text,
            agent_used=agent_label,
            phase=self._infer_phase(session_id),
            pipeline_trace=trace,
            session_id=session_id,
        )

    @staticmethod
    async def _safe_background(coro, label: str = "background") -> None:
        """Run a coroutine with exception logging instead of silent swallowing."""
        try:
            await coro
        except Exception as e:
            logger.error("Background task '%s' failed: %s", label, e)

    def _track_task(self, coro, label: str) -> asyncio.Task:
        """Create a tracked background task."""
        task = asyncio.create_task(self._safe_background(coro, label))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Wait for pending background tasks to complete."""
        if self._pending_tasks:
            logger.info("Waiting for %d background tasks...", len(self._pending_tasks))
            done, pending = await asyncio.wait(self._pending_tasks, timeout=timeout)
            if pending:
                logger.warning("Cancelling %d background tasks after timeout", len(pending))
                for task in pending:
                    task.cancel()

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
    def _build_tool_calls_summary(tool_calls: list[dict]) -> str:
        """Build a compact one-line summary of tool calls for message metadata."""
        if not tool_calls:
            return ""
        parts = []
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            if name == "search_knowledge_base":
                query = args.get("query", "")
                parts.append(f'search_knowledge_base(query="{query}")')
            elif name == "update_family_profile":
                arg_keys = [k for k in ("child_name", "child_age", "diagnosis_status") if args.get(k)]
                parts.append(f"update_family_profile({', '.join(arg_keys)})" if arg_keys else "update_family_profile()")
            elif name == "track_outcome":
                strategy = args.get("strategy_name", "")
                signal = args.get("outcome", "")
                parts.append(f"track_outcome({strategy}: {signal})")
            elif name == "manage_goals":
                action = args.get("action", "")
                desc = args.get("description", "")
                parts.append(f"manage_goals({action}: {desc})")
            else:
                parts.append(name)
        return "; ".join(parts)

    @staticmethod
    def _build_enriched_trace(
        session_id: str,
        turn: int,
        result: dict,
        new_messages: list,
        tool_calls_made: list,
        total_ms: float,
    ) -> EnrichedTrace:
        """Build an EnrichedTrace with reasoning steps and tool results."""
        pipeline_steps = [
            PipelineStep(**s) for s in result.get("trace_steps", [])
        ]

        # Build reasoning chain from new_messages
        reasoning_steps: list[AgentReasoningStep] = []
        tool_records: list[ToolCallRecord] = []
        step_index = 0

        # Build a map of tool_call_id -> ToolMessage content for matching results
        tool_results_map: dict[str, str] = {}
        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                tool_results_map[msg.tool_call_id] = content[:2000]

        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    # AIMessage with tool calls = reasoning + tool invocation
                    thought = msg.content if isinstance(msg.content, str) and msg.content else ""
                    for tc in msg.tool_calls:
                        tc_id = tc.get("id", "")
                        result_text = tool_results_map.get(tc_id, "")
                        record = ToolCallRecord(
                            name=tc.get("name", "unknown"),
                            args=tc.get("args", {}),
                            result=result_text,
                        )
                        tool_records.append(record)
                        reasoning_steps.append(AgentReasoningStep(
                            step_index=step_index,
                            thought=thought,
                            tool_call=record,
                        ))
                        thought = ""  # Only first tool call gets the thought
                        step_index += 1
                elif msg.content:
                    # Final response (no tool calls)
                    reasoning_steps.append(AgentReasoningStep(
                        step_index=step_index,
                        thought=msg.content if isinstance(msg.content, str) else str(msg.content),
                        is_final=True,
                    ))
                    step_index += 1

        route = result.get("route", "pro")
        model_tier = "fast" if route == "flash" else "standard"
        agent_label = "flash_react_agent" if route == "flash" else "react_agent"

        return EnrichedTrace(
            session_id=session_id,
            turn=turn,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_steps=pipeline_steps,
            total_duration_ms=total_ms,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_records,
            model_tier=model_tier,
            agent_used=agent_label,
        )

    @staticmethod
    def _build_trace(
        result: dict,
        total_ms: float,
        tool_calls: list | None = None,
    ) -> PipelineTrace:
        """Build a PipelineTrace from the agent result."""
        route = result.get("route", "pro")
        agent_label = "flash_react_agent" if route == "flash" else "react_agent"

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
            name=agent_label,
            duration_ms=total_ms,
        ))

        return PipelineTrace(
            steps=steps,
            total_duration_ms=total_ms,
            agent_used=agent_label,
        )
