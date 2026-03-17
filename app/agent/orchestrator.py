"""AgentOrchestrator — thin wrapper: manages sessions and invokes the ReAct agent."""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import (
    AgentReasoningStep,
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

    _STATIC_FALLBACK = (
        "I want to make sure I give you the best help. "
        "Could you tell me a bit more about what you'd like to focus on?"
    )

    _TOOL_STATUS_MAP = {
        "search_knowledge_base": "Looking up strategies...",
        "get_family_profile": "Reading your profile...",
        "update_family_profile": "Updating your profile...",
        "track_outcome": "Recording outcome...",
        "manage_goals": "Managing goals...",
    }

    @staticmethod
    def _extract_token_text(chunk) -> str:
        """Extract streamable text from an AIMessageChunk, skipping thinking parts."""
        content = getattr(chunk, "content", "")
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return content if isinstance(content, str) else ""

    def __init__(self, agent, session_store: SessionStoreBase, memory_manager=None, analyzer=None, event_bus=None):
        self._agent = agent
        self._session_store = session_store
        self._memory = memory_manager
        self._analyzer = analyzer
        self._event_bus = event_bus
        self._pending_tasks: set[asyncio.Task] = set()

    async def get_session(self, session_id: str) -> SessionState:
        return await self._session_store.get(session_id)

    def get_session_store(self) -> SessionStoreBase:
        """Public accessor for the session store."""
        return self._session_store

    async def infer_phase(self, session_id: str) -> ConversationPhase:
        """Infer a phase label from session state."""
        return await self._infer_phase(session_id)

    async def seed_session(self, request: SeedSessionRequest) -> None:
        await self._session_store.seed_session(request)

    async def process_stream(
        self, message: str, session_id: str
    ) -> AsyncGenerator[tuple[str, dict], None]:
        """Stream the ReAct agent response as SSE (event_type, data) tuples.

        Event sequence:
            status  -- input gate passed; each tool invocation
            token   -- one text chunk from the final LLM response
            replace -- output gate replaced the streamed response
            done    -- stream complete, session committed
            error   -- timeout or unhandled exception

        Design notes (see docs/superpowers/specs/2026-03-13-streaming-design.md):
        - Tokens filtered by: (1) checkpoint namespace starting with pro/flash_react_agent,
          (2) no tool_call_chunks, (3) skip thinking content parts (type="thinking").
        - Route captured from first on_chain_start with langgraph_node == pro/flash_react_agent.
        - Output gate replacement detected from trace_steps, not string comparison.
        - On error/timeout: persists fallback text, yields error event, returns.
        - Memory and analyzer run as background tasks after done is emitted.
        """
        turn = await self._session_store.increment_turn(session_id)
        logger.info(
            "[agent:stream] === START === session=%s, turn=%d, message=%.80s",
            session_id, turn, message,
        )

        if self._event_bus:
            await self._event_bus.emit(
                "agent", "turn_start", session_id, turn,
                detail={"message_preview": message[:80]},
            )

        from app.config import settings

        # Build history (same logic as the former process())
        stored_messages = await self._session_store.get_messages(session_id)
        latest_summary = await self._session_store.get_latest_summary(session_id)
        summary_through_turn = latest_summary.covers_through_turn if latest_summary else 0

        history_messages = []
        unsummarized_chars = 0
        for entry in stored_messages:
            if entry.get("blocked"):
                continue
            msg_turn = entry.get("turn", 0)
            if summary_through_turn > 0 and msg_turn <= summary_through_turn:
                continue
            if entry["role"] == "user":
                history_messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                history_messages.append(AIMessage(content=entry["content"]))
            unsummarized_chars += len(entry.get("content", ""))

        # Trigger rolling summary if unsummarized history fills context budget
        context_utilization = unsummarized_chars / settings.CONTEXT_MAX_CHARS
        force_summary = context_utilization >= 0.8

        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }

        start = time.time()
        route = "pro"
        accumulated_text = ""
        result_state: dict = {}

        try:
            async with asyncio.timeout(settings.CHAT_TIMEOUT_S):
                async for event in self._agent.astream_events(
                    {
                        "messages": history_messages + [HumanMessage(content=message)],
                        "session_id": session_id,
                    },
                    config=config,
                    version="v2",
                ):
                    etype = event["event"]
                    meta = event.get("metadata", {})
                    node = meta.get("langgraph_node", "")
                    ns = meta.get("langgraph_checkpoint_ns", "")

                    # Detect which react agent ran (sets route for agent_used label)
                    if etype == "on_chain_start" and node in ("pro_react_agent", "flash_react_agent"):
                        route = "flash" if node == "flash_react_agent" else "pro"
                        yield ("status", {"text": "Thinking..."})

                    # Named status per tool call
                    elif etype == "on_tool_start" and node == "tools":
                        tool_name = event.get("name", "")
                        yield ("status", {"text": self._TOOL_STATUS_MAP.get(tool_name, "Working on it...")})

                    # Stream final-response tokens only
                    elif (
                        etype == "on_chat_model_stream"
                        and node == "agent"
                        and (ns.startswith("pro_react_agent") or ns.startswith("flash_react_agent"))
                    ):
                        chunk = event["data"]["chunk"]
                        if getattr(chunk, "tool_call_chunks", None):
                            continue  # tool-invocation step -- skip
                        text = self._extract_token_text(chunk)
                        if text:
                            accumulated_text += text
                            yield ("token", {"text": text})

                    # Capture final graph state from top-level on_chain_end
                    elif etype == "on_chain_end" and not node and not ns:
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            result_state = output

        except asyncio.TimeoutError:
            logger.warning("[agent:stream] Timeout session=%s", session_id)
            await self._session_store.add_message(session_id, "user", message, turn)
            await self._session_store.add_message(session_id, "assistant", self._STATIC_FALLBACK, turn)
            await self._session_store.commit()
            yield ("error", {"message": "Request timed out. Please try again."})
            return

        except Exception as exc:
            logger.warning("[agent:stream] Exception session=%s: %s", session_id, exc)
            await self._session_store.add_message(session_id, "user", message, turn)
            await self._session_store.add_message(session_id, "assistant", self._STATIC_FALLBACK, turn)
            await self._session_store.commit()
            yield ("error", {"message": "Something went wrong. Please try again."})
            return

        total_ms = (time.time() - start) * 1000

        # --- Input-blocked path ---
        if result_state.get("input_blocked"):
            response_text = result_state.get("block_response", "")
            blocked_reason = ""
            for step in result_state.get("trace_steps", []):
                if step.get("name") == "input_gate":
                    blocked_reason = step.get("detail", {}).get("blocked_reason", "")

            await self._session_store.add_message(
                session_id, "user", message, turn,
                blocked=True, blocked_reason=blocked_reason,
            )
            await self._session_store.add_message(
                session_id, "assistant", response_text, turn,
                blocked=True, blocked_reason=blocked_reason,
            )
            enriched = EnrichedTrace(
                session_id=session_id,
                turn=turn,
                timestamp=datetime.now(timezone.utc).isoformat(),
                pipeline_steps=[PipelineStep(**s) for s in result_state.get("trace_steps", [])],
                total_duration_ms=total_ms,
                input_blocked=True,
                blocked_reason=blocked_reason,
                agent_used="input_gate",
            )
            await self._session_store.save_trace(session_id, enriched)
            await self._session_store.commit()

            if self._event_bus:
                await self._event_bus.emit(
                    "agent", "turn_blocked", session_id, turn, total_ms,
                    detail={"reason": blocked_reason},
                )

            trace = self._build_trace(result_state, total_ms)
            phase = await self._infer_phase(session_id)
            yield ("done", {
                "session_id": session_id,
                "agent_used": "input_gate",
                "phase": phase.value,
                "pipeline_trace": trace.model_dump(),
                "response": response_text,
            })
            return

        # --- Normal path: extract final response from result state ---
        all_messages = result_state.get("messages", [])
        last_human_idx = -1
        for i in range(len(all_messages) - 1, -1, -1):
            if isinstance(all_messages[i], HumanMessage):
                last_human_idx = i
                break
        new_messages = all_messages[last_human_idx + 1:] if last_human_idx >= 0 else []

        final_response = ""
        tool_calls_made = []
        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls_made.extend(msg.tool_calls)
                elif msg.content:
                    final_response = _extract_text(msg.content)

        if not final_response.strip():
            final_response = self._STATIC_FALLBACK

        # Emit replace event if output gate replaced the streamed content
        output_gate_replaced = any(
            s.get("name") == "output_gate" and not s.get("detail", {}).get("is_valid", True)
            for s in result_state.get("trace_steps", [])
        )
        if output_gate_replaced:
            yield ("replace", {"text": final_response})

        # Post-processing -- these block done to keep trace accurate
        tool_summary = self._build_tool_calls_summary(tool_calls_made)
        await self._session_store.add_message(session_id, "user", message, turn)
        await self._session_store.add_message(
            session_id, "assistant", final_response, turn,
            tool_calls_summary=tool_summary,
        )

        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                tc_name = tc_query = ""
                for tc in tool_calls_made:
                    if tc.get("id") == msg.tool_call_id:
                        tc_name = tc.get("name", "")
                        tc_query = str(tc.get("args", {}).get("query", ""))
                        break
                if tc_name == "search_knowledge_base":
                    result_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    await self._session_store.save_tool_result(
                        session_id, tc_name, tc_query, result_text, turn,
                    )

        enriched = self._build_enriched_trace(
            session_id, turn, result_state, new_messages, tool_calls_made, total_ms,
        )
        await self._session_store.save_trace(session_id, enriched)
        await self._session_store.commit()

        # Background tasks -- non-blocking, do not delay done
        if self._memory:
            self._track_task(
                self._memory.post_turn_tasks(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=final_response,
                    tool_calls=list(tool_calls_made),
                    force_summary=force_summary,
                ),
                "memory",
            )

        if self._analyzer:
            self._track_task(
                self._analyzer.analyze_turn(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=final_response,
                    enriched_trace=enriched,
                ),
                "analyzer",
            )

        if self._event_bus:
            await self._event_bus.emit(
                "agent", "turn_end", session_id, turn, total_ms,
                detail={"tools": len(tool_calls_made), "model_tier": enriched.model_tier},
            )

        agent_label = "flash_react_agent" if route == "flash" else "react_agent"
        trace = self._build_trace(result_state, total_ms, tool_calls_made)
        phase = await self._infer_phase(session_id)

        logger.info(
            "[agent:stream] === END === session=%s, tools=%d, duration=%.0fms",
            session_id, len(tool_calls_made), total_ms,
        )

        yield ("done", {
            "session_id": session_id,
            "agent_used": agent_label,
            "phase": phase.value,
            "pipeline_trace": trace.model_dump(),
            "response": None,
        })

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

    async def _infer_phase(self, session_id: str) -> ConversationPhase:
        """Infer a phase label from session state for API compatibility."""
        state = await self._session_store.get(session_id)
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
