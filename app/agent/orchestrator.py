"""AgentOrchestrator — thin wrapper: manages sessions and invokes the ReAct agent."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.prompts import SAFE_OUTPUT_FALLBACK
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

    def __init__(self, agent, session_store: SessionStoreBase, memory_manager=None, analyzer=None, event_bus=None, gemini_client=None, output_gate=None):
        self._agent = agent
        self._session_store = session_store
        self._memory = memory_manager
        self._analyzer = analyzer
        self._event_bus = event_bus
        self._gemini_client = gemini_client
        self._output_gate = output_gate

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

        try:
            result = await self._agent.ainvoke(
                {
                    "messages": history_messages + [HumanMessage(content=message)],
                    "session_id": session_id,
                },
                config=config,
            )
        except Exception as e:
            # Handle recursion limit (model loops calling tools without final response)
            total_ms = (time.time() - start) * 1000
            logger.warning("[agent] Agent invocation failed (%s) — using search+synthesis fallback", type(e).__name__)
            response_text = await self._search_and_synthesize(message, session_id)
            if not response_text:
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
        tool_results = []  # Collect tool results for synthesis fallback
        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls_made.extend(msg.tool_calls)
                elif msg.content:
                    # Only non-tool-calling AI messages count as final response
                    response_text = _extract_text(msg.content)
            elif isinstance(msg, ToolMessage) and msg.content:
                tool_results.append(msg.content)

        # Gemini flash-lite often produces empty final responses after tool
        # calls — it calls the tool, gets results, but fails to synthesize
        # them.  When this happens, use a direct LLM call to produce a
        # response grounded in the tool results or a fresh knowledge search.
        is_empty = not response_text.strip() if isinstance(response_text, str) else not response_text
        if is_empty and tool_calls_made:
            # Check if any tool results look like knowledge base search results
            search_results = [r for r in tool_results if "[1]" in r or "Source:" in r]
            if search_results:
                logger.info("[agent] Empty post-tool response — synthesizing from %d search results", len(search_results))
                response_text = await self._synthesize_from_tool_results(
                    message, search_results, session_id,
                )
            else:
                # Tools were called but no search results (e.g., profile updates,
                # goal management).  Synthesize an acknowledgment based on what
                # the tools actually did rather than forcing a knowledge search.
                logger.info("[agent] Empty post-tool response with no search results — synthesizing acknowledgment")
                response_text = await self._synthesize_acknowledgment(
                    message, tool_calls_made, tool_results, session_id,
                )

        # Final fallback if synthesis also failed or no tool calls at all
        is_empty = not response_text.strip() if isinstance(response_text, str) else not response_text
        if is_empty:
            logger.warning("[agent] Empty response from agent — using fallback")
            response_text = (
                "I want to make sure I give you the best help. "
                "Could you tell me a bit more about what you'd like to focus on?"
            )

        # Record turn in conversation history
        self._session_store.add_message(session_id, "user", message, turn)
        self._session_store.add_message(session_id, "assistant", response_text, turn)

        # Build and persist enriched trace
        enriched = self._build_enriched_trace(
            session_id, turn, result, new_messages, tool_calls_made, total_ms,
        )
        self._session_store.save_trace(session_id, enriched)

        # Fire background memory tasks (non-blocking)
        if self._memory:
            asyncio.create_task(self._safe_background(
                self._memory.post_turn_tasks(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=response_text,
                    tool_calls=[tc for tc in tool_calls_made],
                ),
                "memory",
            ))

        # Fire background analyzer (non-blocking)
        if self._analyzer:
            asyncio.create_task(self._safe_background(
                self._analyzer.analyze_turn(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=response_text,
                    enriched_trace=enriched,
                ),
                "analyzer",
            ))

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

        return ChatResponse(
            response=response_text,
            agent_used="react_agent",
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

    async def _gate_check(self, response_text: str) -> str:
        """Run the output gate on a fallback response. Returns safe fallback on violation."""
        if not self._output_gate:
            return response_text
        try:
            check = await self._output_gate.check(response_text)
            if not check.is_valid:
                logger.info("[agent] Output gate caught fallback violation: %s", check.violation_type)
                return SAFE_OUTPUT_FALLBACK
        except Exception as e:
            logger.error("[agent] Output gate failed on fallback (allowing): %s", e)
        return response_text

    async def _synthesize_from_tool_results(
        self, user_message: str, tool_results: list[str], session_id: str,
    ) -> str:
        """Synthesize a coaching response from tool results via direct LLM call.

        This is a fallback for when Gemini flash-lite calls tools correctly
        but produces empty content in its final response.
        """
        if not self._gemini_client:
            return ""

        state = self._session_store.get(session_id)
        child_name = state.family_profile.child_name or "your child"

        results_text = "\n\n".join(tool_results)
        prompt = (
            f"You are a warm ADHD parenting coach. A parent said:\n"
            f"\"{user_message}\"\n\n"
            f"You searched the knowledge base and found these results:\n"
            f"{results_text}\n\n"
            f"Write a warm, practical response (under 200 words) that:\n"
            f"- Validates the parent's effort\n"
            f"- Shares 2-3 specific strategies from the search results\n"
            f"- Uses the child's name ({child_name}) naturally\n"
            f"- Gives concrete first steps\n"
            f"- Does NOT use emojis\n"
            f"Respond directly to the parent."
        )

        try:
            response = await self._gemini_client.generate(prompt, temperature=0.7)
            if response and response.strip():
                return await self._gate_check(response.strip())
        except Exception as e:
            logger.error("[agent] Synthesis fallback failed: %s", e)

        return ""

    async def _synthesize_acknowledgment(
        self,
        user_message: str,
        tool_calls: list[dict],
        tool_results: list[str],
        session_id: str,
    ) -> str:
        """Synthesize a contextual acknowledgment when non-search tools were called.

        Instead of forcing a knowledge-base search (which produces irrelevant
        strategy advice for messages like "My child is 7"), this generates a
        brief acknowledgment grounded in what the tools actually did.
        """
        if not self._gemini_client:
            return ""

        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        tool_summary = ", ".join(tool_names)
        results_text = "\n".join(tool_results) if tool_results else "(no details)"

        state = self._session_store.get(session_id)
        child_name = state.family_profile.child_name or "your child"

        prompt = (
            f"You are a warm ADHD parenting coach. A parent said:\n"
            f"\"{user_message}\"\n\n"
            f"You performed these actions: {tool_summary}\n"
            f"Tool outputs:\n{results_text}\n\n"
            f"Write a brief, warm acknowledgment (under 80 words) that:\n"
            f"- Confirms what you noted or updated\n"
            f"- Asks a natural follow-up question to keep the conversation going\n"
            f"- Uses the child's name ({child_name}) if appropriate\n"
            f"- Does NOT give unsolicited strategy advice\n"
            f"- Does NOT use emojis\n"
            f"Respond directly to the parent."
        )

        try:
            response = await self._gemini_client.generate(prompt, temperature=0.7)
            if response and response.strip():
                return await self._gate_check(response.strip())
        except Exception as e:
            logger.error("[agent] Acknowledgment synthesis failed: %s", e)

        return ""

    async def _search_and_synthesize(self, message: str, session_id: str) -> str:
        """Fallback: generate a direct response when the agent invocation fails."""
        if not self._gemini_client:
            return ""

        state = self._session_store.get(session_id)
        child_name = state.family_profile.child_name or "your child"

        prompt = (
            f"You are a warm ADHD parenting coach. A parent said:\n"
            f"\"{message}\"\n\n"
            f"Write a warm, helpful response (under 150 words) that:\n"
            f"- Validates the parent's concern\n"
            f"- Asks a clarifying question to better understand their situation\n"
            f"- Uses the child's name ({child_name}) if appropriate\n"
            f"- Does NOT give specific medical advice\n"
            f"- Does NOT use emojis\n"
            f"Respond directly to the parent."
        )

        try:
            response = await self._gemini_client.generate(prompt, temperature=0.7)
            if response and response.strip():
                return await self._gate_check(response.strip())
        except Exception as e:
            logger.error("[agent] Search-and-synthesize fallback failed: %s", e)

        return ""

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

        model_tier = result.get("model_tier", "standard")

        return EnrichedTrace(
            session_id=session_id,
            turn=turn,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_steps=pipeline_steps,
            total_duration_ms=total_ms,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_records,
            model_tier=model_tier,
            agent_used="react_agent",
        )

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
