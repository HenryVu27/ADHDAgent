"""AgentOrchestrator — thin wrapper: manages sessions and invokes the ReAct agent."""

import asyncio
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import traceable

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import (
    AgentReasoningStep,
    StreamDonePayload,
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


def _build_multipart_content(message: str, attachments: list[dict]):
    """Build HumanMessage content: plain string or multipart list with media parts.

    Args:
        message: The user's text message.
        attachments: List of attachment dicts with gemini_file_uri and content_type.

    Returns:
        str if no attachments, or list of content parts if attachments present.
    """
    if not attachments:
        return message
    parts = [{"type": "text", "text": message}]
    for att in attachments:
        parts.append({
            "type": "media",
            "file_uri": att["gemini_file_uri"],
            "mime_type": att["content_type"],
        })
    return parts


# LangGraph's create_react_agent emits this exact string when remaining_steps < 2
# and the model still wants to call tools (see chat_agent_executor.py line ~688).
# We also catch variations in case the model echoes or rephrases it.
_DEGRADED_PATTERNS = (
    "sorry, need more steps to process this request",  # exact LangGraph string
    "need more steps",
    "need additional steps",
    "unable to complete",
    "cannot complete",
    "more steps to process",
    "ran out of steps",
    "processing limit",
)


def _is_degraded_response(text: str) -> bool:
    """Return True if the response looks like an LLM 'gave up' message."""
    if not text or not text.strip():
        return True
    lowered = text.strip().lower()
    # Very short responses after tool calls are suspicious
    if len(lowered) < 40:
        return False  # Could be a legitimate brief reply like "You're welcome!"
    return any(pattern in lowered for pattern in _DEGRADED_PATTERNS)


class AgentOrchestrator:
    """Manages sessions and runs the compiled ReAct agent."""

    def __init__(self, agent, session_store: SessionStoreBase, memory_manager=None, analyzer=None, event_bus=None, output_gate=None, gemini_client=None):
        self._agent = agent
        self._session_store = session_store
        self._memory = memory_manager
        self._analyzer = analyzer
        self._event_bus = event_bus
        self._output_gate = output_gate
        self._gemini = gemini_client
        self._pending_tasks: set[asyncio.Task] = set()

    @traceable(name="orchestrator.generate_summary", run_type="llm")
    async def _generate_summary(self, message: str) -> str | None:
        """Generate a contextual one-line summary from the user's message via Flash."""
        if not self._gemini:
            return None
        try:
            prompt = (
                "You are an ADHD parenting coach's internal narrator. "
                "Summarize in under 10 words what you would focus on for this parent's message. "
                "Use present participle form. Do NOT include quotes or punctuation at the end.\n"
                "Examples:\n"
                "- Exploring bedtime routine strategies for a 7-year-old\n"
                "- Considering ways to handle homework meltdowns\n"
                "- Thinking about morning routine structure\n"
                "- Looking into positive reinforcement approaches\n\n"
                f"Parent's message: {message}"
            )
            result = await self._gemini.generate(
                prompt,
                temperature=0.3,
                max_output_tokens=512,
                timeout=5.0,
            )
            # Take only the first line, trim to last complete word
            summary = result.strip().split("\n")[0].strip().rstrip(".")
            if not summary:
                return None
            # Safety: if output looks truncated (no space = single partial word), drop it
            if " " not in summary:
                return None
            return summary
        except Exception:
            logger.debug("[agent] Summary generation failed — skipping")
            return None

    @traceable(name="orchestrator.generate_suggestions", run_type="llm")
    async def _generate_suggestions(
        self, user_message: str, assistant_response: str, session_id: str,
    ) -> list[str]:
        """Generate 0-3 contextual follow-up suggestions via Flash."""
        if not self._gemini:
            return []
        try:
            state = await self._session_store.get(session_id)
            child_name = state.family_profile.child_name or "your child"

            prompt = (
                "You generate follow-up message suggestions for a parent chatting with "
                "an ADHD parenting coach. Based on the conversation below, suggest 2 to 3 "
                "short follow-up messages the parent might want to send next.\n\n"
                "Rules:\n"
                "- Always return 2-3 suggestions — never return an empty array\n"
                "- Keep each suggestion SHORT: 3-8 words, under 40 characters\n"
                "- Write as the PARENT would speak (first person)\n"
                "- Be specific to what was just discussed, not generic\n"
                "- If the coach asked a question, suggest possible answers\n"
                "- If the coach gave strategies, suggest reactions or follow-ups\n"
                "- If the coach shared information, suggest ways to dig deeper\n"
                "- Return ONLY a single-line JSON array of strings, no newlines\n"
                f"- The child's name is {child_name}\n\n"
                f"Parent said: {user_message}\n\n"
                f"Coach replied: {assistant_response[:500]}\n\n"
                "JSON array:"
            )
            suggestions = await self._gemini.extract_json(
                prompt,
                temperature=0.7,
                max_output_tokens=150,
                timeout=10.0,
                disable_thinking=True,
            )
            logger.info("[agent] Suggestions result from LLM: %r", suggestions)
            if not isinstance(suggestions, list):
                return []
            return [
                s.strip() for s in suggestions
                if isinstance(s, str) and s.strip() and len(s.strip()) <= 60
            ][:3]
        except Exception as exc:
            logger.warning("[agent] Suggestion generation failed: %s", exc)
            return []

    async def get_session(self, session_id: str) -> SessionState:
        return await self._session_store.get(session_id)

    def get_session_store(self) -> SessionStoreBase:
        """Public accessor for the session store."""
        return self._session_store

    async def infer_phase(self, session_id: str) -> ConversationPhase:
        """Infer a phase label from session state."""
        return await self._infer_phase(session_id)

    async def seed_session(self, request: SeedSessionRequest, user_id: int | None = None) -> None:
        if user_id is not None:
            await self._session_store._ensure_session(request.session_id, user_id=user_id)
        await self._session_store.seed_session(request)

        # Write-through seed data to user-level profile
        if user_id is not None:
            seed_kwargs: dict = {}
            if request.parent_name:
                seed_kwargs["parent_name"] = request.parent_name
            if request.child_name:
                seed_kwargs["child_name"] = request.child_name
            if request.child_age:
                seed_kwargs["child_age"] = request.child_age
            if request.diagnosis_status:
                seed_kwargs["diagnosis_status"] = request.diagnosis_status
            if request.adhd_subtype:
                seed_kwargs["adhd_subtype"] = request.adhd_subtype
            if request.challenges:
                seed_kwargs["challenge_areas"] = request.challenges
            if request.tried_strategies:
                seed_kwargs["attempted_strategies"] = request.tried_strategies
            if seed_kwargs:
                await self._session_store.update_user_profile(user_id, **seed_kwargs)
                await self._session_store.commit()

    async def _build_history(
        self,
        session_id: str,
        message: str,
        attachment_ids: list[str] | None,
    ) -> tuple[list, bool]:
        """Build conversation history messages for agent input.

        Returns (messages_list, force_summary) where messages_list ends with the
        current user's HumanMessage and force_summary indicates context pressure.
        """
        from app.config import settings

        stored_messages = await self._session_store.get_messages(session_id)
        latest_summary = await self._session_store.get_latest_summary(session_id)
        summary_through_turn = latest_summary.covers_through_turn if latest_summary else 0

        history_att_ids: list[str] = []
        history_entries_with_atts: dict[int, list[str]] = {}
        history_messages = []
        unsummarized_chars = 0
        filtered_entries = []

        for entry in stored_messages:
            if entry.get("blocked"):
                continue
            msg_turn = entry.get("turn", 0)
            if summary_through_turn > 0 and msg_turn <= summary_through_turn:
                continue
            filtered_entries.append(entry)

        for idx, entry in enumerate(filtered_entries):
            if entry["role"] == "user":
                att_ids = entry.get("attachment_ids", [])
                if att_ids:
                    history_att_ids.extend(att_ids)
                    history_entries_with_atts[idx] = att_ids
                history_messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                history_messages.append(AIMessage(content=entry["content"]))
            unsummarized_chars += len(entry.get("content", ""))

        if history_att_ids:
            all_hist_atts = await self._session_store.get_attachments(history_att_ids)
            att_by_id = {a["id"]: a for a in all_hist_atts}
            for idx, att_ids in history_entries_with_atts.items():
                atts = [att_by_id[aid] for aid in att_ids if aid in att_by_id]
                if atts:
                    history_messages[idx] = HumanMessage(
                        content=_build_multipart_content(
                            history_messages[idx].content, atts,
                        )
                    )

        force_summary = (unsummarized_chars / settings.CONTEXT_MAX_CHARS) >= 0.8

        attachments = []
        if attachment_ids:
            attachments = await self._session_store.get_attachments(attachment_ids)
        message_content = _build_multipart_content(message, attachments)
        history_messages.append(HumanMessage(content=message_content))

        return history_messages, force_summary

    async def _handle_input_blocked(
        self,
        session_id: str,
        turn: int,
        message: str,
        result: dict,
        total_ms: float,
        attachment_ids: list[str] | None,
    ) -> StreamDonePayload:
        """Handle an input-gate block: persist, trace, and return payload."""
        response_text = result.get("block_response", "")
        blocked_reason = ""
        for step in result.get("trace_steps", []):
            if step.get("name") == "input_gate":
                blocked_reason = step.get("detail", {}).get("blocked_reason", "")

        await self._session_store.add_message(
            session_id, "user", message, turn,
            blocked=True, blocked_reason=blocked_reason,
            attachment_ids=attachment_ids,
        )
        await self._session_store.add_message(
            session_id, "assistant", response_text, turn,
            blocked=True, blocked_reason=blocked_reason,
        )

        trace = self._build_trace(result, total_ms)
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
        await self._session_store.save_trace(session_id, enriched)
        await self._session_store.commit()

        if self._event_bus:
            await self._event_bus.emit(
                "agent", "turn_blocked", session_id, turn, total_ms,
                detail={"reason": blocked_reason},
            )

        logger.info("[agent] === BLOCKED === session=%s, reason=input_gate", session_id)
        return StreamDonePayload(
            response=response_text,
            agent_used="input_gate",
            phase=await self._infer_phase(session_id),
            pipeline_trace=trace,
            session_id=session_id,
        )

    async def _persist_turn(
        self,
        session_id: str,
        turn: int,
        message: str,
        response_text: str,
        new_messages: list,
        tool_calls_made: list[dict],
        result: dict,
        total_ms: float,
        attachment_ids: list[str] | None,
    ) -> EnrichedTrace:
        """Persist messages, tool results, and trace for a completed turn."""
        tool_summary = self._build_tool_calls_summary(tool_calls_made)
        await self._session_store.add_message(
            session_id, "user", message, turn, attachment_ids=attachment_ids,
        )
        await self._session_store.add_message(
            session_id, "assistant", response_text, turn,
            tool_calls_summary=tool_summary,
        )

        for msg in new_messages:
            if isinstance(msg, ToolMessage):
                tc_name = ""
                tc_query = ""
                for tc in tool_calls_made:
                    if tc.get("id") == msg.tool_call_id:
                        tc_name = tc.get("name", "")
                        tc_query = str(tc.get("args", {}).get("query", ""))
                        break
                if tc_name in ("search_knowledge_base", "get_document_details"):
                    result_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    await self._session_store.save_tool_result(
                        session_id, tc_name, tc_query, result_text, turn,
                    )

        enriched = self._build_enriched_trace(
            session_id, turn, result, new_messages, tool_calls_made, total_ms,
        )
        await self._session_store.save_trace(session_id, enriched)
        await self._session_store.commit()
        return enriched

    @traceable(name="orchestrator.process")
    async def process(self, message: str, session_id: str, attachment_ids: list[str] | None = None) -> StreamDonePayload:
        """Run the ReAct agent for a single parent message."""
        turn = await self._session_store.increment_turn(session_id)
        logger.info(
            "[agent] === START === session=%s, turn=%d, message=%.80s",
            session_id, turn, message,
        )

        if self._event_bus:
            await self._event_bus.emit("agent", "turn_start", session_id, turn, detail={"message_preview": message[:80]})

        from app.config import settings

        messages, force_summary = await self._build_history(
            session_id, message, attachment_ids,
        )

        start = time.time()
        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }

        try:
            result = await self._agent.ainvoke(
                {"messages": messages, "session_id": session_id},
                config=config,
            )
        except Exception as e:
            total_ms = (time.time() - start) * 1000
            logger.warning("[agent] Agent invocation failed (%s) — using static fallback", type(e).__name__)
            response_text = (
                "I want to make sure I give you the best help. "
                "Could you tell me a bit more about what you'd like to focus on?"
            )
            await self._session_store.add_message(session_id, "user", message, turn, attachment_ids=attachment_ids)
            await self._session_store.add_message(session_id, "assistant", response_text, turn)
            await self._session_store.commit()
            trace = PipelineTrace(
                steps=[PipelineStep(name="react_agent", duration_ms=total_ms, detail={"error": str(e)})],
                total_duration_ms=total_ms,
                agent_used="react_agent_fallback",
            )
            return StreamDonePayload(
                response=response_text,
                agent_used="react_agent_fallback",
                phase=await self._infer_phase(session_id),
                pipeline_trace=trace,
                session_id=session_id,
            )

        total_ms = (time.time() - start) * 1000

        if result.get("input_blocked"):
            return await self._handle_input_blocked(
                session_id, turn, message, result, total_ms, attachment_ids,
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

        response_text, tool_calls_made = self._extract_response(new_messages)

        # Run output gate (outside graph to avoid blocking streaming)
        response_text, result = await self._run_output_gate(response_text, result)

        enriched = await self._persist_turn(
            session_id, turn, message, response_text,
            new_messages, tool_calls_made, result, total_ms, attachment_ids,
        )

        trace = self._build_trace(result, total_ms, tool_calls_made)

        await self._fire_background_tasks(
            session_id, turn, message, response_text,
            tool_calls_made, enriched, force_summary, total_ms,
        )

        logger.info(
            "[agent] === END === session=%s, tools=%d, duration=%.0fms",
            session_id, len(tool_calls_made), total_ms,
        )

        route = result.get("route", "pro")
        agent_label = "flash_react_agent" if route == "flash" else "react_agent"

        return StreamDonePayload(
            response=response_text,
            agent_used=agent_label,
            phase=await self._infer_phase(session_id),
            pipeline_trace=trace,
            session_id=session_id,
        )

    @traceable(name="orchestrator.process_stream")
    async def process_stream(self, message: str, session_id: str, user_id: int | None = None, attachment_ids: list[str] | None = None):
        """Streaming version of process(). Yields (event_type, data) tuples.

        Event types:
            status  {"text": str}       — pipeline stage updates
            token   {"text": str}       — one LLM response chunk
            reset   {}                  — clear streamed text (tool call starting)
            replace {"text": str}       — output gate replaced the streamed response
            done    {StreamDonePayload} — final payload with trace
            error   {"message": str}    — exception details
        """
        if user_id is not None:
            await self._session_store._ensure_session(session_id, user_id=user_id)
        turn = await self._session_store.increment_turn(session_id)
        logger.info(
            "[agent] === STREAM START === session=%s, turn=%d, message=%.80s",
            session_id, turn, message,
        )

        # Trigger longitudinal summary for returning users on first turn
        if turn == 1 and user_id is not None and self._memory:
            user_sessions = await self._session_store.get_all_sessions(user_id=user_id)
            previous = [s for s in user_sessions if s.session_id != session_id]
            if previous:
                self._track_task(
                    self._memory.end_of_session_tasks(user_id, previous[0].session_id),
                    "longitudinal_summary",
                )

        if self._event_bus:
            await self._event_bus.emit("agent", "turn_start", session_id, turn,
                                       detail={"message_preview": message[:80]})

        from app.config import settings

        messages, force_summary = await self._build_history(
            session_id, message, attachment_ids,
        )

        start = time.time()
        config = {
            "configurable": {"session_id": session_id},
            "recursion_limit": settings.AGENT_MAX_TOOL_STEPS * 2 + 5,
        }
        input_data = {
            "messages": messages,
            "session_id": session_id,
            "user_id": user_id,
        }

        result = None

        try:
            # Fire summary generation in parallel — non-blocking
            summary_task = asyncio.create_task(self._generate_summary(message))
            summary_text: str | None = None
            summary_sent = False

            token_count = 0
            token_start = None
            streamed_text = ""  # accumulate all streamed tokens for fallback
            async with asyncio.timeout(settings.CHAT_TIMEOUT_S):
                async for event in self._agent.astream_events(
                    input_data, config=config, version="v2"
                ):
                    # Check if summary is ready and hasn't been sent yet
                    if not summary_sent and summary_task.done():
                        try:
                            summary_text = summary_task.result()
                        except Exception:
                            summary_text = None
                        if summary_text:
                            yield ("summary", {"text": summary_text})
                        summary_sent = True

                    kind = event["event"]
                    metadata = event.get("metadata", {})
                    node = metadata.get("langgraph_node")

                    # Detect route from input gate — skip summary for simple messages
                    if kind == "on_chain_end" and node == "input_gate":
                        gate_output = event.get("data", {}).get("output", {})
                        if isinstance(gate_output, dict):
                            route = gate_output.get("route", "pro")
                            blocked = gate_output.get("input_blocked", False)
                            logger.info("[gate] input: %s route=%s", "BLOCKED" if blocked else "pass", route)
                            if route == "flash" and not summary_sent:
                                summary_task.cancel()
                                summary_sent = True

                    # Tool invocation — reset streamed text so only the final
                    # post-tool response is displayed (intermediate reasoning
                    # before tool calls is not the real response).
                    if kind == "on_tool_start":
                        if streamed_text:
                            streamed_text = ""
                            token_count = 0
                            token_start = None
                            yield ("reset", {})
                        tool_name = event.get("name", "")
                        tool_input = event.get("data", {}).get("input", {})

                        if tool_name == "search_knowledge_base":
                            query = tool_input.get("query", "strategies")
                            logger.info("[agent] tool: search_knowledge_base(%.70s)", query)
                            status = f"Searching for '{query}'..."
                        elif tool_name == "get_document_details":
                            status = "Reading document details..."
                        elif tool_name == "get_related_documents":
                            status = "Finding related strategies..."
                        elif tool_name == "update_family_profile":
                            fields = [k for k, v in tool_input.items()
                                      if v is not None and k != "config"]
                            logger.info("[agent] tool: update_family_profile(%s)", ", ".join(fields))
                            if fields:
                                status = f"Noting {', '.join(fields[:2])}..."
                            else:
                                status = "Updating your profile..."
                        elif tool_name == "track_outcome":
                            strategy = tool_input.get("strategy_name", "a strategy")
                            logger.info("[agent] tool: track_outcome(%s)", strategy)
                            status = f"Recording how {strategy} went..."
                        elif tool_name == "manage_goals":
                            action = tool_input.get("action", "managing")
                            desc = tool_input.get("description", "")
                            logger.info("[agent] tool: manage_goals(action=%s, %.50s)", action, desc)
                            action_verb = {"add": "Adding", "complete": "Completing", "list": "Listing"}.get(action, action.capitalize() + "ing")
                            if desc:
                                short_desc = desc[:40].rstrip()
                                status = f"{action_verb} goal: {short_desc}..."
                            else:
                                status = "Reviewing goals..."
                        elif tool_name == "get_family_profile":
                            logger.info("[agent] tool: get_family_profile")
                            status = "Reviewing your family's info..."
                        elif tool_name == "search_web":
                            query = tool_input.get("query", "")
                            logger.info("[agent] tool: search_web(%.70s)", query)
                            status = f"Searching the web for '{query[:50]}'..."
                        else:
                            logger.info("[agent] tool: %s", tool_name)
                            status = "Working on it..."
                        yield ("status", {"text": status})

                    # Stream LLM tokens (text only, skip tool-call chunks)
                    elif kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        # DEBUG: log every stream event (set LOG_LEVEL=DEBUG to see)
                        logger.debug(
                            "[stream-debug] on_chat_model_stream node=%s content_type=%s",
                            node,
                            type(chunk.content).__name__ if chunk and hasattr(chunk, "content") else "N/A",
                        )
                        if (
                            chunk
                            and hasattr(chunk, "content")
                            and chunk.content
                            and not getattr(chunk, "tool_call_chunks", None)
                            and not getattr(chunk, "tool_calls", None)
                        ):
                            if isinstance(chunk.content, str):
                                text = chunk.content
                            elif isinstance(chunk.content, list):
                                # Gemini content parts: extract "text" parts,
                                # skip "thinking" parts
                                text = "".join(
                                    part.get("text", "")
                                    for part in chunk.content
                                    if isinstance(part, dict)
                                    and part.get("type") == "text"
                                )
                            else:
                                text = ""
                            if text:
                                if token_count == 0:
                                    # Ensure summary is sent before first token
                                    if not summary_sent:
                                        if not summary_task.done():
                                            try:
                                                await asyncio.wait_for(
                                                    asyncio.shield(summary_task), timeout=0.5
                                                )
                                            except (TimeoutError, asyncio.TimeoutError):
                                                pass
                                        if summary_task.done():
                                            try:
                                                summary_text = summary_task.result()
                                            except Exception:
                                                summary_text = None
                                            if summary_text:
                                                yield ("summary", {"text": summary_text})
                                        summary_sent = True

                                token_count += 1
                                streamed_text += text
                                now = time.time()
                                if token_start is None:
                                    token_start = now
                                logger.debug(
                                    "[stream-debug] token #%d at +%.0fms",
                                    token_count,
                                    (now - token_start) * 1000,
                                )
                                yield ("token", {"text": text})

                    # Capture final state from the last chain-end with messages
                    elif kind == "on_chain_end":
                        output = event.get("data", {}).get("output")
                        if (
                            output
                            and isinstance(output, dict)
                            and "messages" in output
                        ):
                            result = output

        except TimeoutError:
            if not summary_task.done():
                summary_task.cancel()
            total_ms = (time.time() - start) * 1000
            if streamed_text.strip() and len(streamed_text.strip()) > 50:
                logger.warning(
                    "[agent] Timed out after %.0fms — using streamed text (%d chars)",
                    total_ms, len(streamed_text),
                )
                response_text = streamed_text.strip()
                await self._session_store.add_message(session_id, "user", message, turn, attachment_ids=attachment_ids)
                await self._session_store.add_message(session_id, "assistant", response_text, turn)
                await self._session_store.commit()
                yield (
                    "done",
                    StreamDonePayload(
                        response=response_text,
                        agent_used="react_agent_partial",
                        phase=await self._infer_phase(session_id),
                        pipeline_trace=PipelineTrace(
                            steps=[PipelineStep(name="react_agent", duration_ms=total_ms, detail={"error": "timeout"})],
                            total_duration_ms=total_ms,
                            agent_used="react_agent_partial",
                        ),
                        session_id=session_id,
                        summary=None,
                    ).model_dump(),
                )
            else:
                yield ("error", {"message": f"Response timed out after {settings.CHAT_TIMEOUT_S}s"})
            return

        except Exception as e:
            if not summary_task.done():
                summary_task.cancel()
            total_ms = (time.time() - start) * 1000

            # If we already streamed meaningful content, use it instead of
            # a generic fallback — the user saw it and it's better than nothing.
            if streamed_text.strip() and len(streamed_text.strip()) > 50:
                logger.warning(
                    "[agent] Stream failed (%s) — using streamed text (%d chars) as response",
                    type(e).__name__, len(streamed_text),
                )
                response_text = streamed_text.strip()
                agent_label = "react_agent_partial"
            else:
                logger.warning(
                    "[agent] Stream failed (%s) — using static fallback",
                    type(e).__name__,
                )
                response_text = (
                    "I want to make sure I give you the best help. "
                    "Could you tell me a bit more about what you'd like to focus on?"
                )
                agent_label = "react_agent_fallback"

            await self._session_store.add_message(session_id, "user", message, turn, attachment_ids=attachment_ids)
            await self._session_store.add_message(
                session_id, "assistant", response_text, turn
            )
            await self._session_store.commit()
            trace = PipelineTrace(
                steps=[
                    PipelineStep(
                        name="react_agent",
                        duration_ms=total_ms,
                        detail={"error": str(e)},
                    )
                ],
                total_duration_ms=total_ms,
                agent_used=agent_label,
            )
            # Only send error event if we don't have streamed content to show
            if agent_label == "react_agent_fallback":
                yield ("error", {"message": str(e)})
            yield (
                "done",
                StreamDonePayload(
                    response=response_text,
                    agent_used=agent_label,
                    phase=await self._infer_phase(session_id),
                    pipeline_trace=trace,
                    session_id=session_id,
                    summary=None,
                ).model_dump(),
            )
            return

        total_ms = (time.time() - start) * 1000

        if result is None:
            yield ("error", {"message": "No result from agent"})
            return

        # --- Input blocked ---
        if result.get("input_blocked"):
            payload = await self._handle_input_blocked(
                session_id, turn, message, result, total_ms, attachment_ids,
            )
            yield ("done", payload.model_dump())
            return

        # --- Normal response processing ---
        all_messages = result.get("messages", [])
        last_human_idx = -1
        for i in range(len(all_messages) - 1, -1, -1):
            if isinstance(all_messages[i], HumanMessage):
                last_human_idx = i
                break
        new_messages = (
            all_messages[last_human_idx + 1:] if last_human_idx >= 0 else []
        )

        response_text, tool_calls_made = self._extract_response(new_messages, streamed_text=streamed_text)

        enriched = await self._persist_turn(
            session_id, turn, message, response_text,
            new_messages, tool_calls_made, result, total_ms, attachment_ids,
        )

        trace = self._build_trace(result, total_ms, tool_calls_made)

        route = result.get("route", "pro")
        agent_label = (
            "flash_react_agent" if route == "flash" else "react_agent"
        )

        logger.info(
            "[agent] === STREAM END === session=%s, model=%s, tools=%d, duration=%.0fms",
            session_id, enriched.model_tier if enriched else "?", len(tool_calls_made), total_ms,
        )

        # Small yield to flush token events to the client before sending done.
        # Without this, tokens and done can land in the same TCP packet, causing
        # the frontend to batch them into a single React render (streaming bubble
        # never shown).
        await asyncio.sleep(0.05)

        # Fire suggestion generation early so it runs during the flush
        suggestions_task = asyncio.create_task(
            self._generate_suggestions(message, response_text, session_id)
        )

        yield (
            "done",
            StreamDonePayload(
                response=response_text,
                agent_used=agent_label,
                phase=await self._infer_phase(session_id),
                pipeline_trace=trace,
                session_id=session_id,
                summary=summary_text,
            ).model_dump(),
        )

        # Yield suggestions after done — short timeout so the stream closes quickly
        try:
            suggestions = await asyncio.wait_for(suggestions_task, timeout=6.0)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[agent] Suggestion generation timed out")
            suggestions = []
        except Exception as exc:
            logger.warning("[agent] Suggestion generation failed: %s", exc)
            suggestions = []
        logger.info("[agent] Suggestions result: %s", suggestions)
        if suggestions:
            yield ("suggestions", {"suggestions": suggestions})

        # Background tasks (fire after done so they don't block the response)
        if self._output_gate:
            self._track_task(
                self._run_output_gate_background(session_id, turn, response_text),
                "output_gate",
            )
        await self._fire_background_tasks(
            session_id, turn, message, response_text,
            tool_calls_made, enriched, force_summary, total_ms,
        )

    async def _fire_background_tasks(
        self,
        session_id: str,
        turn: int,
        message: str,
        response_text: str,
        tool_calls_made: list[dict],
        enriched: EnrichedTrace,
        force_summary: bool,
        total_ms: float,
    ) -> None:
        """Fire non-blocking memory, analyzer, and event_bus tasks."""
        if self._memory:
            self._track_task(
                self._memory.post_turn_tasks(
                    session_id=session_id,
                    turn=turn,
                    user_message=message,
                    assistant_response=response_text,
                    tool_calls=[tc for tc in tool_calls_made],
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
                    assistant_response=response_text,
                    enriched_trace=enriched,
                ),
                "analyzer",
            )
        if self._event_bus:
            await self._event_bus.emit(
                "agent", "turn_end", session_id, turn, total_ms,
                detail={
                    "tools": len(tool_calls_made),
                    "model_tier": enriched.model_tier,
                },
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

    async def _run_output_gate_background(self, session_id: str, turn: int, response_text: str) -> None:
        """Run output gate as a background task. If violation detected, log it."""
        try:
            check = await self._output_gate.check(response_text)
            if not check.is_valid:
                logger.warning(
                    "[agent] Output gate violation (background): session=%s turn=%d type=%s",
                    session_id, turn, check.violation_type,
                )
        except Exception as e:
            logger.warning("[agent] Output gate background check failed: %s", e)

    async def _run_output_gate(self, response_text: str, result: dict) -> tuple[str, dict]:
        """Run output gate check and return (possibly replaced) response + updated result."""
        if not self._output_gate:
            return response_text, result

        from app.agent.prompts import SAFE_OUTPUT_FALLBACK

        start = time.time()
        try:
            check = await self._output_gate.check(response_text)
        except Exception as e:
            logger.warning("[agent] Output gate failed (%s) — allowing response", type(e).__name__)
            duration_ms = (time.time() - start) * 1000
            trace_step = {"name": "output_gate", "duration_ms": duration_ms, "detail": {"is_valid": True, "error": str(e)}}
            result.setdefault("trace_steps", []).append(trace_step)
            return response_text, result

        duration_ms = (time.time() - start) * 1000
        trace_step = {
            "name": "output_gate",
            "duration_ms": duration_ms,
            "detail": {"is_valid": check.is_valid, "violation_type": check.violation_type},
        }
        result.setdefault("trace_steps", []).append(trace_step)

        if not check.is_valid:
            logger.info("[gate] output: BLOCKED type=%s (%.0fms)", check.violation_type, duration_ms)
            return SAFE_OUTPUT_FALLBACK, result

        logger.info("[gate] output: pass (%.0fms)", duration_ms)
        return response_text, result

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
    def _extract_response(
        new_messages: list,
        streamed_text: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Extract final response text and tool calls from agent output messages.

        Args:
            new_messages: Messages produced after the user's HumanMessage.
            streamed_text: Accumulated streamed tokens (streaming path only).

        Returns:
            (response_text, tool_calls_made)
        """
        response_text = ""
        all_ai_texts: list[str] = []
        tool_calls_made: list[dict] = []

        for msg in new_messages:
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_calls_made.extend(msg.tool_calls)
                elif msg.content:
                    text = _extract_text(msg.content)
                    if text.strip():
                        all_ai_texts.append(text.strip())
                    response_text = text

        is_empty = not response_text.strip() if isinstance(response_text, str) else not response_text
        degraded = _is_degraded_response(response_text) if not is_empty else False

        # Recovery priority:
        # 1. streamed_text (if available and substantial -- streaming path)
        # 2. all_ai_texts (earlier good AI messages -- both paths)
        # 3. static fallback
        if is_empty or degraded:
            if streamed_text and streamed_text.strip():
                logger.warning(
                    "[agent] %s — using streamed text (%d chars)",
                    "Degraded final response" if degraded else "Empty final AIMessage",
                    len(streamed_text),
                )
                response_text = streamed_text.strip()
            else:
                good_texts = [t for t in all_ai_texts if not _is_degraded_response(t)]
                if good_texts:
                    logger.warning(
                        "[agent] %s — using combined AI text",
                        "Degraded final response" if degraded else "Empty final AIMessage",
                    )
                    response_text = "\n\n".join(good_texts)
                else:
                    logger.warning(
                        "[agent] %s — using static fallback",
                        "Degraded response" if degraded else "Empty response from agent",
                    )
                    response_text = (
                        "I want to make sure I give you the best help. "
                        "Could you tell me a bit more about what you'd like to focus on?"
                    )
        elif (
            not is_empty
            and not degraded
            and streamed_text
            and streamed_text.strip()
            and len(response_text.strip()) < len(streamed_text.strip()) * 0.3
            and len(streamed_text.strip()) > 100
        ):
            logger.warning(
                "[agent] Degraded final response (%d chars) vs streamed (%d chars) — using streamed text",
                len(response_text), len(streamed_text),
            )
            response_text = streamed_text.strip()
        elif (
            not is_empty
            and not degraded
            and len(all_ai_texts) > 1
            and len(response_text.strip()) < 80
        ):
            combined = "\n\n".join(all_ai_texts[:-1])
            if len(combined) > len(response_text.strip()) * 2:
                logger.warning("[agent] Degraded final response — using combined AI text")
                response_text = combined

        return response_text, tool_calls_made

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
            elif name == "get_document_details":
                doc_id = args.get("document_id", "")
                parts.append(f'get_document_details(document_id="{doc_id}")')
            elif name == "get_related_documents":
                doc_id = args.get("document_id", "")
                parts.append(f'get_related_documents(document_id="{doc_id}")')
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
