"""Agent pipeline: input_gate -> prepare_context + ReAct agent -> output_gate.

Builds a custom StateGraph that wraps create_react_agent with guardrail gate
nodes. The ReAct agent's internal loop (reason -> tool -> reason -> respond)
is unchanged — we add gate nodes around it.
"""

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.agent.prompts import SAFE_OUTPUT_FALLBACK
from app.agent.state import CoachingState
from app.config import settings
from app.guardrails.validator import InputGate, OutputGate

logger = logging.getLogger(__name__)


def build_agent(
    tools: list,
    prepare_context,
    input_gate: InputGate,
    output_gate: OutputGate,
):
    """Build the full pipeline graph with guardrail gates.

    Graph topology:
        input_gate -> (blocked? -> END) | (allowed? -> react_agent -> output_gate -> END)

    Args:
        tools: List of LangChain tools the agent can call.
        prepare_context: Async function for context assembly (pre_model_hook).
        input_gate: InputGate instance for crisis + jailbreak classification.
        output_gate: OutputGate instance for output scope classification.

    Returns:
        Compiled LangGraph.
    """
    # Build the inner ReAct agent (with context assembly as pre_model_hook)
    if settings.MODEL_ROUTING_ENABLED:
        from app.agent.model_router import create_model_selector
        model = create_model_selector()
        logger.info(
            "Model routing enabled: fast=%s, standard=%s, complex=%s",
            settings.GEMINI_MODEL_FAST,
            settings.GEMINI_MODEL_STANDARD,
            settings.GEMINI_MODEL_COMPLEX,
        )
    else:
        model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.7,
            max_output_tokens=6000,
        )

    react_agent = create_react_agent(
        model=model,
        tools=tools,
        pre_model_hook=prepare_context,
        state_schema=CoachingState,
    )

    # Define gate node functions

    async def input_gate_node(state: CoachingState):
        """Run input gate classifier on the user message."""
        if input_gate is None:
            return {}

        # Find latest human message
        latest_human = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_human = msg
                break

        if not latest_human:
            return {}

        start = time.time()
        check = await input_gate.check(latest_human.content)
        duration_ms = (time.time() - start) * 1000

        trace_step = {
            "name": "input_gate",
            "duration_ms": duration_ms,
            "detail": {
                "is_allowed": check.is_allowed,
                "blocked_reason": check.blocked_reason,
            },
        }

        if not check.is_allowed:
            logger.info("Input gate blocked: %s", check.blocked_reason)
            return {
                "input_blocked": True,
                "block_response": check.override_response or "",
                "trace_steps": [trace_step],
                "messages": [AIMessage(content=check.override_response or "")],
            }

        return {"trace_steps": [trace_step]}

    async def output_gate_node(state: CoachingState):
        """Run output gate classifier on the agent's response."""
        if output_gate is None:
            return {}

        messages = state["messages"]

        # Find the last AI message (the agent's final response)
        last_ai = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                last_ai = msg
                break

        if not last_ai:
            return {}

        # Normalize content
        response_text = last_ai.content
        if not isinstance(response_text, str):
            from app.agent.orchestrator import _extract_text
            response_text = _extract_text(response_text)

        start = time.time()
        check = await output_gate.check(response_text)
        duration_ms = (time.time() - start) * 1000

        trace_step = {
            "name": "output_gate",
            "duration_ms": duration_ms,
            "detail": {
                "is_valid": check.is_valid,
                "violation_type": check.violation_type,
            },
        }

        if not check.is_valid:
            logger.info("Output gate triggered: %s", check.violation_type)
            safe_msg = AIMessage(content=SAFE_OUTPUT_FALLBACK)
            new_messages = [m for m in messages if m is not last_ai] + [safe_msg]
            return {
                "messages": new_messages,
                "trace_steps": [trace_step],
            }

        return {"trace_steps": [trace_step]}

    # Route after input gate
    def route_after_input_gate(state: CoachingState):
        if state.get("input_blocked"):
            return END
        return "react_agent"

    # Build the outer pipeline graph
    graph = StateGraph(CoachingState)

    graph.add_node("input_gate", input_gate_node)
    graph.add_node("react_agent", react_agent)
    graph.add_node("output_gate", output_gate_node)

    graph.set_entry_point("input_gate")
    graph.add_conditional_edges("input_gate", route_after_input_gate, {END: END, "react_agent": "react_agent"})
    graph.add_edge("react_agent", "output_gate")
    graph.add_edge("output_gate", END)

    compiled = graph.compile()

    logger.info(
        "Agent pipeline built: model=%s, tools=%d, routing=%s",
        settings.GEMINI_MODEL,
        len(tools),
        settings.MODEL_ROUTING_ENABLED,
    )
    return compiled
