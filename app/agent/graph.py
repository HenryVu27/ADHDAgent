"""Agent pipeline: input_gate -> prepare_context + ReAct agent -> END.

Builds a custom StateGraph that wraps create_react_agent with an input
guardrail gate node. The output gate is run by the orchestrator AFTER
the graph finishes so it doesn't block token streaming.
"""

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage  # noqa: F401 — HumanMessage used in input_gate_node
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.agent.state import CoachingState
from app.config import settings
from app.guardrails.validator import InputGate

logger = logging.getLogger(__name__)


def build_agent(
    tools: list,
    prepare_context,
    input_gate: InputGate,
    output_gate=None,  # kept for API compat; no longer wired into graph
):
    """Build the pipeline graph with input gate + ReAct agent.

    Graph topology:
        input_gate -> (blocked? -> END) | (allowed? -> react_agent -> END)

    The output gate is intentionally excluded from the graph so that token
    streaming via astream_events is not blocked by the output gate API call.
    The orchestrator runs the output gate after the graph completes.

    Args:
        tools: List of LangChain tools the agent can call.
        prepare_context: Async function for context assembly (pre_model_hook).
        input_gate: InputGate instance for crisis + jailbreak classification.
        output_gate: Unused (kept for backward compatibility).

    Returns:
        Compiled LangGraph.
    """
    # Build two ReAct agents: Pro (complex queries) and Flash (simple messages)
    pro_model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_AGENT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.7,
        max_output_tokens=2048,
        thinking_budget=settings.GEMINI_THINKING_BUDGET,
    )
    flash_model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_FAST_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.7,
        max_output_tokens=2048,
        thinking_budget=settings.GEMINI_FAST_THINKING_BUDGET,
    )

    pro_react_agent = create_react_agent(
        model=pro_model,
        tools=tools,
        pre_model_hook=prepare_context,
        state_schema=CoachingState,
    )
    flash_react_agent = create_react_agent(
        model=flash_model,
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

        from app.agent.orchestrator import _extract_text
        start = time.time()
        user_text = _extract_text(latest_human.content)
        check = await input_gate.check(user_text)
        duration_ms = (time.time() - start) * 1000

        trace_step = {
            "name": "input_gate",
            "duration_ms": duration_ms,
            "detail": {
                "is_allowed": check.is_allowed,
                "blocked_reason": check.blocked_reason,
                "route": check.route,
                "fast_path_bypassed": check.fast_path_bypassed,
                "fast_path_score": check.fast_path_score,
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

        return {"trace_steps": [trace_step], "route": check.route}

    # Route after input gate: blocked -> END, simple -> flash, complex -> pro
    def route_after_input_gate(state: CoachingState):
        if state.get("input_blocked"):
            return END
        if state.get("route") == "flash":
            return "flash_react_agent"
        return "pro_react_agent"

    # Build the outer pipeline graph (no output gate — handled by orchestrator)
    graph = StateGraph(CoachingState)

    graph.add_node("input_gate", input_gate_node)
    graph.add_node("pro_react_agent", pro_react_agent)
    graph.add_node("flash_react_agent", flash_react_agent)

    graph.set_entry_point("input_gate")
    graph.add_conditional_edges("input_gate", route_after_input_gate, {
        END: END,
        "pro_react_agent": "pro_react_agent",
        "flash_react_agent": "flash_react_agent",
    })
    graph.add_edge("pro_react_agent", END)
    graph.add_edge("flash_react_agent", END)

    compiled = graph.compile()

    logger.info(
        "Agent pipeline built: pro_model=%s (thinking=%d), fast_model=%s (thinking=%d), tools=%d",
        settings.GEMINI_AGENT_MODEL,
        settings.GEMINI_THINKING_BUDGET,
        settings.GEMINI_FAST_MODEL,
        settings.GEMINI_FAST_THINKING_BUDGET,
        len(tools),
    )
    return compiled
