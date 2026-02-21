"""ReAct agent graph builder using LangGraph's create_react_agent."""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.agent.state import CoachingState
from app.config import settings

logger = logging.getLogger(__name__)


def build_agent(
    tools: list,
    pre_model_hook,
    post_model_hook,
):
    """Build and compile the ReAct agent graph.

    Args:
        tools: List of LangChain tools the agent can call.
        pre_model_hook: Async function run before each LLM call.
        post_model_hook: Async function run after final response.

    Returns:
        Compiled LangGraph agent.
    """
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

    agent = create_react_agent(
        model=model,
        tools=tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=CoachingState,
    )

    logger.info(
        "ReAct agent built: model=%s, tools=%d, routing=%s",
        settings.GEMINI_MODEL,
        len(tools),
        settings.MODEL_ROUTING_ENABLED,
    )
    return agent
