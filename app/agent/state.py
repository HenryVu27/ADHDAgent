"""CoachingState — graph state schema for the ReAct agent."""

from typing import Annotated

from langgraph.graph import MessagesState
from langgraph.managed.is_last_step import RemainingStepsManager


class CoachingState(MessagesState):
    """Extends MessagesState with coaching-specific fields.

    MessagesState provides `messages: list[AnyMessage]` managed by LangGraph.
    remaining_steps is required by create_react_agent to cap the ReAct loop.
    """

    remaining_steps: Annotated[int, RemainingStepsManager]
    session_id: str
    input_blocked: bool = False
    block_response: str = ""
    trace_steps: Annotated[list[dict], lambda a, b: a + b] = []
    model_tier: str = "standard"
