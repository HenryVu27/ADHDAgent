"""Context engineering utilities for formatting conversation history."""

from app.config import settings


def format_conversation_window(
    history: list[dict],
    max_turns: int | None = None,
) -> str:
    """Format recent conversation turns for LLM prompt injection.

    Returns "Parent: {msg}\nCoach: {response}" pairs for the last max_turns turns.
    Production enhancement: summarize older turns beyond the window.
    """
    if not history:
        return ""
    if max_turns is None:
        max_turns = settings.CONTEXT_WINDOW_TURNS
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        lines.append(f"Parent: {turn.get('user_message', '')}")
        lines.append(f"Coach: {turn.get('agent_response', '')}")
    return "\n".join(lines)
