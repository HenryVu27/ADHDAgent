"""ReAct agent tools — the agent calls these to interact with knowledge and session state."""

import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agent.store_protocol import SessionStoreBase
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Module-level references set by create_tools()
_retriever: HybridRetriever | None = None
_session_store: SessionStoreBase | None = None


def _get_session_id(config: RunnableConfig) -> str:
    """Extract session_id from LangGraph config."""
    return config.get("configurable", {}).get("session_id", "default")


def create_tools(
    retriever: HybridRetriever,
    session_store: SessionStoreBase,
) -> list:
    """Create the 5 agent tools, bound to the given retriever and session store."""
    global _retriever, _session_store
    _retriever = retriever
    _session_store = session_store

    return [
        search_knowledge_base,
        get_family_profile,
        update_family_profile,
        track_outcome,
        manage_goals,
    ]


@tool
async def search_knowledge_base(
    query: str,
    document_type: Optional[str] = None,
    tags: Optional[list[str]] = None,
    config: RunnableConfig = None,
) -> str:
    """Search the ADHD parenting knowledge base for evidence-based strategies, facts, and guidance.

    Call this when you need to find strategies, research, or guidance to help the parent.
    Always search before recommending strategies so your advice is evidence-based.

    Args:
        query: What to search for (e.g., "homework strategies for 8 year old with ADHD")
        document_type: Optional filter — "strategy", "guidance", or "fact"
        tags: Optional tag filters (e.g., ["homework", "executive_function"])
    """
    from app.models.schemas import RetrievalFilters

    filters = None
    if document_type or tags:
        filters = RetrievalFilters(document_type=document_type, tags=tags)

    response = await _retriever.retrieve(query=query, filters=filters)

    if not response.results:
        return "No relevant documents found. Try a different search query."

    parts = []
    for i, result in enumerate(response.results, 1):
        source = f" (Source: {result.source})" if result.source else ""
        parts.append(
            f"[{i}] {result.document_name}{source}\n"
            f"    {result.content[:500]}"
        )

    return "\n\n".join(parts)


@tool
def get_family_profile(config: RunnableConfig = None) -> str:
    """Get the current family profile to check what you already know about this family.

    Call this before asking questions to avoid asking for information you already have.
    """
    session_id = _get_session_id(config)
    state = _session_store.get(session_id)
    profile = state.family_profile

    lines = []
    if profile.child_name:
        lines.append(f"Child's name: {profile.child_name}")
    if profile.child_age:
        lines.append(f"Child's age: {profile.child_age}")
    if profile.diagnosis_status:
        lines.append(f"Diagnosis status: {profile.diagnosis_status}")
    if profile.challenge_areas:
        lines.append(f"Challenge areas: {', '.join(profile.challenge_areas)}")
    if profile.attempted_strategies:
        lines.append(f"Strategies tried: {', '.join(profile.attempted_strategies)}")
    if profile.good_day_description:
        lines.append(f"Good day looks like: {profile.good_day_description}")
    if profile.hardest_situations:
        lines.append(f"Hardest situations: {', '.join(profile.hardest_situations)}")

    if not lines:
        return "No family profile information yet. Start by learning about the family."

    # Include active goals and strategies
    if state.active_strategies:
        lines.append(f"Active strategies: {', '.join(state.active_strategies)}")
    if state.goals:
        active_goals = [g.description for g in state.goals if g.status == "active"]
        if active_goals:
            lines.append(f"Active goals: {', '.join(active_goals)}")

    return "\n".join(lines)


@tool
def update_family_profile(
    child_name: Optional[str] = None,
    child_age: Optional[str] = None,
    diagnosis_status: Optional[str] = None,
    challenge_areas: Optional[list[str]] = None,
    attempted_strategies: Optional[list[str]] = None,
    good_day_description: Optional[str] = None,
    hardest_situations: Optional[list[str]] = None,
    config: RunnableConfig = None,
) -> str:
    """Update the family profile when you learn new information about the family.

    Call this whenever the parent mentions details about their child or situation.
    For example, if they say "my 7-year-old has trouble with homework", call this
    with child_age="7" and challenge_areas=["homework"].

    Args:
        child_name: The child's name
        child_age: The child's age (as a string, e.g. "7")
        diagnosis_status: ADHD diagnosis status (e.g. "diagnosed", "suspected", "evaluating")
        challenge_areas: Areas of difficulty (e.g. ["homework", "bedtime", "emotions"])
        attempted_strategies: Strategies already tried (e.g. ["timer", "reward chart"])
        good_day_description: What a good day looks like for the family
        hardest_situations: Specific hard situations (e.g. ["homework time", "morning routine"])
    """
    session_id = _get_session_id(config)

    updates = {
        k: v for k, v in {
            "child_name": child_name,
            "child_age": child_age,
            "diagnosis_status": diagnosis_status,
            "challenge_areas": challenge_areas,
            "attempted_strategies": attempted_strategies,
            "good_day_description": good_day_description,
            "hardest_situations": hardest_situations,
        }.items() if v is not None
    }

    if not updates:
        return "No updates provided."

    profile = _session_store.update_profile(session_id, **updates)

    updated_fields = list(updates.keys())
    return f"Profile updated: {', '.join(updated_fields)}. Current profile has {len([f for f in profile.model_dump().values() if f])} fields populated."


@tool
def track_outcome(
    strategy_name: str,
    outcome: str,
    notes: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Log whether a strategy worked, didn't work, or had mixed results.

    Call this when the parent reports back on how a strategy went.

    Args:
        strategy_name: Name of the strategy (e.g., "visual timer for homework")
        outcome: "positive", "negative", or "mixed"
        notes: Optional details about what happened
    """
    session_id = _get_session_id(config)

    if outcome not in ("positive", "negative", "mixed"):
        return f"Invalid outcome '{outcome}'. Must be 'positive', 'negative', or 'mixed'."

    entry = _session_store.add_outcome(
        session_id=session_id,
        strategy_name=strategy_name,
        outcome=outcome,
        notes=notes or "",
    )

    # Also track as active strategy if positive
    if outcome == "positive":
        _session_store.add_active_strategy(session_id, strategy_name)

    return f"Outcome recorded: '{strategy_name}' -> {outcome}." + (f" Notes: {notes}" if notes else "")


@tool
def manage_goals(
    action: str,
    description: Optional[str] = None,
    config: RunnableConfig = None,
) -> str:
    """Add, complete, or list the family's goals.

    Call this to help the family set and track goals for their ADHD management journey.

    Args:
        action: "add" to create a new goal, "complete" to mark one done, "list" to see all goals
        description: The goal description (required for "add" and "complete")
    """
    session_id = _get_session_id(config)

    if action not in ("add", "complete", "list"):
        return f"Invalid action '{action}'. Must be 'add', 'complete', or 'list'."

    if action in ("add", "complete") and not description:
        return f"Description required for '{action}' action."

    goals = _session_store.manage_goal(
        session_id=session_id,
        action=action,
        description=description or "",
    )

    if not goals:
        return "No goals set yet."

    lines = []
    for g in goals:
        status_marker = "[done]" if g.status == "completed" else "[active]"
        lines.append(f"  {status_marker} {g.description}")

    return f"Goals ({len(goals)} total):\n" + "\n".join(lines)
