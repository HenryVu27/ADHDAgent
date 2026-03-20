"""ReAct agent tools — closure factory binds dependencies per agent instance."""

import logging
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agent.store_protocol import SessionStoreBase
from app.models.schemas import RetrievalResult
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def _get_session_id(config: RunnableConfig) -> str:
    """Extract session_id from LangGraph config."""
    return config.get("configurable", {}).get("session_id", "default")


def _get_user_id(config: RunnableConfig) -> int | None:
    """Extract user_id from LangGraph config."""
    return config.get("configurable", {}).get("user_id")


def _age_to_range(age_str: str) -> str | None:
    """Map a child's age string to an age_range filter value."""
    try:
        age = int(age_str)
    except (ValueError, TypeError):
        return None
    if age <= 5:
        return "preschool"
    elif age <= 12:
        return "school_age"
    else:
        return "adolescent"


def _format_summary(i: int, result: RetrievalResult) -> str:
    """Format a single retrieval result as a compact summary (no steps/key_points)."""
    lines = []

    # Header: name + score
    score_str = f"  (score: {result.score:.2f})" if result.score else ""
    lines.append(f"[{i}] {result.document_name}{score_str}")

    # Evidence + Ages
    meta_parts = []
    if result.evidence_level:
        meta_parts.append(f"Evidence: {result.evidence_level}")
    if result.age_range:
        meta_parts.append(f"Ages: {', '.join(result.age_range)}")
    if meta_parts:
        lines.append(f"    {' | '.join(meta_parts)}")

    # Tags
    if result.tags:
        lines.append(f"    Tags: {', '.join(result.tags)}")

    # Document ID
    lines.append(f"    ID: {result.document_id}")

    # Truncated description (~150 chars)
    doc = result.full_doc
    description = doc.get("description", "") if doc else ""
    if description:
        truncated = description[:150].rstrip()
        if len(description) > 150:
            truncated += "..."
        lines.append(f"    {truncated}")

    return "\n".join(lines)


def _format_result(i: int, result: RetrievalResult) -> str:
    """Format a single retrieval result with structured metadata."""
    doc = result.full_doc
    lines = []

    # Header: name + metadata line
    meta_parts = []
    if result.evidence_level:
        meta_parts.append(f"Evidence: {result.evidence_level}")
    if result.age_range:
        meta_parts.append(f"Ages: {', '.join(result.age_range)}")
    if result.source:
        meta_parts.append(f"Source: {result.source}")
    meta_line = " | ".join(meta_parts) if meta_parts else ""

    score_str = f"  (score: {result.score:.2f})" if result.score else ""
    lines.append(f"[{i}] {result.document_name}{score_str}")
    if meta_line:
        lines.append(f"    {meta_line}")
    if result.tags:
        lines.append(f"    Tags: {', '.join(result.tags)}")
    lines.append(f"    ID: {result.document_id}")

    # Description
    description = doc.get("description", "") if doc else ""
    if description:
        lines.append(f"    {description}")

    # Steps (numbered list for strategy docs)
    steps = doc.get("steps", []) if doc else []
    if steps:
        lines.append("    Steps:")
        for j, step in enumerate(steps, 1):
            lines.append(f"      {j}. {step}")

    # Key points (bulleted list for guidance/fact docs)
    key_points = doc.get("key_points", []) if doc else []
    if key_points:
        lines.append("    Key points:")
        for point in key_points:
            lines.append(f"      - {point}")

    # Citations
    citations = result.citations
    if citations:
        cite_names = [c.get("source_name", "") for c in citations if c.get("source_name")]
        if cite_names:
            lines.append(f"    Citations: {'; '.join(cite_names)}")

    return "\n".join(lines)


def create_tools(
    retriever: HybridRetriever,
    session_store: SessionStoreBase,
    graphiti_client=None,
) -> list:
    """Create the 7 agent tools, bound to the given retriever and session store.

    Each tool closes over the provided dependencies — no module-level globals.
    Multiple calls with different dependencies produce independent tool sets.
    """

    @tool
    async def search_knowledge_base(
        query: str,
        document_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
        age_range: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Search the ADHD parenting knowledge base for curated strategies, facts, and guidance.

        Returns summaries with relevance scores. Use topic/strategy keywords, not emotional phrasing.
        Search once per topic — if results are not relevant, answer from your own knowledge instead of retrying.

        Args:
            query: What to search for using topic/strategy keywords (e.g., "attention focus strategies executive function")
            document_type: Optional filter — "strategy", "guidance", or "fact"
            tags: Optional tag filters (e.g., ["homework", "executive_function"])
            age_range: Optional age filter — "preschool", "school_age", or "adolescent"
        """
        try:
            from app.models.schemas import RetrievalFilters

            session_id = _get_session_id(config)
            state = await session_store.get(session_id)

            # Auto-apply age filter from family profile if not explicitly provided
            effective_age_range = age_range
            if not effective_age_range and state.family_profile.child_age:
                effective_age_range = _age_to_range(state.family_profile.child_age)

            filters = None
            if document_type or tags or effective_age_range:
                filters = RetrievalFilters(
                    document_type=document_type,
                    tags=tags,
                    age_range=effective_age_range,
                )

            response = await retriever.retrieve(
                query=query, filters=filters, state=state, skip_rewrite=True,
            )

            if not response.results:
                return "No relevant documents found. Try a different search query."

            # Quality summary line
            scores = [r.score for r in response.results]
            quality_line = f"{len(response.results)} results (top: {max(scores):.2f}, lowest: {min(scores):.2f})"

            parts = [quality_line, ""]
            for i, result in enumerate(response.results, 1):
                parts.append(_format_summary(i, result))

            return "\n\n".join(parts)
        except Exception as e:
            logger.exception("search_knowledge_base failed")
            return f"Error: knowledge base search failed ({type(e).__name__}). Try a simpler query or different terms."

    @tool
    async def get_document_details(
        document_id: str,
        config: RunnableConfig = None,
    ) -> str:
        """Get the full content of a specific knowledge base document by its ID.

        After scanning search summaries, read the full content of the most relevant documents.
        Always read details before recommending specific steps to a parent.

        Args:
            document_id: The document ID from search results (e.g., "working_memory_scaffolds_daily_tasks")
        """
        try:
            doc = retriever._store.get_document_by_id(document_id)
            if not doc:
                return f"Document '{document_id}' not found. Check the ID from search results."

            lines = []

            # Header
            lines.append(doc.get("name", ""))

            # Metadata
            meta_parts = []
            if doc.get("evidence_level"):
                meta_parts.append(f"Evidence: {doc['evidence_level']}")
            if doc.get("source"):
                meta_parts.append(f"Source: {doc['source']}")
            if meta_parts:
                lines.append(" | ".join(meta_parts))

            if doc.get("age_range"):
                lines.append(f"Ages: {', '.join(doc['age_range'])}")
            if doc.get("tags"):
                lines.append(f"Tags: {', '.join(doc['tags'])}")

            # Description
            if doc.get("description"):
                lines.append("")
                lines.append(doc["description"])

            # Steps
            steps = doc.get("steps", [])
            if steps:
                lines.append("")
                lines.append("Steps:")
                for j, step in enumerate(steps, 1):
                    lines.append(f"  {j}. {step}")

            # Key points
            key_points = doc.get("key_points", [])
            if key_points:
                lines.append("")
                lines.append("Key points:")
                for point in key_points:
                    lines.append(f"  - {point}")

            # Citations
            citations = doc.get("citations", [])
            if citations:
                cite_names = [c.get("source_name", "") for c in citations if c.get("source_name")]
                if cite_names:
                    lines.append(f"\nCitations: {'; '.join(cite_names)}")

            # Related documents
            related_docs = retriever._store.get_related_docs(document_id)
            if related_docs:
                related_names = [d.get("name", "") for d in related_docs]
                lines.append(f"\nRelated documents: {', '.join(related_names)}")

            return "\n".join(lines)
        except Exception as e:
            logger.exception("get_document_details failed")
            return f"Error: could not retrieve document ({type(e).__name__}). Check the document ID and try again."

    @tool
    async def get_related_documents(
        document_id: str,
        config: RunnableConfig = None,
    ) -> str:
        """Get summaries of documents related to a specific knowledge base document.

        Explore connected topics when search results are close but not quite right.
        Useful for finding connected strategies or facts.

        Args:
            document_id: The document ID to find related documents for
        """
        try:
            doc = retriever._store.get_document_by_id(document_id)
            if not doc:
                return f"Document '{document_id}' not found. Check the ID from search results."

            related_docs = retriever._store.get_related_docs(document_id)
            if not related_docs:
                return f"No related documents found for '{doc.get('name', document_id)}'. Try searching with different terms instead."

            doc_name = doc.get("name", document_id)
            header = f'{len(related_docs)} related documents for "{doc_name}":'

            parts = [header, ""]
            for i, related in enumerate(related_docs, 1):
                # Build a RetrievalResult-like summary without score
                lines = []
                lines.append(f"[{i}] {related.get('name', '')}")

                meta_parts = []
                if related.get("evidence_level"):
                    meta_parts.append(f"Evidence: {related['evidence_level']}")
                if related.get("age_range"):
                    meta_parts.append(f"Ages: {', '.join(related['age_range'])}")
                if meta_parts:
                    lines.append(f"    {' | '.join(meta_parts)}")

                if related.get("tags"):
                    lines.append(f"    Tags: {', '.join(related['tags'])}")

                lines.append(f"    ID: {related.get('id', '')}")

                description = related.get("description", "")
                if description:
                    truncated = description[:150].rstrip()
                    if len(description) > 150:
                        truncated += "..."
                    lines.append(f"    {truncated}")

                parts.append("\n".join(lines))

            return "\n\n".join(parts)
        except Exception as e:
            logger.exception("get_related_documents failed")
            return f"Error: could not retrieve related documents ({type(e).__name__}). Try searching with different terms instead."

    @tool
    async def get_family_profile(config: RunnableConfig = None) -> str:
        """Get the current family profile to check what you already know about this family.

        Call this before asking questions to avoid asking for information you already have.
        """
        try:
            session_id = _get_session_id(config)
            state = await session_store.get(session_id)
            profile = state.family_profile

            lines = []
            if profile.child_name:
                lines.append(f"Child's name: {profile.child_name}")
            if profile.child_age:
                lines.append(f"Child's age: {profile.child_age}")
            if profile.diagnosis_status:
                lines.append(f"Diagnosis status: {profile.diagnosis_status}")
            if profile.adhd_subtype:
                lines.append(f"ADHD subtype: {profile.adhd_subtype}")
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
        except Exception as e:
            logger.exception("get_family_profile failed")
            return f"Error: could not load family profile ({type(e).__name__}). Try again."

    @tool
    async def update_family_profile(
        child_name: Optional[str] = None,
        child_age: Optional[str] = None,
        diagnosis_status: Optional[str] = None,
        adhd_subtype: Optional[str] = None,
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
            adhd_subtype: ADHD subtype (e.g. "inattentive", "hyperactive-impulsive", "combined")
            challenge_areas: Areas of difficulty (e.g. ["homework", "bedtime", "emotions"])
            attempted_strategies: Strategies already tried (e.g. ["timer", "reward chart"])
            good_day_description: What a good day looks like for the family
            hardest_situations: Specific hard situations (e.g. ["homework time", "morning routine"])
        """
        try:
            session_id = _get_session_id(config)

            updates = {
                k: v for k, v in {
                    "child_name": child_name,
                    "child_age": child_age,
                    "diagnosis_status": diagnosis_status,
                    "adhd_subtype": adhd_subtype,
                    "challenge_areas": challenge_areas,
                    "attempted_strategies": attempted_strategies,
                    "good_day_description": good_day_description,
                    "hardest_situations": hardest_situations,
                }.items() if v is not None
            }

            if not updates:
                return "No updates provided."

            profile = await session_store.update_profile(session_id, **updates)

            updated_fields = list(updates.keys())
            return f"Profile updated: {', '.join(updated_fields)}. Current profile has {len([f for f in profile.model_dump().values() if f])} fields populated."
        except Exception as e:
            logger.exception("update_family_profile failed")
            return f"Error: could not update family profile ({type(e).__name__}). Try again with fewer fields."

    @tool
    async def track_outcome(
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
        if outcome not in ("positive", "negative", "mixed"):
            return f"Invalid outcome '{outcome}'. Must be 'positive', 'negative', or 'mixed'."

        try:
            session_id = _get_session_id(config)

            entry = await session_store.add_outcome(
                session_id=session_id,
                strategy_name=strategy_name,
                outcome=outcome,
                notes=notes or "",
            )

            # Also track as active strategy if positive
            if outcome == "positive":
                await session_store.add_active_strategy(session_id, strategy_name)

            return f"Outcome recorded: '{strategy_name}' -> {outcome}." + (f" Notes: {notes}" if notes else "")
        except Exception as e:
            logger.exception("track_outcome failed")
            return f"Error: could not record outcome ({type(e).__name__}). Try again."

    @tool
    async def manage_goals(
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
        if action not in ("add", "complete", "list"):
            return f"Invalid action '{action}'. Must be 'add', 'complete', or 'list'."

        if action in ("add", "complete") and not description:
            return f"Description required for '{action}' action."

        try:
            session_id = _get_session_id(config)

            goals = await session_store.manage_goal(
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
        except Exception as e:
            logger.exception("manage_goals failed")
            return f"Error: could not manage goals ({type(e).__name__}). Try again."

    all_tools = [
        search_knowledge_base,
        get_document_details,
        get_related_documents,
        get_family_profile,
        update_family_profile,
        track_outcome,
        manage_goals,
    ]

    if graphiti_client is not None:
        from app.config import settings

        @tool
        async def search_memory(
            query: str,
            config: RunnableConfig = None,
        ) -> str:
            """Search conversation memory for what this family has shared,
            tried, or experienced. Use when you need to recall past
            discussions, strategy outcomes, emotional patterns, or
            family context from previous turns or sessions."""
            user_id = _get_user_id(config)
            group_ids = [str(user_id)] if user_id is not None else None

            try:
                edges = await graphiti_client.search(
                    query,
                    group_ids=group_ids,
                    num_results=settings.GRAPHITI_SEARCH_RESULTS,
                )
            except Exception as e:
                logger.error("search_memory failed: %s", e)
                return "Memory search is temporarily unavailable."

            if not edges:
                return "No relevant memories found for this query."

            lines = []
            for i, edge in enumerate(edges, 1):
                fact = getattr(edge, "fact", str(edge))
                valid_at = getattr(edge, "valid_at", None)
                invalid_at = getattr(edge, "invalid_at", None)
                status = ""
                if invalid_at is not None:
                    status = " [no longer active]"
                date_str = ""
                if valid_at:
                    date_str = f" (since {valid_at.strftime('%b %Y')})"
                lines.append(f"[{i}] {fact}{date_str}{status}")

            return f"{len(edges)} memories found:\n\n" + "\n".join(lines)

        all_tools.append(search_memory)

    return all_tools
