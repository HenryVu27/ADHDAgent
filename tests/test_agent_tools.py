"""Tests for the 7 ReAct agent tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.session_store import create_in_memory_store
from app.agent.tools import (
    _age_to_range,
    _format_result,
    _format_summary,
    create_tools,
)
from app.models.schemas import RetrievalResponse, RetrievalResult


@pytest.fixture
async def session_store():
    return await create_in_memory_store()


_MOCK_DOCS = {
    "doc1": {
        "id": "doc1",
        "name": "Visual Timer Strategy",
        "description": "Use a visual timer to help your child manage time during homework.",
        "document_type": "strategy",
        "tags": ["homework", "executive_function"],
        "age_range": ["school_age"],
        "evidence_level": "strong",
        "source": "APA Guidelines",
        "citations": [{"source_name": "APA Behavior Guide"}],
        "related_ids": ["doc2"],
        "steps": [
            "Set a timer for 10-15 minutes of focused work",
            "Take a 5-minute movement break",
            "Repeat until homework is complete",
        ],
    },
    "doc2": {
        "id": "doc2",
        "name": "Homework Routine",
        "description": "Establish a consistent homework routine for children with ADHD.",
        "document_type": "guidance",
        "tags": ["homework"],
        "age_range": [],
        "evidence_level": "moderate",
        "source": "",
        "citations": [],
        "related_ids": [],
        "key_points": [
            "Designate a quiet, distraction-free workspace",
            "Use consistent timing each day",
        ],
    },
}


@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=RetrievalResponse(
        results=[
            RetrievalResult(
                document_id="doc1",
                document_name="Visual Timer Strategy",
                content="Use a visual timer to help your child...",
                score=0.9,
                source="APA Guidelines",
                tags=["homework", "executive_function"],
                evidence_level="strong",
                document_type="strategy",
                age_range=["school_age"],
                citations=[{"source_name": "APA Behavior Guide"}],
                full_doc=_MOCK_DOCS["doc1"],
            ),
            RetrievalResult(
                document_id="doc2",
                document_name="Homework Routine",
                content="Establish a consistent homework routine...",
                score=0.8,
                tags=["homework"],
                evidence_level="moderate",
                document_type="guidance",
                full_doc=_MOCK_DOCS["doc2"],
            ),
        ],
    ))
    # Mock _store for get_document_details and get_related_documents tools
    mock_store = MagicMock()
    mock_store.get_document_by_id = lambda doc_id: _MOCK_DOCS.get(doc_id)
    mock_store.get_related_docs = lambda doc_id: (
        [_MOCK_DOCS[rid] for rid in _MOCK_DOCS.get(doc_id, {}).get("related_ids", []) if rid in _MOCK_DOCS]
    )
    retriever._store = mock_store
    return retriever


@pytest.fixture
async def tools(mock_retriever, session_store):
    return create_tools(retriever=mock_retriever, session_store=session_store)


@pytest.fixture
async def tool_set(mock_retriever, session_store):
    """Returns a dict mapping tool name -> tool callable."""
    tools_list = create_tools(retriever=mock_retriever, session_store=session_store)
    return {t.name: t for t in tools_list}


def _config(session_id="test"):
    return {"configurable": {"session_id": session_id}}


class TestSearchKnowledgeBase:

    async def test_returns_summary_results(self, tool_set, mock_retriever):
        result = await tool_set["search_knowledge_base"].ainvoke(
            {"query": "homework strategies"},
            config=_config(),
        )
        # Quality summary line
        assert "2 results" in result
        assert "top: 0.90" in result
        assert "lowest: 0.80" in result
        # Document names present
        assert "Visual Timer Strategy" in result
        assert "Homework Routine" in result
        # Score in summary
        assert "score: 0.90" in result
        # Summary metadata
        assert "Evidence: strong" in result
        assert "ID: doc1" in result
        assert "Tags: homework, executive_function" in result
        # Summaries should NOT contain full steps/key_points
        assert "Steps:" not in result
        assert "Key points:" not in result

    async def test_returns_no_results_message(self, session_store):
        empty_retriever = AsyncMock()
        empty_retriever.retrieve = AsyncMock(
            return_value=RetrievalResponse(results=[])
        )
        ts = {t.name: t for t in create_tools(retriever=empty_retriever, session_store=session_store)}
        result = await ts["search_knowledge_base"].ainvoke(
            {"query": "quantum physics"},
            config=_config(),
        )
        assert "No relevant documents found" in result

    async def test_passes_filters_and_skip_rewrite(self, tool_set, mock_retriever):
        await tool_set["search_knowledge_base"].ainvoke(
            {"query": "homework", "document_type": "strategy", "tags": ["homework"]},
            config=_config(),
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["filters"] is not None
        assert call_args.kwargs["skip_rewrite"] is True

    async def test_auto_applies_age_filter(self, tool_set, mock_retriever, session_store):
        await session_store.update_profile("age_test", child_age="7")
        await session_store.commit()
        await tool_set["search_knowledge_base"].ainvoke(
            {"query": "homework help"},
            config=_config("age_test"),
        )
        call_args = mock_retriever.retrieve.call_args
        filters = call_args.kwargs["filters"]
        assert filters is not None
        assert filters.age_range == "school_age"

    async def test_explicit_age_overrides_profile(self, tool_set, mock_retriever, session_store):
        await session_store.update_profile("age_test2", child_age="4")
        await session_store.commit()
        await tool_set["search_knowledge_base"].ainvoke(
            {"query": "strategies", "age_range": "adolescent"},
            config=_config("age_test2"),
        )
        call_args = mock_retriever.retrieve.call_args
        filters = call_args.kwargs["filters"]
        assert filters.age_range == "adolescent"

    async def test_no_age_filter_without_profile(self, tool_set, mock_retriever):
        await tool_set["search_knowledge_base"].ainvoke(
            {"query": "strategies"},
            config=_config("no_age"),
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["filters"] is None


class TestGetFamilyProfile:

    async def test_empty_profile(self, tool_set):
        result = await tool_set["get_family_profile"].ainvoke({}, config=_config("empty_session"))
        assert "No family profile information yet" in result

    async def test_populated_profile(self, tool_set, session_store):
        await session_store.update_profile("prof1", child_name="Kai", child_age="7")
        await session_store.update_profile("prof1", challenge_areas=["homework"])
        await session_store.commit()
        result = await tool_set["get_family_profile"].ainvoke({}, config=_config("prof1"))
        assert "Kai" in result
        assert "7" in result
        assert "homework" in result

    async def test_includes_goals(self, tool_set, session_store):
        await session_store.update_profile("goal_prof", child_name="Kai")
        await session_store.manage_goal("goal_prof", "add", "Homework by 6pm")
        await session_store.commit()
        result = await tool_set["get_family_profile"].ainvoke({}, config=_config("goal_prof"))
        assert "Homework by 6pm" in result


class TestUpdateFamilyProfile:

    async def test_updates_fields(self, tool_set, session_store):
        result = await tool_set["update_family_profile"].ainvoke(
            {"child_name": "Kai", "child_age": "7"},
            config=_config("update1"),
        )
        assert "Profile updated" in result
        assert "child_name" in result

        await session_store.commit()
        profile = (await session_store.get("update1")).family_profile
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    async def test_no_updates(self, tool_set):
        result = await tool_set["update_family_profile"].ainvoke({}, config=_config("noop"))
        assert "No updates provided" in result


class TestTrackOutcome:

    async def test_tracks_positive(self, tool_set, session_store):
        result = await tool_set["track_outcome"].ainvoke(
            {"strategy_name": "visual timer", "outcome": "positive", "notes": "worked well"},
            config=_config("outcome1"),
        )
        assert "positive" in result
        await session_store.commit()
        state = await session_store.get("outcome1")
        assert len(state.outcomes) == 1
        assert "visual timer" in state.active_strategies

    async def test_tracks_negative(self, tool_set, session_store):
        result = await tool_set["track_outcome"].ainvoke(
            {"strategy_name": "reward chart", "outcome": "negative"},
            config=_config("outcome2"),
        )
        assert "negative" in result
        await session_store.commit()
        state = await session_store.get("outcome2")
        assert len(state.outcomes) == 1
        assert "reward chart" not in state.active_strategies

    async def test_invalid_outcome(self, tool_set):
        result = await tool_set["track_outcome"].ainvoke(
            {"strategy_name": "timer", "outcome": "maybe"},
            config=_config("bad"),
        )
        assert "Invalid outcome" in result


class TestManageGoals:

    async def test_add_goal(self, tool_set, session_store):
        result = await tool_set["manage_goals"].ainvoke(
            {"action": "add", "description": "Homework done by 6pm"},
            config=_config("goals1"),
        )
        assert "Homework done by 6pm" in result
        assert "[active]" in result

    async def test_complete_goal(self, tool_set, session_store):
        await tool_set["manage_goals"].ainvoke(
            {"action": "add", "description": "Homework done by 6pm"},
            config=_config("goals2"),
        )
        result = await tool_set["manage_goals"].ainvoke(
            {"action": "complete", "description": "Homework done by 6pm"},
            config=_config("goals2"),
        )
        assert "[done]" in result

    async def test_list_goals(self, tool_set, session_store):
        await tool_set["manage_goals"].ainvoke({"action": "add", "description": "Goal A"}, config=_config("goals3"))
        await tool_set["manage_goals"].ainvoke({"action": "add", "description": "Goal B"}, config=_config("goals3"))
        result = await tool_set["manage_goals"].ainvoke({"action": "list"}, config=_config("goals3"))
        assert "2 total" in result

    async def test_invalid_action(self, tool_set):
        result = await tool_set["manage_goals"].ainvoke(
            {"action": "delete", "description": "test"},
            config=_config("bad"),
        )
        assert "Invalid action" in result

    async def test_add_without_description(self, tool_set):
        result = await tool_set["manage_goals"].ainvoke({"action": "add"}, config=_config("bad2"))
        assert "Description required" in result


class TestAgeToRange:

    def test_preschool(self):
        assert _age_to_range("3") == "preschool"
        assert _age_to_range("5") == "preschool"

    def test_school_age(self):
        assert _age_to_range("6") == "school_age"
        assert _age_to_range("8") == "school_age"
        assert _age_to_range("12") == "school_age"

    def test_adolescent(self):
        assert _age_to_range("13") == "adolescent"
        assert _age_to_range("16") == "adolescent"

    def test_invalid(self):
        assert _age_to_range("unknown") is None
        assert _age_to_range("") is None
        assert _age_to_range(None) is None


class TestFormatResult:

    def test_strategy_with_steps(self):
        result = RetrievalResult(
            document_id="s1",
            document_name="Timer Strategy",
            content="...",
            score=0.9,
            source="CDC",
            tags=["homework", "focus"],
            evidence_level="strong",
            age_range=["school_age"],
            citations=[{"source_name": "CDC Guidelines"}],
            full_doc={
                "description": "A timer-based approach.",
                "steps": ["Step one", "Step two"],
            },
        )
        formatted = _format_result(1, result)
        assert "[1] Timer Strategy" in formatted
        assert "score: 0.90" in formatted
        assert "Evidence: strong" in formatted
        assert "Source: CDC" in formatted
        assert "Tags: homework, focus" in formatted
        assert "ID: s1" in formatted
        assert "A timer-based approach." in formatted
        assert "Steps:" in formatted
        assert "1. Step one" in formatted
        assert "2. Step two" in formatted
        assert "CDC Guidelines" in formatted

    def test_guidance_with_key_points(self):
        result = RetrievalResult(
            document_id="g1",
            document_name="Sleep Hygiene",
            content="...",
            score=0.8,
            evidence_level="moderate",
            document_type="guidance",
            full_doc={
                "description": "Good sleep habits matter.",
                "key_points": ["Consistent bedtime", "No screens before bed"],
            },
        )
        formatted = _format_result(1, result)
        assert "Key points:" in formatted
        assert "- Consistent bedtime" in formatted
        assert "- No screens before bed" in formatted

    def test_minimal_result_no_full_doc(self):
        result = RetrievalResult(
            document_id="m1",
            document_name="Basic Doc",
            content="...",
            score=0.5,
        )
        formatted = _format_result(1, result)
        assert "[1] Basic Doc" in formatted


class TestFormatSummary:

    def test_summary_includes_score_and_metadata(self):
        result = RetrievalResult(
            document_id="s1",
            document_name="Timer Strategy",
            content="...",
            score=0.85,
            tags=["homework", "focus"],
            evidence_level="strong",
            age_range=["school_age"],
            full_doc={
                "description": "A timer-based approach for managing homework time effectively.",
            },
        )
        formatted = _format_summary(1, result)
        assert "[1] Timer Strategy  (score: 0.85)" in formatted
        assert "Evidence: strong" in formatted
        assert "Ages: school_age" in formatted
        assert "Tags: homework, focus" in formatted
        assert "ID: s1" in formatted
        assert "A timer-based approach" in formatted

    def test_summary_truncates_long_description(self):
        result = RetrievalResult(
            document_id="long1",
            document_name="Long Doc",
            content="...",
            score=0.7,
            full_doc={
                "description": "A" * 200,
            },
        )
        formatted = _format_summary(1, result)
        assert "..." in formatted

    def test_summary_does_not_include_steps(self):
        result = RetrievalResult(
            document_id="s1",
            document_name="Strategy",
            content="...",
            score=0.9,
            full_doc={
                "description": "Desc.",
                "steps": ["Step one", "Step two"],
            },
        )
        formatted = _format_summary(1, result)
        assert "Steps:" not in formatted
        assert "Step one" not in formatted


class TestGetDocumentDetails:

    async def test_returns_full_document(self, tool_set, mock_retriever):
        result = await tool_set["get_document_details"].ainvoke(
            {"document_id": "doc1"},
            config=_config(),
        )
        assert "Visual Timer Strategy" in result
        assert "Evidence: strong" in result
        assert "Source: APA Guidelines" in result
        assert "Steps:" in result
        assert "Set a timer for 10-15 minutes" in result
        assert "Take a 5-minute movement break" in result
        assert "APA Behavior Guide" in result

    async def test_returns_related_documents_hint(self, tool_set, mock_retriever):
        result = await tool_set["get_document_details"].ainvoke(
            {"document_id": "doc1"},
            config=_config(),
        )
        assert "Related documents:" in result
        assert "Homework Routine" in result

    async def test_not_found(self, tool_set, mock_retriever):
        result = await tool_set["get_document_details"].ainvoke(
            {"document_id": "nonexistent"},
            config=_config(),
        )
        assert "not found" in result

    async def test_document_with_key_points(self, tool_set, mock_retriever):
        result = await tool_set["get_document_details"].ainvoke(
            {"document_id": "doc2"},
            config=_config(),
        )
        assert "Homework Routine" in result
        assert "Key points:" in result
        assert "Designate a quiet, distraction-free workspace" in result


class TestGetRelatedDocuments:

    async def test_returns_related_summaries(self, tool_set, mock_retriever):
        result = await tool_set["get_related_documents"].ainvoke(
            {"document_id": "doc1"},
            config=_config(),
        )
        assert '1 related documents for "Visual Timer Strategy"' in result
        assert "Homework Routine" in result
        assert "ID: doc2" in result

    async def test_no_related_docs(self, tool_set, mock_retriever):
        result = await tool_set["get_related_documents"].ainvoke(
            {"document_id": "doc2"},
            config=_config(),
        )
        assert "No related documents found" in result
        assert "Homework Routine" in result

    async def test_not_found(self, tool_set, mock_retriever):
        result = await tool_set["get_related_documents"].ainvoke(
            {"document_id": "nonexistent"},
            config=_config(),
        )
        assert "not found" in result


class TestCreateTools:

    def test_returns_7_tools(self, tools):
        assert len(tools) == 7


class TestToolErrorHandling:
    """Verify every tool returns a structured error, not a raw traceback."""

    async def test_search_returns_error_on_retriever_failure(self, session_store):
        failing_retriever = AsyncMock()
        failing_retriever.retrieve = AsyncMock(side_effect=RuntimeError("connection refused"))
        ts = {t.name: t for t in create_tools(retriever=failing_retriever, session_store=session_store)}

        result = await ts["search_knowledge_base"].ainvoke(
            {"query": "homework"}, config=_config(),
        )
        assert "Error:" in result
        assert "RuntimeError" in result
        assert "Traceback" not in result

    async def test_get_details_returns_error_on_store_failure(self, session_store):
        failing_retriever = AsyncMock()
        failing_retriever._store = MagicMock()
        failing_retriever._store.get_document_by_id = MagicMock(side_effect=RuntimeError("disk full"))
        ts = {t.name: t for t in create_tools(retriever=failing_retriever, session_store=session_store)}

        result = await ts["get_document_details"].ainvoke(
            {"document_id": "doc1"}, config=_config(),
        )
        assert "Error:" in result
        assert "Traceback" not in result

    async def test_get_related_returns_error_on_store_failure(self, session_store):
        failing_retriever = AsyncMock()
        failing_retriever._store = MagicMock()
        failing_retriever._store.get_document_by_id = MagicMock(side_effect=RuntimeError("disk full"))
        ts = {t.name: t for t in create_tools(retriever=failing_retriever, session_store=session_store)}

        result = await ts["get_related_documents"].ainvoke(
            {"document_id": "doc1"}, config=_config(),
        )
        assert "Error:" in result
        assert "Traceback" not in result

    async def test_get_profile_returns_error_on_store_failure(self):
        failing_store = AsyncMock()
        failing_store.get = AsyncMock(side_effect=RuntimeError("db locked"))
        ts = {t.name: t for t in create_tools(retriever=AsyncMock(), session_store=failing_store)}

        result = await ts["get_family_profile"].ainvoke({}, config=_config())
        assert "Error:" in result
        assert "Traceback" not in result

    async def test_update_profile_returns_error_on_store_failure(self):
        failing_store = AsyncMock()
        failing_store.update_profile = AsyncMock(side_effect=RuntimeError("db locked"))
        ts = {t.name: t for t in create_tools(retriever=AsyncMock(), session_store=failing_store)}

        result = await ts["update_family_profile"].ainvoke(
            {"child_name": "Kai"}, config=_config(),
        )
        assert "Error:" in result
        assert "Traceback" not in result

    async def test_track_outcome_returns_error_on_store_failure(self):
        failing_store = AsyncMock()
        failing_store.add_outcome = AsyncMock(side_effect=RuntimeError("db locked"))
        ts = {t.name: t for t in create_tools(retriever=AsyncMock(), session_store=failing_store)}

        result = await ts["track_outcome"].ainvoke(
            {"strategy_name": "timer", "outcome": "positive"}, config=_config(),
        )
        assert "Error:" in result
        assert "Traceback" not in result

    async def test_manage_goals_returns_error_on_store_failure(self):
        failing_store = AsyncMock()
        failing_store.manage_goal = AsyncMock(side_effect=RuntimeError("db locked"))
        ts = {t.name: t for t in create_tools(retriever=AsyncMock(), session_store=failing_store)}

        result = await ts["manage_goals"].ainvoke(
            {"action": "add", "description": "test goal"}, config=_config(),
        )
        assert "Error:" in result
        assert "Traceback" not in result

    async def test_independent_tool_sets(self, session_store):
        """Two create_tools() calls produce independent tool sets."""
        retriever_a = AsyncMock()
        retriever_a.retrieve = AsyncMock(return_value=RetrievalResponse(results=[]))
        retriever_b = AsyncMock()
        retriever_b.retrieve = AsyncMock(return_value=RetrievalResponse(results=[]))

        tools_a = create_tools(retriever=retriever_a, session_store=session_store)
        tools_b = create_tools(retriever=retriever_b, session_store=session_store)

        search_a = {t.name: t for t in tools_a}["search_knowledge_base"]
        search_b = {t.name: t for t in tools_b}["search_knowledge_base"]

        await search_a.ainvoke({"query": "test"}, config=_config())
        await search_b.ainvoke({"query": "test"}, config=_config())

        # Each retriever was called exactly once — no cross-contamination
        retriever_a.retrieve.assert_called_once()
        retriever_b.retrieve.assert_called_once()
