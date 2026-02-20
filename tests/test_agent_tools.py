"""Tests for the 5 ReAct agent tools."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.session_store import SessionStateStore
from app.agent.tools import (
    create_tools,
    get_family_profile,
    manage_goals,
    search_knowledge_base,
    track_outcome,
    update_family_profile,
)
from app.models.schemas import RetrievalResponse, RetrievalResult


@pytest.fixture
def session_store():
    return SessionStateStore()


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
            ),
            RetrievalResult(
                document_id="doc2",
                document_name="Homework Routine",
                content="Establish a consistent homework routine...",
                score=0.8,
                tags=["homework"],
            ),
        ],
    ))
    return retriever


@pytest.fixture
def tools(mock_retriever, session_store):
    return create_tools(retriever=mock_retriever, session_store=session_store)


def _config(session_id="test"):
    return {"configurable": {"session_id": session_id}}


class TestSearchKnowledgeBase:

    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, tools, mock_retriever):
        result = await search_knowledge_base.ainvoke(
            {"query": "homework strategies"},
            config=_config(),
        )
        assert "Visual Timer Strategy" in result
        assert "Homework Routine" in result
        assert "(Source: APA Guidelines)" in result

    @pytest.mark.asyncio
    async def test_returns_no_results_message(self, tools):
        from app.agent import tools as tools_module
        tools_module._retriever.retrieve = AsyncMock(
            return_value=RetrievalResponse(results=[])
        )
        result = await search_knowledge_base.ainvoke(
            {"query": "quantum physics"},
            config=_config(),
        )
        assert "No relevant documents found" in result

    @pytest.mark.asyncio
    async def test_passes_filters(self, tools, mock_retriever):
        await search_knowledge_base.ainvoke(
            {"query": "homework", "document_type": "strategy", "tags": ["homework"]},
            config=_config(),
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["filters"] is not None


class TestGetFamilyProfile:

    def test_empty_profile(self, tools):
        result = get_family_profile.invoke({}, config=_config("empty_session"))
        assert "No family profile information yet" in result

    def test_populated_profile(self, tools, session_store):
        session_store.update_profile("prof1", child_name="Kai", child_age="7")
        session_store.update_profile("prof1", challenge_areas=["homework"])
        result = get_family_profile.invoke({}, config=_config("prof1"))
        assert "Kai" in result
        assert "7" in result
        assert "homework" in result

    def test_includes_goals(self, tools, session_store):
        session_store.update_profile("goal_prof", child_name="Kai")
        session_store.manage_goal("goal_prof", "add", "Homework by 6pm")
        result = get_family_profile.invoke({}, config=_config("goal_prof"))
        assert "Homework by 6pm" in result


class TestUpdateFamilyProfile:

    def test_updates_fields(self, tools, session_store):
        result = update_family_profile.invoke(
            {"child_name": "Kai", "child_age": "7"},
            config=_config("update1"),
        )
        assert "Profile updated" in result
        assert "child_name" in result

        profile = session_store.get("update1").family_profile
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    def test_no_updates(self, tools):
        result = update_family_profile.invoke({}, config=_config("noop"))
        assert "No updates provided" in result


class TestTrackOutcome:

    def test_tracks_positive(self, tools, session_store):
        result = track_outcome.invoke(
            {"strategy_name": "visual timer", "outcome": "positive", "notes": "worked well"},
            config=_config("outcome1"),
        )
        assert "positive" in result
        state = session_store.get("outcome1")
        assert len(state.outcomes) == 1
        assert "visual timer" in state.active_strategies

    def test_tracks_negative(self, tools, session_store):
        result = track_outcome.invoke(
            {"strategy_name": "reward chart", "outcome": "negative"},
            config=_config("outcome2"),
        )
        assert "negative" in result
        state = session_store.get("outcome2")
        assert len(state.outcomes) == 1
        assert "reward chart" not in state.active_strategies

    def test_invalid_outcome(self, tools):
        result = track_outcome.invoke(
            {"strategy_name": "timer", "outcome": "maybe"},
            config=_config("bad"),
        )
        assert "Invalid outcome" in result


class TestManageGoals:

    def test_add_goal(self, tools, session_store):
        result = manage_goals.invoke(
            {"action": "add", "description": "Homework done by 6pm"},
            config=_config("goals1"),
        )
        assert "Homework done by 6pm" in result
        assert "[active]" in result

    def test_complete_goal(self, tools, session_store):
        manage_goals.invoke(
            {"action": "add", "description": "Homework done by 6pm"},
            config=_config("goals2"),
        )
        result = manage_goals.invoke(
            {"action": "complete", "description": "Homework done by 6pm"},
            config=_config("goals2"),
        )
        assert "[done]" in result

    def test_list_goals(self, tools, session_store):
        manage_goals.invoke({"action": "add", "description": "Goal A"}, config=_config("goals3"))
        manage_goals.invoke({"action": "add", "description": "Goal B"}, config=_config("goals3"))
        result = manage_goals.invoke({"action": "list"}, config=_config("goals3"))
        assert "2 total" in result

    def test_invalid_action(self, tools):
        result = manage_goals.invoke(
            {"action": "delete", "description": "test"},
            config=_config("bad"),
        )
        assert "Invalid action" in result

    def test_add_without_description(self, tools):
        result = manage_goals.invoke({"action": "add"}, config=_config("bad2"))
        assert "Description required" in result
