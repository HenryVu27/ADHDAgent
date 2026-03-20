"""Tests for prompt context assembly helpers."""

import pytest

from app.agent.prompts import (
    build_conversation_state,
    build_system_prompt,
    format_goals_and_outcomes,
    format_structured_facts,
)
from app.models.schemas import FamilyProfile, Goal, Outcome


class TestFormatStructuredFacts:

    def test_empty_profile_no_strategies(self):
        result = format_structured_facts(FamilyProfile(), [])
        assert "Not yet gathered" in result

    def test_format_structured_facts_includes_parent_name(self):
        profile = FamilyProfile(parent_name="Sarah", child_name="Alex", child_age="8")
        result = format_structured_facts(profile, [])
        assert "Parent's name: Sarah" in result
        assert "Child's name: Alex" in result

    def test_with_child_name_and_age(self):
        profile = FamilyProfile(child_name="Kai", child_age="7")
        result = format_structured_facts(profile, [])
        assert "Kai" in result
        assert "7" in result

    def test_with_challenge_areas(self):
        profile = FamilyProfile(challenge_areas=["homework", "bedtime"])
        result = format_structured_facts(profile, [])
        assert "homework" in result
        assert "bedtime" in result

    def test_with_active_strategies(self):
        profile = FamilyProfile(child_name="Kai")
        result = format_structured_facts(profile, ["visual timer", "reward chart"])
        assert "visual timer" in result
        assert "reward chart" in result

    def test_with_all_fields(self):
        profile = FamilyProfile(
            child_name="Kai",
            child_age="7",
            diagnosis_status="diagnosed",
            challenge_areas=["homework"],
            attempted_strategies=["timer"],
            good_day_description="Calm morning",
            hardest_situations=["after school"],
        )
        result = format_structured_facts(profile, ["visual timer"])
        assert "Kai" in result
        assert "diagnosed" in result
        assert "Calm morning" in result
        assert "after school" in result
        assert "visual timer" in result

    def test_returns_bullet_list(self):
        profile = FamilyProfile(child_name="Kai", child_age="7")
        result = format_structured_facts(profile, [])
        lines = result.strip().split("\n")
        assert all(line.startswith("- ") for line in lines)


class TestFormatGoalsAndOutcomes:

    def test_no_goals_no_outcomes(self):
        result = format_goals_and_outcomes([], [])
        assert "No goals set yet" in result

    def test_active_goals(self):
        goals = [
            Goal(description="Homework by 6pm"),
            Goal(description="Morning routine"),
        ]
        result = format_goals_and_outcomes(goals, [])
        assert "Homework by 6pm" in result
        assert "Morning routine" in result
        assert "Active goals" in result

    def test_completed_goals(self):
        goals = [
            Goal(description="Homework by 6pm", status="completed"),
        ]
        result = format_goals_and_outcomes(goals, [])
        assert "Completed goals" in result
        assert "Homework by 6pm" in result

    def test_mixed_goals(self):
        goals = [
            Goal(description="Active goal", status="active"),
            Goal(description="Done goal", status="completed"),
        ]
        result = format_goals_and_outcomes(goals, [])
        assert "Active goals" in result
        assert "Completed goals" in result

    def test_outcomes(self):
        outcomes = [
            Outcome(strategy_name="visual timer", signal="positive", detail="worked great"),
            Outcome(strategy_name="reward chart", signal="negative"),
        ]
        result = format_goals_and_outcomes([], outcomes)
        assert "visual timer: positive (worked great)" in result
        assert "reward chart: negative" in result

    def test_only_last_3_outcomes(self):
        outcomes = [
            Outcome(strategy_name=f"strategy_{i}", signal="positive")
            for i in range(5)
        ]
        result = format_goals_and_outcomes([], outcomes)
        assert "strategy_2" in result
        assert "strategy_3" in result
        assert "strategy_4" in result
        assert "strategy_0" not in result
        assert "strategy_1" not in result

    def test_active_goals_capped(self):
        """10 active goals should show only the last 5 with annotation."""
        goals = [Goal(description=f"Goal {i}") for i in range(10)]
        result = format_goals_and_outcomes(goals, [], max_active_goals=5)
        assert "showing 5 of 10" in result
        assert "Goal 5" in result
        assert "Goal 9" in result
        assert "Goal 0" not in result
        assert "Goal 4" not in result

    def test_completed_goals_capped(self):
        """6 completed goals should show only the last 2 with annotation."""
        goals = [Goal(description=f"Done {i}", status="completed") for i in range(6)]
        result = format_goals_and_outcomes(goals, [], max_completed_goals=2)
        assert "showing 2 of 6" in result
        assert "Done 4" in result
        assert "Done 5" in result
        assert "Done 0" not in result

    def test_under_cap_shows_all_without_annotation(self):
        """3 active goals with cap=5 should show all without 'showing' annotation."""
        goals = [Goal(description=f"Goal {i}") for i in range(3)]
        result = format_goals_and_outcomes(goals, [], max_active_goals=5)
        assert "showing" not in result
        assert "Goal 0" in result
        assert "Goal 1" in result
        assert "Goal 2" in result

    def test_outcomes_cap_parameter(self):
        """Custom max_outcomes parameter should be respected."""
        outcomes = [
            Outcome(strategy_name=f"strat_{i}", signal="positive")
            for i in range(6)
        ]
        result = format_goals_and_outcomes([], outcomes, max_outcomes=2)
        assert "strat_4" in result
        assert "strat_5" in result
        assert "strat_0" not in result


class TestBuildSystemPrompt:

    def test_returns_full_prompt(self):
        profile = FamilyProfile(child_name="Kai", child_age="7")
        result = build_system_prompt(
            profile=profile,
            active_strategies=[],
            goals=[],
            outcomes=[],
        )
        assert "ADHD parenting coach" in result
        assert "Kai" in result
        assert "No goals set yet" in result
        assert "beginning of the conversation" in result

    def test_includes_session_summary(self):
        result = build_system_prompt(
            profile=FamilyProfile(),
            active_strategies=[],
            goals=[],
            outcomes=[],
            session_summary="Parent discussed homework challenges.",
        )
        assert "Parent discussed homework challenges." in result

    def test_includes_goals_section(self):
        goals = [Goal(description="Better bedtime routine")]
        result = build_system_prompt(
            profile=FamilyProfile(),
            active_strategies=[],
            goals=goals,
            outcomes=[],
        )
        assert "Better bedtime routine" in result

    def test_section_headers_present(self):
        result = build_system_prompt(
            profile=FamilyProfile(),
            active_strategies=[],
            goals=[],
            outcomes=[],
        )
        assert "## What You Know About This Family" in result
        assert "## Session Summary" in result
        assert "## Goals and Progress" in result

    def test_no_old_session_context_placeholder(self):
        """Verify the old {session_context} placeholder is gone."""
        result = build_system_prompt(
            profile=FamilyProfile(),
            active_strategies=[],
            goals=[],
            outcomes=[],
        )
        assert "{session_context}" not in result

    def test_includes_search_result_guidance(self):
        result = build_system_prompt(
            profile=FamilyProfile(),
            active_strategies=[],
            goals=[],
            outcomes=[],
        )
        assert "## Using Search Results" in result
        assert "Evidence framing" in result
        assert "Cite sources" in result


class TestPromptEnhancements:

    def _build_prompt(self):
        return build_system_prompt(
            profile=FamilyProfile(),
            active_strategies=[],
            goals=[],
            outcomes=[],
        )

    def test_has_xml_structure(self):
        prompt = self._build_prompt()
        for tag in [
            "<role>", "<boundaries>", "<family-context>",
            "<tools>", "<examples>", "<reasoning>", "<response-guide>",
        ]:
            assert tag in prompt, f"Missing XML tag: {tag}"

    def test_has_consequence_awareness(self):
        prompt = self._build_prompt()
        assert "entire response" in prompt
        assert "discarded and replaced" in prompt

    def test_has_personalization_instructions(self):
        prompt = self._build_prompt()
        assert "parent's own language" in prompt

    def test_has_reasoning_checklist(self):
        prompt = self._build_prompt()
        assert "Before responding" in prompt
        assert "staying within scope" in prompt

    def test_has_adaptive_length(self):
        prompt = self._build_prompt()
        assert "1-2 sentences" in prompt
        assert "Keep every response under 200 words" not in prompt

    def test_has_diverse_examples(self):
        prompt = self._build_prompt()
        assert prompt.count("Parent:") >= 5


class TestBuildConversationState:

    def test_basic_output_format(self):
        result = build_conversation_state(turn=3, phase="strategy")
        assert "<conversation_state>" in result
        assert "</conversation_state>" in result
        assert "<turn>3</turn>" in result
        assert "<phase>strategy</phase>" in result

    def test_includes_tool_calls(self):
        result = build_conversation_state(
            turn=5, phase="progress",
            recent_tool_calls=["search_knowledge_base", "update_family_profile"],
        )
        assert "<last_tools>search_knowledge_base, update_family_profile</last_tools>" in result

    def test_includes_focus(self):
        result = build_conversation_state(
            turn=2, phase="intake", active_topic="My child struggles with homework",
        )
        assert "<focus>My child struggles with homework</focus>" in result

    def test_omits_tools_when_none(self):
        result = build_conversation_state(turn=1, phase="intake")
        assert "last_tools" not in result

    def test_omits_focus_when_empty(self):
        result = build_conversation_state(turn=1, phase="intake", active_topic="")
        assert "focus" not in result

    def test_all_fields(self):
        result = build_conversation_state(
            turn=10, phase="progress",
            recent_tool_calls=["track_outcome"],
            active_topic="Timer worked today",
        )
        assert "<turn>10</turn>" in result
        assert "<phase>progress</phase>" in result
        assert "<last_tools>track_outcome</last_tools>" in result
        assert "<focus>Timer worked today</focus>" in result


    def test_conversation_state_includes_datetime(self):
        from datetime import datetime
        result = build_conversation_state(
            turn=3,
            phase="strategy",
            current_datetime=datetime(2026, 2, 22, 15, 30),
        )
        assert "Sunday" in result
        assert "afternoon" in result

    def test_conversation_state_morning(self):
        from datetime import datetime
        result = build_conversation_state(
            turn=1,
            phase="intake",
            current_datetime=datetime(2026, 2, 23, 8, 0),
        )
        assert "Monday" in result
        assert "morning" in result

    def test_conversation_state_evening(self):
        from datetime import datetime
        result = build_conversation_state(
            turn=1,
            phase="intake",
            current_datetime=datetime(2026, 2, 23, 19, 0),
        )
        assert "evening" in result

    def test_conversation_state_without_datetime(self):
        result = build_conversation_state(turn=1, phase="intake")
        assert "datetime" not in result

def test_enhanced_boundaries_in_system_prompt():
    """System prompt should include enhanced boundary instructions."""
    from app.agent.prompts import build_system_prompt
    from app.models.schemas import FamilyProfile

    prompt = build_system_prompt(
        profile=FamilyProfile(),
        active_strategies=[],
        goals=[],
        outcomes=[],
    )
    # New boundary sections
    assert "only available in English" in prompt
    assert "Staying on topic" in prompt
    assert "Content safety" in prompt
    assert "harmful practices" in prompt
    # Existing boundaries still present
    assert "NEVER discuss medication" in prompt
    assert "NEVER make or suggest a diagnosis" in prompt
