"""Tests for prompt context assembly helpers."""

import pytest

from app.agent.prompts import (
    build_system_prompt,
    format_goals_and_outcomes,
    format_structured_facts,
)
from app.models.schemas import FamilyProfile, Goal, Outcome


class TestFormatStructuredFacts:

    def test_empty_profile_no_strategies(self):
        result = format_structured_facts(FamilyProfile(), [])
        assert "Not yet gathered" in result

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
            Outcome(goal_description="visual timer", signal="positive", detail="worked great"),
            Outcome(goal_description="reward chart", signal="negative"),
        ]
        result = format_goals_and_outcomes([], outcomes)
        assert "visual timer: positive (worked great)" in result
        assert "reward chart: negative" in result

    def test_only_last_3_outcomes(self):
        outcomes = [
            Outcome(goal_description=f"strategy_{i}", signal="positive")
            for i in range(5)
        ]
        result = format_goals_and_outcomes([], outcomes)
        assert "strategy_2" in result
        assert "strategy_3" in result
        assert "strategy_4" in result
        assert "strategy_0" not in result
        assert "strategy_1" not in result


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
