"""Tests for SessionStateStore — profile, outcomes, goals, seeding."""

import pytest

from app.agent.session_store import SessionStateStore
from app.models.schemas import SeedSessionRequest


class TestSessionStateStore:

    def setup_method(self):
        self.store = SessionStateStore()

    def test_get_creates_new_session(self):
        state = self.store.get("new_session")
        assert state.session_id == "new_session"
        assert state.turn_count == 0
        assert state.family_profile.child_name is None

    def test_get_returns_same_session(self):
        s1 = self.store.get("s1")
        s1.turn_count = 5
        s2 = self.store.get("s1")
        assert s2.turn_count == 5

    def test_increment_turn(self):
        assert self.store.increment_turn("t1") == 1
        assert self.store.increment_turn("t1") == 2
        assert self.store.increment_turn("t1") == 3

    def test_update_profile_scalar_fields(self):
        profile = self.store.update_profile("p1", child_name="Kai", child_age="7")
        assert profile.child_name == "Kai"
        assert profile.child_age == "7"

    def test_update_profile_list_append(self):
        self.store.update_profile("p1", challenge_areas=["homework"])
        profile = self.store.update_profile("p1", challenge_areas=["bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    def test_update_profile_list_dedup(self):
        self.store.update_profile("p1", challenge_areas=["homework"])
        profile = self.store.update_profile("p1", challenge_areas=["homework", "bedtime"])
        assert profile.challenge_areas == ["homework", "bedtime"]

    def test_update_profile_ignores_none(self):
        self.store.update_profile("p1", child_name="Kai")
        profile = self.store.update_profile("p1", child_name=None, child_age="8")
        assert profile.child_name == "Kai"
        assert profile.child_age == "8"

    def test_add_outcome(self):
        outcome = self.store.add_outcome("o1", "visual timer", "positive", "worked great")
        assert outcome.goal_description == "visual timer"
        assert outcome.signal == "positive"
        assert outcome.detail == "worked great"

        state = self.store.get("o1")
        assert len(state.outcomes) == 1

    def test_manage_goal_add(self):
        goals = self.store.manage_goal("g1", "add", "Homework done by 6pm")
        assert len(goals) == 1
        assert goals[0].description == "Homework done by 6pm"
        assert goals[0].status == "active"

    def test_manage_goal_complete(self):
        self.store.manage_goal("g1", "add", "Homework done by 6pm")
        goals = self.store.manage_goal("g1", "complete", "Homework done by 6pm")
        assert goals[0].status == "completed"

    def test_manage_goal_complete_case_insensitive(self):
        self.store.manage_goal("g1", "add", "Homework done by 6pm")
        goals = self.store.manage_goal("g1", "complete", "homework done by 6pm")
        assert goals[0].status == "completed"

    def test_manage_goal_list(self):
        self.store.manage_goal("g1", "add", "Goal A")
        self.store.manage_goal("g1", "add", "Goal B")
        goals = self.store.manage_goal("g1", "list")
        assert len(goals) == 2

    def test_seed_session(self):
        request = SeedSessionRequest(
            session_id="seed1",
            child_name="Kai",
            child_age="8",
            challenges=["homework", "bedtime"],
            tried_strategies=["timer"],
            goals=["Homework by 6pm"],
        )
        self.store.seed_session(request)

        state = self.store.get("seed1")
        assert state.family_profile.child_name == "Kai"
        assert state.family_profile.child_age == "8"
        assert state.family_profile.challenge_areas == ["homework", "bedtime"]
        assert state.family_profile.attempted_strategies == ["timer"]
        assert len(state.goals) == 1
        assert state.goals[0].description == "Homework by 6pm"

    def test_sessions_isolated(self):
        self.store.update_profile("s1", child_name="Kai")
        self.store.update_profile("s2", child_name="Alex")
        assert self.store.get("s1").family_profile.child_name == "Kai"
        assert self.store.get("s2").family_profile.child_name == "Alex"
