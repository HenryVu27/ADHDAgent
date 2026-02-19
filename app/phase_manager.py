"""
Phase Manager — simple Python state machine for conversation phases.

Replaces the ASP/rules engine layer. Handles:
- Phase transitions (intake -> strategy -> progress -> followup)
- Agent routing (phase -> agent name)
- Directive and constraint generation for response prompts
- Progressive profiling from raw message text
"""

import re

from app.models.schemas import (
    ConversationPhase,
    PhaseDecision,
    SessionState,
)

# Valid phase transitions
VALID_TRANSITIONS: dict[ConversationPhase, list[ConversationPhase]] = {
    ConversationPhase.intake: [ConversationPhase.strategy],
    ConversationPhase.strategy: [ConversationPhase.progress],
    ConversationPhase.progress: [ConversationPhase.strategy, ConversationPhase.followup],
    ConversationPhase.followup: [ConversationPhase.strategy, ConversationPhase.progress],
}

# Phase -> default agent
PHASE_TO_AGENT: dict[ConversationPhase, str] = {
    ConversationPhase.intake: "intake",
    ConversationPhase.strategy: "strategy",
    ConversationPhase.progress: "progress",
    ConversationPhase.followup: "strategy",
}

# Keywords for signal detection
_FRUSTRATION_KEYWORDS = {
    "frustrated", "exhausted", "burned out", "burnout", "tried everything",
    "at my wit", "overwhelmed", "can't take", "giving up", "hopeless",
}
_PROGRESS_KEYWORDS = {
    "progress", "update", "working", "tried", "better", "worse",
    "improved", "helped", "goal",
}
_NEW_STRATEGY_KEYWORDS = {
    "different", "alternative", "another", "new strategy", "something else",
    "other approach",
}

# Keywords for profile extraction
_CHALLENGE_KEYWORDS = {
    "homework": "homework",
    "bedtime": "bedtime",
    "meltdown": "emotion",
    "tantrum": "emotion",
    "angry": "emotion",
    "anger": "emotion",
    "emotional": "emotion",
    "focus": "attention",
    "attention": "attention",
    "distract": "attention",
    "hyperactiv": "hyperactivity",
    "impulsiv": "impulsivity",
    "morning": "morning_routine",
    "school": "school",
    "transition": "transitions",
}
_SITUATION_KEYWORDS = {"homework", "bedtime", "morning", "school", "mealtime", "transition"}


class PhaseManager:
    """Simple phase state machine and agent router."""

    def decide(self, message: str, state: SessionState) -> PhaseDecision:
        """Determine phase, route to agent, and generate directives/constraints."""
        message_lower = message.lower()
        directives: list[str] = []
        constraints: list[str] = []
        phase = state.phase
        agent = PHASE_TO_AGENT.get(phase, "strategy")
        phase_changed = False

        # Always empathetic
        constraints.append("tone_empathetic")
        if self._parent_is_frustrated(message_lower):
            constraints.append("must_acknowledge_frustration")

        # Phase-specific logic
        if phase == ConversationPhase.intake:
            directives.append("gather_info")
            if self.check_intake_complete(state):
                state.phase = ConversationPhase.strategy
                phase = ConversationPhase.strategy
                agent = "strategy"
                directives.append("transition_to_strategy")
                phase_changed = True

        elif phase == ConversationPhase.strategy:
            directives.append("recommend_strategy")
            constraints.append("must_have_action_steps")
            if self._mentions_progress(message_lower):
                if self._can_transition(state, ConversationPhase.progress):
                    state.phase = ConversationPhase.progress
                    phase = ConversationPhase.progress
                    agent = "progress"
                    directives.append("transition_to_progress")
                    phase_changed = True

        elif phase == ConversationPhase.progress:
            directives.append("track_progress")
            if self._wants_new_strategy(message_lower):
                state.phase = ConversationPhase.strategy
                phase = ConversationPhase.strategy
                agent = "strategy"
                directives.append("transition_to_strategy")
                phase_changed = True

        elif phase == ConversationPhase.followup:
            directives.append("check_in")

        return PhaseDecision(
            agent=agent,
            phase=phase,
            directives=directives,
            constraints=constraints,
            phase_changed=phase_changed,
        )

    def update_profile(self, message: str, state: SessionState) -> None:
        """Progressive profiling: extract age, challenges, situations from raw message."""
        message_lower = message.lower()

        # Age extraction (only if not already known)
        if not state.family_profile.child_age:
            age_match = re.search(r"(\d{1,2})\s*(?:year|yr|-year)", message_lower)
            if age_match:
                state.family_profile.child_age = age_match.group(1)

        # Challenge extraction
        for keyword, category in _CHALLENGE_KEYWORDS.items():
            if keyword in message_lower:
                if category not in state.family_profile.challenge_areas:
                    state.family_profile.challenge_areas.append(category)

        # Situation extraction
        for keyword in _SITUATION_KEYWORDS:
            if keyword in message_lower:
                if keyword not in state.family_profile.hardest_situations:
                    state.family_profile.hardest_situations.append(keyword)

    def check_intake_complete(self, state: SessionState) -> bool:
        """Check if intake has gathered enough information to proceed."""
        profile = state.family_profile
        return bool(
            profile.child_age
            and profile.challenge_areas
            and profile.attempted_strategies
        )

    def check_strategy_ready(self, state: SessionState) -> bool:
        """Check if there are active strategies for progress tracking."""
        return len(state.active_strategies) > 0

    def _can_transition(self, state: SessionState, target: ConversationPhase) -> bool:
        """Check if a transition to target phase is valid."""
        allowed = VALID_TRANSITIONS.get(state.phase, [])
        if target not in allowed:
            return False
        if state.phase == ConversationPhase.strategy and target == ConversationPhase.progress:
            return self.check_strategy_ready(state)
        return True

    def _parent_is_frustrated(self, message_lower: str) -> bool:
        return any(kw in message_lower for kw in _FRUSTRATION_KEYWORDS)

    def _mentions_progress(self, message_lower: str) -> bool:
        return any(kw in message_lower for kw in _PROGRESS_KEYWORDS)

    def _wants_new_strategy(self, message_lower: str) -> bool:
        return any(kw in message_lower for kw in _NEW_STRATEGY_KEYWORDS)
