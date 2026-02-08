"""
ASP (Answer Set Programming) Reasoning Engine.

This module interfaces with the Clingo ASP solver to determine valid
conversation moves based on:
- Current conversation state
- Extracted predicates from the parent's message
- Clinician-defined rules and safety constraints

The ASP program encodes:
1. What topics are valid to discuss given the conversation state
2. What transitions are allowed between conversation phases
3. What safety constraints must hold at all times
"""

from pathlib import Path

RULES_DIR = Path(__file__).parent / "rules"


class ASPEngine:
    """Interface to the ASP solver for conversation reasoning."""

    def __init__(self):
        self.rules = self._load_rules()
        self.session_facts: dict[str, list[str]] = {}

    def _load_rules(self) -> dict[str, str]:
        """Load ASP rule files from the rules directory."""
        rules = {}
        for rule_file in RULES_DIR.glob("*.lp"):
            rules[rule_file.stem] = rule_file.read_text()
        return rules

    def reason(self, predicates: list[dict], session_id: str) -> list[str]:
        """
        Given extracted predicates and conversation state, use ASP to
        determine valid conversation moves.

        For now, this uses rule-based logic as a foundation.
        The full ASP solver (Clingo/s(CASP)) integration will replace this
        with proper answer set computation.

        Args:
            predicates: Structured predicates extracted from parent message
            session_id: Current session identifier

        Returns:
            List of ASP-derived directives (e.g., ["recommend_strategy", "track_progress"])
        """
        # Initialize session if new
        if session_id not in self.session_facts:
            self.session_facts[session_id] = ["phase(intake)"]

        facts = self.session_facts[session_id]
        directives = []

        # Convert predicates to ASP-style facts
        asp_facts = self._predicates_to_facts(predicates)

        # Determine directives based on current state and new facts
        if "phase(intake)" in facts:
            directives.append("gather_info")
            # Check if we have enough info to transition
            if self._has_sufficient_context(facts):
                directives.append("transition(intake, strategy)")
                facts.remove("phase(intake)")
                facts.append("phase(strategy)")

        elif "phase(strategy)" in facts:
            # Check what the parent is asking about
            if any("progress" in f or "goal" in f or "update" in f for f in asp_facts):
                directives.append("track_progress")
            else:
                directives.append("recommend_strategy")

        # Safety constraints always apply
        for fact in asp_facts:
            if "crisis" in fact or "harm" in fact:
                directives = ["safety_redirect"]
                break

        # Store updated facts
        self.session_facts[session_id] = facts

        return directives if directives else ["gather_info"]

    def _predicates_to_facts(self, predicates: list[dict]) -> list[str]:
        """Convert extracted predicates to ASP-compatible fact strings."""
        facts = []
        for pred in predicates:
            predicate_name = pred.get("predicate", "unknown")
            subject = pred.get("subject", "")
            category = pred.get("category", "")
            facts.append(f"{predicate_name}({subject},{category})")
        return facts

    def _has_sufficient_context(self, facts: list[str]) -> bool:
        """Check if we have enough context to move past intake."""
        context_facts = [f for f in facts if f.startswith("context(")]
        return len(context_facts) >= 3

    def add_fact(self, session_id: str, fact: str):
        """Add a fact to the session's knowledge base."""
        if session_id not in self.session_facts:
            self.session_facts[session_id] = []
        self.session_facts[session_id].append(fact)

    def get_facts(self, session_id: str) -> list[str]:
        """Get all facts for a session."""
        return self.session_facts.get(session_id, [])
