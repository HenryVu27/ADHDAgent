from app.agents.intake import IntakeAgent
from app.agents.strategy import StrategyAgent
from app.agents.progress import ProgressAgent
from app.agents.safety import SafetyMonitor


class AgentOrchestrator:
    """
    Supervisor agent that routes conversations to specialized agents
    based on ASP directives and conversation state.

    The orchestrator follows a strict priority:
    1. Safety monitor always runs first (can override everything)
    2. ASP directives determine which agent handles the turn
    3. Falls back to strategy agent for general coaching
    """

    def __init__(self):
        self.safety = SafetyMonitor()
        self.agents = {
            "intake": IntakeAgent(),
            "strategy": StrategyAgent(),
            "progress": ProgressAgent(),
        }
        self.session_states: dict[str, dict] = {}

    def process(
        self,
        message: str,
        predicates: list[dict],
        asp_directives: list[str],
        session_id: str,
    ) -> dict:
        # Always check safety first - this overrides everything
        safety_result = self.safety.check(message)
        if not safety_result["safe"]:
            return {
                "response": self.safety.process(message, {"session_id": session_id}),
                "agent": "safety",
            }

        # Initialize session state if new
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "phase": "intake",
                "turn_count": 0,
                "predicates_history": [],
            }

        state = self.session_states[session_id]
        state["turn_count"] += 1
        state["predicates_history"].extend(predicates)

        # Route based on ASP directives
        selected_agent = self._select_agent(asp_directives, state)
        context = {
            "session_id": session_id,
            "predicates": predicates,
            "state": state,
        }

        response = selected_agent.process(message, context)

        return {
            "response": response,
            "agent": selected_agent.name,
        }

    def _select_agent(self, asp_directives: list[str], state: dict):
        """Select the appropriate agent based on ASP directives and state."""
        # Check if any agent explicitly matches the ASP directives
        for agent in self.agents.values():
            if agent.can_handle(asp_directives):
                return agent

        # Default routing based on conversation phase
        phase = state.get("phase", "intake")
        if phase in self.agents:
            return self.agents[phase]

        return self.agents["strategy"]

    def get_state(self, session_id: str) -> dict:
        return self.session_states.get(session_id, {"phase": "new", "turn_count": 0})
