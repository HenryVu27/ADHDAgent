from app.agents.base import BaseAgent


class ProgressAgent(BaseAgent):
    """
    Tracks behavioral goals, celebrates wins, and helps adjust plans
    based on what's working or not working.
    """

    name = "progress"
    description = "Tracks goals and progress, celebrates wins, adjusts plans"

    def __init__(self):
        self.session_goals: dict[str, list[dict]] = {}

    def process(self, message: str, context: dict) -> str:
        session_id = context.get("session_id", "default")
        goals = self.session_goals.get(session_id, [])

        if not goals:
            return (
                "It sounds like you're ready to set some goals! Let's start small "
                "and specific. What's one thing you'd like to see improve this week? "
                "For example: 'Start homework within 10 minutes of being asked' or "
                "'Complete the bedtime routine without a meltdown 3 times this week.'"
            )

        return (
            "Let's check in on how things have been going. "
            "Which of your goals would you like to talk about? "
            "Remember, progress isn't always linear, and noticing what's "
            "happening is itself a big step."
        )

    def add_goal(self, session_id: str, goal: dict):
        if session_id not in self.session_goals:
            self.session_goals[session_id] = []
        self.session_goals[session_id].append(goal)

    def can_handle(self, asp_directives: list[str]) -> bool:
        return any("progress" in d or "track" in d or "goal" in d for d in asp_directives)
