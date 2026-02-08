from app.agents.base import BaseAgent


INTAKE_QUESTIONS = [
    "How old is your child, and when were they diagnosed with ADHD?",
    "What are the biggest daily challenges you face related to your child's ADHD?",
    "What strategies have you tried so far? What has worked or not worked?",
    "Are there specific situations (homework, bedtime, transitions) that are hardest?",
    "What does a good day look like for your family?",
]


class IntakeAgent(BaseAgent):
    """
    Gathers family context through structured intake questions.
    Builds a profile of the child, family situation, and current challenges.
    """

    name = "intake"
    description = "Gathers family context and child profile through guided intake"

    def __init__(self):
        self.session_progress: dict[str, int] = {}

    def process(self, message: str, context: dict) -> str:
        session_id = context.get("session_id", "default")
        progress = self.session_progress.get(session_id, 0)

        if progress == 0:
            self.session_progress[session_id] = 1
            return (
                "Welcome! I'm here to help you build strategies that work for your "
                "family. Let's start by getting to know your situation a bit.\n\n"
                f"{INTAKE_QUESTIONS[0]}"
            )

        # Acknowledge what the parent shared, then ask the next question
        if progress < len(INTAKE_QUESTIONS):
            self.session_progress[session_id] = progress + 1
            acknowledgment = "Thank you for sharing that. "
            return f"{acknowledgment}{INTAKE_QUESTIONS[progress]}"

        # Intake complete
        return (
            "Thank you for walking me through all of that. I have a much better "
            "picture of your family's situation now. Based on what you've shared, "
            "I'd like to suggest some strategies. Would you like to start with the "
            "area that feels most challenging right now?"
        )

    def can_handle(self, asp_directives: list[str]) -> bool:
        return any("intake" in d or "gather_info" in d for d in asp_directives)

    def get_progress(self, session_id: str) -> dict:
        progress = self.session_progress.get(session_id, 0)
        return {
            "questions_asked": progress,
            "total_questions": len(INTAKE_QUESTIONS),
            "complete": progress >= len(INTAKE_QUESTIONS),
        }
