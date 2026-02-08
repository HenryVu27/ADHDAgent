from app.agents.base import BaseAgent


# Topics that require immediate redirection to professional resources
CRISIS_KEYWORDS = [
    "hurt", "harm", "suicide", "kill", "abuse", "danger", "emergency",
    "violent", "threatening", "unsafe",
]

# Topics the chatbot should not provide advice on
OUT_OF_SCOPE_TOPICS = [
    "medication", "dosage", "prescription", "diagnosis", "diagnose",
    "legal", "custody", "divorce",
]

CRISIS_RESPONSE = (
    "I want to make sure you and your family are safe. What you're describing "
    "sounds like it needs immediate professional support.\n\n"
    "**If there is an immediate safety concern:**\n"
    "- Call 911 for emergencies\n"
    "- Call or text 988 for the Suicide & Crisis Lifeline\n"
    "- Text HOME to 741741 for the Crisis Text Line\n\n"
    "Please reach out to a professional who can help right away."
)

OUT_OF_SCOPE_RESPONSE = (
    "That's an important topic, but it's outside what I'm able to help with. "
    "Questions about {topic} should be directed to your child's healthcare "
    "provider, who knows your family's specific situation.\n\n"
    "Is there something else I can help you with today, like behavioral "
    "strategies or daily routine planning?"
)


class SafetyMonitor(BaseAgent):
    """
    Monitors all conversations for safety concerns and out-of-scope topics.
    Acts as a guardrail layer that can override other agents.
    """

    name = "safety"
    description = "Enforces clinician-informed safety guardrails"

    def process(self, message: str, context: dict) -> str:
        """If this agent is invoked, a safety concern was detected."""
        safety_check = self.check(message)

        if safety_check["crisis"]:
            return CRISIS_RESPONSE

        if safety_check["out_of_scope"]:
            topic = safety_check["detected_topic"]
            return OUT_OF_SCOPE_RESPONSE.format(topic=topic)

        return ""

    def check(self, message: str) -> dict:
        """
        Check a message for safety concerns. Called on every message
        before routing to other agents.
        """
        message_lower = message.lower()

        # Check for crisis indicators
        for keyword in CRISIS_KEYWORDS:
            if keyword in message_lower:
                return {
                    "safe": False,
                    "crisis": True,
                    "out_of_scope": False,
                    "detected_topic": keyword,
                }

        # Check for out-of-scope topics
        for topic in OUT_OF_SCOPE_TOPICS:
            if topic in message_lower:
                return {
                    "safe": False,
                    "crisis": False,
                    "out_of_scope": True,
                    "detected_topic": topic,
                }

        return {
            "safe": True,
            "crisis": False,
            "out_of_scope": False,
            "detected_topic": None,
        }

    def can_handle(self, asp_directives: list[str]) -> bool:
        return any("safety" in d or "crisis" in d or "redirect" in d for d in asp_directives)
