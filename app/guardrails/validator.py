"""
NeMo Guardrails multi-rail validation.

Provides input rails (jailbreak, crisis, out-of-scope, content, topic)
and output rails (medication, diagnosis, scope). No keyword fallback —
NeMo failure raises GuardrailsError.
"""

import asyncio
import logging
import time
from pathlib import Path

from app.config import settings
from app.models.schemas import GuardrailsError, InputCheckResult, OutputCheckResult

logger = logging.getLogger(__name__)

# Pre-built responses for input rail blocks.
# IMPORTANT: These must match the bot response templates in config/rails.co.
# NeMo's Colang requires inline strings, so the templates are duplicated there.
# If you update wording here, update rails.co to match.
CRISIS_RESPONSE = (
    "I hear what you're sharing, and I want you to know it matters. "
    "What you're describing sounds like it needs immediate professional support.\n\n"
    "**If there is an immediate safety concern:**\n"
    "- Call 911 for emergencies\n"
    "- Call or text 988 for the Suicide & Crisis Lifeline\n"
    "- Text HOME to 741741 for the Crisis Text Line\n"
    "- Childhelp National Child Abuse Hotline: 1-800-422-4453\n\n"
    "Please reach out to a professional who can help right away. "
    "I'll be here whenever you're ready to continue our conversation."
)

OUT_OF_SCOPE_RESPONSE = (
    "That's an important question, but it falls outside what I can help with. "
    "Questions about medical topics should be directed to your child's healthcare "
    "provider, who knows your family's specific situation.\n\n"
    "I'm here to help with behavioral strategies, daily routines, "
    "and practical parenting approaches. What would you like to work on?"
)

OFF_TOPIC_RESPONSE = (
    "I'm specifically designed to help with ADHD parenting strategies and "
    "challenges. Could you tell me more about what's going on with your child "
    "so I can suggest some helpful approaches?"
)

BLOCKED_RESPONSE = (
    "I'm not able to respond to that. I'm here to help with evidence-based "
    "ADHD parenting strategies. How can I help you with your child today?"
)

# Unique phrases from rails.co bot response templates.
# When NeMo blocks input, generate_async returns the template text — match
# against distinctive substrings of those templates (not the template names).
LANGUAGE_RESPONSE = (
    "I'm currently only available in English. Could you share what's going on "
    "in English so I can help you with strategies for your child?"
)

_INPUT_BLOCK_MARKERS = {
    "i want you to know it matters": ("crisis", CRISIS_RESPONSE),
    "988": ("crisis", CRISIS_RESPONSE),
    "falls outside what i can help with": ("out_of_scope", OUT_OF_SCOPE_RESPONSE),
    "directed to your child's healthcare provider": ("out_of_scope", OUT_OF_SCOPE_RESPONSE),
    "specifically designed to help with adhd parenting strategies": ("off_topic", OFF_TOPIC_RESPONSE),
    "only available in english": ("language", LANGUAGE_RESPONSE),
}


class GuardrailsValidator:
    """
    Multi-rail guardrails using NeMo.

    Two public methods: check_input() and check_output().
    No keyword fallback — failure raises GuardrailsError.
    """

    def __init__(self, gemini_client=None):
        self._rails = None
        self._timeout_s = settings.NEMO_GUARDRAILS_TIMEOUT_MS / 1000
        self._gemini_client = gemini_client

        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            logger.warning("nest_asyncio not installed — NeMo may have event loop issues")

        try:
            from nemoguardrails import LLMRails, RailsConfig

            config_path = Path(__file__).parent / "config"
            config = RailsConfig.from_path(str(config_path))

            if gemini_client:
                from app.guardrails.gemini_provider import make_gemini_llm
                llm = make_gemini_llm(gemini_client)
                self._rails = LLMRails(config, llm=llm)
            else:
                self._rails = LLMRails(config)

            logger.info("NeMo Guardrails initialized (input + output rails)")
        except ImportError:
            raise GuardrailsError("nemoguardrails package not installed")
        except Exception as e:
            raise GuardrailsError(f"NeMo Guardrails init failed: {e}")

    async def check_input(self, user_message: str, context: dict | None = None) -> InputCheckResult:
        """Run NeMo input rails (jailbreak, crisis, out-of-scope, content, topic).

        Returns InputCheckResult with is_allowed, blocked_reason, override_response.
        Raises GuardrailsError if NeMo fails.
        """
        start = time.time()
        try:
            result = await asyncio.wait_for(
                self._nemo_input_check(user_message, context),
                timeout=self._timeout_s,
            )
            result.duration_ms = (time.time() - start) * 1000
            return result
        except asyncio.TimeoutError:
            duration = (time.time() - start) * 1000
            raise GuardrailsError(f"Input rails timed out after {duration:.0f}ms")
        except GuardrailsError:
            raise
        except Exception as e:
            raise GuardrailsError(f"Input rails failed: {e}")

    async def check_output(self, bot_response: str, context: dict | None = None) -> OutputCheckResult:
        """Run NeMo output rails (medication, diagnosis, scope).

        Returns OutputCheckResult with is_valid, violation_type.
        Raises GuardrailsError if NeMo fails.
        """
        start = time.time()
        try:
            result = await asyncio.wait_for(
                self._nemo_output_check(bot_response, context),
                timeout=self._timeout_s,
            )
            result.duration_ms = (time.time() - start) * 1000
            return result
        except asyncio.TimeoutError:
            duration = (time.time() - start) * 1000
            raise GuardrailsError(f"Output rails timed out after {duration:.0f}ms")
        except GuardrailsError:
            raise
        except Exception as e:
            raise GuardrailsError(f"Output rails failed: {e}")

    async def _nemo_input_check(self, user_message: str, context: dict | None = None) -> InputCheckResult:
        """Run NeMo input rail check.

        NeMo's Colang 1.0 rail checks operate on $user_message (single message)
        regardless of conversation history. Passing multi-turn messages confuses
        NeMo's internal state machine and produces unreliable results. False
        positives for short contextual replies are handled in the hooks layer.
        """
        messages = [{"role": "user", "content": user_message}]

        result = await self._rails.generate_async(messages=messages)
        output_text = result.get("content", result) if isinstance(result, dict) else str(result)
        logger.debug("NeMo input check output (%d chars): %s", len(output_text), output_text[:300])

        # Check if NeMo blocked the input by matching unique phrases from
        # the bot response templates defined in rails.co.  When a rail fires,
        # NeMo returns the template text — we match against distinctive substrings.
        output_lower = output_text.lower()
        for marker, (reason, response) in _INPUT_BLOCK_MARKERS.items():
            if marker in output_lower:
                logger.info("NeMo input blocked: reason=%s, marker=%r", reason, marker)
                return InputCheckResult(
                    is_allowed=False,
                    blocked_reason=reason,
                    override_response=response,
                )

        # NeMo's built-in "refuse to respond" has no explicit template in
        # rails.co, so check if the response is very short (typical of NeMo
        # refusal defaults like "I'm sorry, I can't respond to that.")
        # AND doesn't look like a real coaching response.
        if len(output_text) < 100 and any(
            phrase in output_lower
            for phrase in ["i'm sorry", "sorry, i can't", "sorry, i cannot"]
        ):
            logger.info("NeMo input blocked: reason=blocked (NeMo default refusal)")
            return InputCheckResult(
                is_allowed=False,
                blocked_reason="blocked",
                override_response=BLOCKED_RESPONSE,
            )

        return InputCheckResult(is_allowed=True)

    async def _nemo_output_check(self, bot_response: str, context: dict | None = None) -> OutputCheckResult:
        """Run output rail checks via direct LLM classification.

        NeMo's generate_async is designed for end-to-end pipelines (user → response),
        not standalone output validation.  Calling it with only an assistant message
        produces unreliable results.  Instead, we call the LLM directly using the
        same prompt templates defined in config.yml.
        """
        if self._gemini_client is None:
            logger.warning("No Gemini client — skipping output rails")
            return OutputCheckResult(is_valid=True)

        output_checks = [
            (
                "medication",
                "Check if this chatbot response recommends or suggests medication.\n\n"
                "Violations include: recommending specific drugs (Adderall, Ritalin, etc.), "
                "suggesting medication classes (stimulants, non-stimulants), advising "
                "dosage changes, or even indirect suggestions (\"stimulant treatment might help\").\n\n"
                "ALLOWED: acknowledging medication the parent already mentioned "
                "(\"since your child is on medication\"), brief redirect to provider "
                "(\"that's a question for your pediatrician\", \"your child's healthcare "
                "provider can help with that\").\n\n"
                "Does this response recommend or suggest medication?"
            ),
            (
                "diagnosis",
                "Check if this chatbot response diagnoses or suggests a diagnosis.\n\n"
                "Violations: \"your child has ADHD\", \"sounds like ADHD\", \"likely meets "
                "criteria\", \"I believe your child has...\", \"should be evaluated for ADHD\", "
                "or any statement that functions as a diagnostic opinion.\n\n"
                "ALLOWED: acknowledging an existing diagnosis the parent shared "
                "(\"since your child was diagnosed with ADHD\", \"given the ADHD diagnosis\").\n\n"
                "Does this response make or suggest a diagnosis?"
            ),
            (
                "scope",
                "Check if this chatbot response ventures outside its scope as an ADHD "
                "parenting coach.\n\n"
                "Violations: providing legal advice, detailed nutrition therapy, "
                "psychiatric treatment protocols, occupational therapy specifics, "
                "or any medical specialty guidance.\n\n"
                "ALLOWED: behavioral strategies, parenting techniques, emotional "
                "support, brief redirects to appropriate professionals.\n\n"
                "Does this response go outside the scope of ADHD parenting coaching?"
            ),
        ]

        async def _run_single_check(violation_type: str, check_prompt: str) -> tuple[str, str]:
            """Run one output rail check. Returns (violation_type, answer)."""
            try:
                prompt = (
                    f"You are a safety classifier for an ADHD parenting coach chatbot.\n\n"
                    f"{check_prompt}\n\n"
                    f"Response to check: \"{bot_response}\"\n\n"
                    f"Answer only \"yes\" or \"no\"."
                )
                result = await self._gemini_client.generate(prompt, temperature=0.0)
                answer = result.strip().lower()
                logger.debug("Output rail [%s] answer: %r", violation_type, answer)
                return (violation_type, answer)
            except Exception as e:
                logger.error("Output rail [%s] failed: %s", violation_type, e)
                return (violation_type, "no")

        # Run all 3 checks in parallel. return_exceptions=True prevents a
        # CancelledError in one task from killing the others when wait_for
        # times out in the caller.
        results = await asyncio.gather(
            *[_run_single_check(vt, cp) for vt, cp in output_checks],
            return_exceptions=True,
        )

        # Return the first violation found (preserving check priority order)
        for r in results:
            if isinstance(r, BaseException):
                logger.error("Output rail check raised: %s", r)
                continue
            violation_type, answer = r
            if answer.startswith("yes"):
                logger.info("Output rail [%s] triggered", violation_type)
                return OutputCheckResult(
                    is_valid=False,
                    violation_type=violation_type,
                )

        return OutputCheckResult(is_valid=True)
