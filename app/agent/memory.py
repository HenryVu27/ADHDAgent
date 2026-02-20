"""MemoryManager — async background tasks for rolling summarization,
gated fact extraction, and episodic memory creation.

Runs after the response is sent to the parent. Non-blocking.
"""

import asyncio
import logging

from app.agent.store_protocol import SessionStoreBase
from app.config import settings
from app.models.schemas import EpisodicMemory, SessionSummary

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages rolling summaries, fact extraction, and episodic memories."""

    def __init__(self, session_store: SessionStoreBase, gemini_client):
        self._store = session_store
        self._gemini = gemini_client

    async def post_turn_tasks(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Fire background memory tasks after a turn completes.

        Runs summarization, fact extraction, and episodic memory in parallel.
        Errors are caught and logged — never block the response pipeline.
        """
        tasks = []

        # Rolling summary: only at interval boundaries
        if turn > 0 and turn % settings.SUMMARY_INTERVAL_TURNS == 0:
            tasks.append(self._update_summary(session_id, turn))

        # Fact extraction: only for substantive user messages
        if len(user_message.strip()) >= settings.FACT_EXTRACTION_MIN_LENGTH:
            tasks.append(self._extract_facts(session_id, user_message, turn))

        # Episodic memory: only when track_outcome was called this turn
        tool_calls = tool_calls or []
        outcome_calls = [
            tc for tc in tool_calls
            if tc.get("name") == "track_outcome"
        ]
        if outcome_calls:
            tasks.append(self._create_episode(
                session_id, turn, user_message, assistant_response, outcome_calls,
            ))

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Memory task %d failed: %s", i, result)

    async def _update_summary(self, session_id: str, current_turn: int) -> None:
        """Generate a rolling summary covering turns since the last summary."""
        if not self._gemini:
            return

        # Determine which turns to summarize
        existing = self._store.get_latest_summary(session_id)
        start_turn = (existing.covers_through_turn + 1) if existing else 1

        messages = self._store.get_messages_range(session_id, start_turn, current_turn)
        if not messages:
            return

        # Build conversation text for summarization
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        prior_summary = existing.summary if existing else ""
        prompt = f"""Summarize this ADHD coaching conversation concisely. Focus on:
- Key family information shared
- Strategies discussed or recommended
- Parent's emotional state and concerns
- Any outcomes or progress reported

{f"Previous summary: {prior_summary}" if prior_summary else ""}

New conversation to incorporate:
{conversation_text}

Write a concise summary (2-4 sentences) that captures the most important context for continuing this conversation."""

        try:
            summary_text = await self._gemini.generate(prompt, temperature=0.3)
            self._store.save_summary(
                session_id,
                SessionSummary(summary=summary_text.strip(), covers_through_turn=current_turn),
            )
            logger.info("Summary updated for session %s through turn %d", session_id, current_turn)
        except Exception as e:
            logger.error("Summary generation failed for session %s: %s", session_id, e)
            raise

    async def _extract_facts(self, session_id: str, user_message: str, turn: int) -> None:
        """Extract structured facts from the user message and update the profile."""
        if not self._gemini:
            return

        prompt = f"""Extract any family profile facts from this parent's message about their child with ADHD.
Return a JSON object with only the fields that are explicitly mentioned. Valid fields:
- child_name (string)
- child_age (string)
- diagnosis_status (string: "diagnosed", "suspected", "evaluating")
- challenge_areas (list of strings)
- attempted_strategies (list of strings)
- good_day_description (string)
- hardest_situations (list of strings)

If no profile facts are mentioned, return an empty object {{}}.

Parent message: "{user_message}"
"""

        try:
            facts = await self._gemini.extract_json(prompt)
            if isinstance(facts, dict) and facts:
                # Filter to valid profile fields only
                valid_fields = {
                    "child_name", "child_age", "diagnosis_status",
                    "challenge_areas", "attempted_strategies",
                    "good_day_description", "hardest_situations",
                }
                filtered = {k: v for k, v in facts.items() if k in valid_fields and v}
                if filtered:
                    self._store.update_profile(session_id, **filtered)
                    logger.info(
                        "Facts extracted for session %s (turn %d): %s",
                        session_id, turn, list(filtered.keys()),
                    )
        except Exception as e:
            logger.error("Fact extraction failed for session %s: %s", session_id, e)
            raise

    async def _create_episode(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        assistant_response: str,
        outcome_calls: list[dict],
    ) -> None:
        """Create an episodic memory when an outcome is tracked."""
        for tc in outcome_calls:
            args = tc.get("args", {})
            strategy_name = args.get("strategy_name", "unknown strategy")
            outcome = args.get("outcome", "")
            notes = args.get("notes", "")

            summary = f"Parent reported {outcome} outcome for '{strategy_name}'"
            if notes:
                summary += f": {notes}"

            episode = EpisodicMemory(
                event_type="outcome_reported",
                summary=summary,
                outcome=outcome,
                strategies_involved=[strategy_name],
                emotional_context=self._infer_emotion(user_message),
                turn_range_start=turn,
                turn_range_end=turn,
            )
            self._store.add_episode(session_id, episode)
            logger.info(
                "Episode created for session %s: %s -> %s",
                session_id, strategy_name, outcome,
            )

    @staticmethod
    def _infer_emotion(message: str) -> str:
        """Simple rule-based emotion inference from the user message."""
        lower = message.lower()
        if any(w in lower for w in ("frustrated", "angry", "mad", "furious", "fed up")):
            return "frustrated"
        if any(w in lower for w in ("worried", "anxious", "scared", "nervous", "afraid")):
            return "anxious"
        if any(w in lower for w in ("happy", "great", "amazing", "wonderful", "excited", "thrilled")):
            return "positive"
        if any(w in lower for w in ("sad", "hopeless", "overwhelmed", "exhausted", "tired")):
            return "overwhelmed"
        if any(w in lower for w in ("worked", "helped", "better", "improved", "progress")):
            return "hopeful"
        return ""
