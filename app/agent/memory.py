"""MemoryManager — async background tasks for rolling summarization,
gated fact extraction, and episodic memory creation.

Runs after the response is sent to the parent. Non-blocking.
"""

import asyncio
import logging
import time

from app.agent.store_protocol import SessionStoreBase
from app.config import settings
from app.models.schemas import EpisodeLink, EpisodicMemory, SessionSummary

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages rolling summaries, fact extraction, and episodic memories."""

    def __init__(self, session_store: SessionStoreBase, gemini_client, event_bus=None):
        self._store = session_store
        self._gemini = gemini_client
        self._event_bus = event_bus

    async def post_turn_tasks(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict] | None = None,
        force_summary: bool = False,
    ) -> None:
        """Fire background memory tasks after a turn completes.

        Runs summarization, fact extraction, and episodic memory in parallel.
        Errors are caught and logged — never block the response pipeline.
        """
        tasks = []

        # Rolling summary: at interval boundaries OR when context is filling up
        if turn > 0 and (turn % settings.SUMMARY_INTERVAL_TURNS == 0 or force_summary):
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

        # Goal events: goal_set and goal_completed
        goal_calls = [
            tc for tc in tool_calls
            if tc.get("name") == "manage_goals"
            and tc.get("args", {}).get("action") in ("add", "complete")
        ]
        if goal_calls:
            tasks.append(self._create_goal_episode(session_id, turn, user_message, goal_calls))

        # Emotional shift: run for any non-empty user message (emotion can appear in short messages)
        if user_message.strip():
            tasks.append(self._run_emotional_shift_check(session_id, turn, user_message))

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
        t0 = time.monotonic()

        # Determine which turns to summarize
        existing = await self._store.get_latest_summary(session_id)
        start_turn = (existing.covers_through_turn + 1) if existing else 1

        messages = await self._store.get_messages_range(session_id, start_turn, current_turn)
        if not messages:
            return

        # Load episodes from this turn window for importance weighting
        all_episodes = await self._store.get_episodes_with_ids(session_id, limit=50)
        window_episodes = [
            ep for _id, ep in all_episodes
            if ep.turn_range_start >= start_turn and ep.turn_range_start <= current_turn
        ]
        key_events_block = ""
        if window_episodes:
            event_lines = []
            for ep in window_episodes:
                line = f"- [{ep.event_type}] {ep.summary}"
                if ep.emotional_context:
                    line += f" (mood: {ep.emotional_context})"
                event_lines.append(line)
            key_events_block = "\nKey events this window:\n" + "\n".join(event_lines)

        # Build conversation text for summarization
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )

        # Load structured profile so the summary doesn't duplicate it
        state = await self._store.get(session_id)
        profile = state.family_profile
        known_facts = []
        if profile.child_name:
            known_facts.append(f"child_name={profile.child_name}")
        if profile.child_age:
            known_facts.append(f"child_age={profile.child_age}")
        if profile.challenge_areas:
            known_facts.append(f"challenges={', '.join(profile.challenge_areas)}")
        if profile.attempted_strategies:
            known_facts.append(f"tried={', '.join(profile.attempted_strategies)}")
        profile_note = (
            f"Already stored in structured profile (do NOT repeat in summary): {'; '.join(known_facts)}"
            if known_facts else ""
        )

        prior_summary = existing.summary if existing else ""
        prompt = f"""Summarize the emotional and narrative arc of this ADHD coaching conversation.

Focus ONLY on what is NOT already captured in the structured family profile:
- How the parent is feeling and what's weighing on them
- Specific concerns, quotes, or worries they've expressed
- How the conversation has evolved (what was tried, how they reacted)
- Relational context (frustration level, trust built, resistance encountered)

{profile_note}

Omit: demographic facts, strategy names, diagnosis details, and anything already in the previous summary.

{f"Previous summary: {prior_summary}" if prior_summary else ""}
{key_events_block}
{f"Pay special attention to the key events above -- they represent important moments that should be preserved in the summary." if key_events_block else ""}
New conversation to incorporate:
{conversation_text}

Write a concise summary (2-4 sentences) focused on narrative and emotional context only."""

        try:
            summary_text = await self._gemini.generate(prompt, temperature=0.0, max_output_tokens=256,
                                                           timeout=settings.MEMORY_TIMEOUT_S)
            await self._store.save_summary(
                session_id,
                SessionSummary(summary=summary_text.strip(), covers_through_turn=current_turn),
            )
            logger.info("Summary updated for session %s through turn %d", session_id, current_turn)
            if self._event_bus:
                await self._event_bus.emit("memory", "summary_updated", session_id, current_turn,
                                           duration_ms=(time.monotonic() - t0) * 1000)
        except Exception as e:
            logger.error("Summary generation failed for session %s: %s", session_id, e)
            raise

    async def _extract_facts(self, session_id: str, user_message: str, turn: int) -> None:
        """Extract structured facts from the user message and update the profile."""
        if not self._gemini:
            return
        t0 = time.monotonic()

        # Include recent conversation history so pronouns can be resolved
        recent_messages = await self._store.get_messages(session_id, limit=6)
        # Filter to last 3 user/assistant exchanges
        recent_history = [m for m in recent_messages if not m.get("blocked")][-3:]
        history_text = ""
        if recent_history:
            lines = []
            for entry in recent_history:
                if entry.get("role") == "user":
                    lines.append(f"Parent: {entry['content']}")
                elif entry.get("role") == "assistant":
                    lines.append(f"Coach: {entry['content']}")
            history_text = "\n".join(lines)

        # Include current profile so we know what's already captured
        state = await self._store.get(session_id)
        profile = state.family_profile
        known_facts = []
        if profile.child_name:
            known_facts.append(f"child_name: {profile.child_name}")
        if profile.child_age:
            known_facts.append(f"child_age: {profile.child_age}")
        profile_text = ", ".join(known_facts) if known_facts else "None yet"

        prompt = f"""Extract any family profile facts from this parent's message about their child with ADHD.
Return a JSON object with only the fields that are explicitly mentioned or clearly implied. Valid fields:
- child_name (string)
- child_age (string, e.g. "7" or "8-9" if ambiguous)
- diagnosis_status (string: "diagnosed", "suspected", "evaluating", "not diagnosed")
- adhd_subtype (string: "inattentive", "hyperactive-impulsive", "combined")
- challenge_areas (list of strings)
- attempted_strategies (list of strings)
- good_day_description (string)
- hardest_situations (list of strings)
- negated_strategies (list of strings — strategies explicitly stated to NOT work, be abandoned, or cause problems)

For negated_strategies: include a strategy name ONLY if the parent explicitly says it failed, stopped working,
was abandoned, caused problems, or they no longer use it. Examples: "we stopped using timers",
"the reward chart didn't work", "sticker charts made things worse".

If the parent corrects previously shared information, extract the CORRECTED value.
If information is ambiguous, use the parent's phrasing (e.g., "about 8 or 9" -> "8-9").
If no NEW profile facts are mentioned, return an empty object {{}}.

Already known: {profile_text}
{f"Recent conversation context:{chr(10)}{history_text}" if history_text else ""}

Parent message:
<parent_message>
{user_message}
</parent_message>
"""

        try:
            facts = await self._gemini.extract_json(prompt, max_output_tokens=512,
                                                        timeout=settings.MEMORY_TIMEOUT_S)
            if isinstance(facts, dict) and facts:
                # Handle negated strategies — create negative outcomes
                negated = facts.get("negated_strategies", [])
                if isinstance(negated, list) and negated:
                    for strategy in negated:
                        if isinstance(strategy, str) and strategy.strip():
                            await self._store.add_outcome(
                                session_id,
                                strategy_name=strategy.strip(),
                                outcome="negative",
                                notes="Parent indicated this strategy is not working or was abandoned.",
                            )
                            logger.info(
                                "Negated strategy recorded for session %s (turn %d): %s",
                                session_id, turn, strategy,
                            )
                            if self._event_bus:
                                await self._event_bus.emit(
                                    "memory", "strategy_negated", session_id, turn,
                                    detail={"strategy": strategy},
                                )

                # Filter to valid profile fields only
                valid_fields = {
                    "child_name", "child_age", "diagnosis_status", "adhd_subtype",
                    "challenge_areas", "attempted_strategies",
                    "good_day_description", "hardest_situations",
                }
                filtered = {k: v for k, v in facts.items() if k in valid_fields and v}
                if filtered:
                    await self._store.update_profile(session_id, **filtered)
                    logger.info(
                        "Facts extracted for session %s (turn %d): %s",
                        session_id, turn, list(filtered.keys()),
                    )
                    if self._event_bus:
                        await self._event_bus.emit("memory", "facts_extracted", session_id, turn,
                                                   duration_ms=(time.monotonic() - t0) * 1000,
                                                   detail={"fields": list(filtered.keys())})
                    # Check for scalar corrections via changelog
                    scalar_fields = {"child_name", "child_age", "diagnosis_status", "adhd_subtype", "good_day_description"}
                    corrections = {k: v for k, v in filtered.items() if k in scalar_fields}
                    if corrections and self._event_bus:
                        changelog = await self._store.get_profile_changelog(session_id)
                        # A changelog entry for a field we just updated means a correction was detected
                        corrected_field_names = set(corrections.keys())
                        detected = [
                            c for c in changelog
                            if c.field in corrected_field_names
                            and c.new_value == str(corrections.get(c.field, ""))
                        ]
                        if detected:
                            await self._event_bus.emit(
                                "memory", "profile_corrected", session_id, turn,
                                detail={"fields": [c.field for c in detected]},
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
        emotional_context = await self._infer_emotion(session_id, user_message)

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
                emotional_context=emotional_context,
                turn_range_start=turn,
                turn_range_end=turn,
            )
            episode_id = await self._store.add_episode(session_id, episode)
            logger.info(
                "Episode created for session %s: %s -> %s",
                session_id, strategy_name, outcome,
            )
            if self._event_bus:
                await self._event_bus.emit("memory", "episode_created", session_id, turn,
                                           detail={"strategy": strategy_name, "outcome": outcome})
            asyncio.create_task(self._link_episode(session_id, episode_id, episode))

    _HIGH_INTENSITY_EMOTIONS = frozenset({"overwhelmed", "frustrated", "anxious"})
    _POSITIVE_EMOTIONS = frozenset({"positive", "hopeful"})

    async def _create_goal_episode(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        goal_calls: list[dict],
    ) -> None:
        """Create an episodic memory for goal set or completion events."""
        emotional_context = await self._infer_emotion(session_id, user_message)

        for tc in goal_calls:
            args = tc.get("args", {})
            action = args.get("action", "")
            description = args.get("description", "unknown goal")

            if action == "add":
                event_type = "goal_set"
                summary = f"Parent set new goal: '{description}'"
                outcome = ""
            elif action == "complete":
                event_type = "goal_completed"
                summary = f"Parent completed goal: '{description}'"
                outcome = "positive"
            else:
                continue

            episode = EpisodicMemory(
                event_type=event_type,
                summary=summary,
                outcome=outcome,
                strategies_involved=[],
                emotional_context=emotional_context,
                turn_range_start=turn,
                turn_range_end=turn,
            )
            episode_id = await self._store.add_episode(session_id, episode)
            logger.info("Goal episode created for session %s: %s", session_id, event_type)
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", event_type, session_id, turn,
                    detail={"goal": description},
                )
            asyncio.create_task(self._link_episode(session_id, episode_id, episode))

    async def _create_emotional_shift_episode(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        emotion: str,
    ) -> None:
        """Create an episode for any detected non-neutral emotional state."""
        if not emotion:
            return

        if emotion in self._HIGH_INTENSITY_EMOTIONS:
            outcome = "mixed"
        elif emotion in self._POSITIVE_EMOTIONS:
            outcome = "positive"
        else:
            logger.warning("Uncategorized emotion '%s'; defaulting outcome to 'mixed'", emotion)
            outcome = "mixed"

        summary = f"Parent expressed {emotion} emotional state at turn {turn}"

        episode = EpisodicMemory(
            event_type="emotional_shift",
            summary=summary,
            outcome=outcome,
            strategies_involved=[],
            emotional_context=emotion,
            turn_range_start=turn,
            turn_range_end=turn,
        )
        episode_id = await self._store.add_episode(session_id, episode)
        logger.info("Emotional shift episode for session %s: %s", session_id, emotion)
        if self._event_bus:
            await self._event_bus.emit(
                "memory", "emotional_shift", session_id, turn,
                detail={"emotion": emotion},
            )
        asyncio.create_task(self._link_episode(session_id, episode_id, episode))

    async def _run_emotional_shift_check(self, session_id: str, turn: int, user_message: str) -> None:
        """Infer emotion and create an episode for any non-neutral emotional state."""
        emotion = await self._infer_emotion(session_id, user_message)
        await self._create_emotional_shift_episode(session_id, turn, user_message, emotion)

    async def _link_episode(self, session_id: str, new_episode_id: int, new_episode: EpisodicMemory) -> None:
        """Find related prior episodes and create rule-based links."""
        try:
            prior = await self._store.get_episodes_with_ids(session_id, limit=20)
            for prior_id, prior_ep in prior:
                if prior_id == new_episode_id:
                    continue
                link_type = None
                link_reason = ""

                # Same strategy involved
                shared_strategies = set(new_episode.strategies_involved) & set(prior_ep.strategies_involved)
                if shared_strategies:
                    link_type = "same_strategy"
                    link_reason = f"Both involve: {', '.join(shared_strategies)}"

                # Same strong emotion
                elif (new_episode.emotional_context
                      and prior_ep.emotional_context
                      and new_episode.emotional_context == prior_ep.emotional_context
                      and new_episode.emotional_context in self._HIGH_INTENSITY_EMOTIONS | self._POSITIVE_EMOTIONS):
                    link_type = "same_emotion"
                    link_reason = f"Both reflect '{new_episode.emotional_context}' emotional state"

                if link_type:
                    link = EpisodeLink(
                        source_id=new_episode_id,
                        target_id=prior_id,
                        link_type=link_type,
                        link_reason=link_reason,
                    )
                    await self._store.add_episode_link(session_id, link)
        except Exception as e:
            logger.warning("Episode linking failed (non-critical): %s", e)

    async def _infer_emotion(self, session_id: str, user_message: str) -> str:
        """Classify the parent's emotional state using an LLM call with conversation context."""
        if not self._gemini:
            return ""
        t0 = time.monotonic()

        valid_emotions = {"frustrated", "anxious", "positive", "overwhelmed", "hopeful", "neutral"}

        # Build conversation context from recent history
        recent_messages = await self._store.get_messages(session_id, limit=6)
        recent_history = [m for m in recent_messages if not m.get("blocked")][-3:]
        context_lines = []
        for entry in recent_history:
            if entry.get("role") == "user":
                context_lines.append(f"Parent: {entry['content']}")
            elif entry.get("role") == "assistant":
                context_lines.append(f"Coach: {entry['content']}")
        conversation_context = (
            f"Recent conversation:\n{chr(10).join(context_lines)}" if context_lines else ""
        )

        prompt = f"""Classify the PARENT's emotional state from this ADHD coaching conversation message.

Choose the MOST fitting word. Default to neutral only when the message is purely factual with no emotional signal.

frustrated — parent is annoyed, exhausted, at their limit ("I can't keep doing this", "nothing works")
anxious — parent is worried or uncertain ("I'm scared about his future", "what if this doesn't help")
overwhelmed — parent is stretched beyond capacity ("I have no idea where to start", "it's too much")
hopeful — parent sees progress or possibility ("this might actually work", "I think we're getting there")
positive — parent is happy or reporting clear success ("it worked great", "so much better this week")
neutral — purely factual, no emotional tone ("he's 7", "we tried timers once")

IMPORTANT: If the parent is describing a struggle, difficulty, or asking for help, that usually signals some level of frustration or anxiety — not neutral.

{conversation_context}

Return exactly one word from: frustrated, anxious, positive, overwhelmed, hopeful, neutral

Parent message: {user_message}"""

        try:
            result = await self._gemini.generate(
                prompt, temperature=0.0, max_output_tokens=16,
                timeout=settings.MEMORY_TIMEOUT_S,
            )
            emotion = result.strip().lower()
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", "emotion_inferred", session_id, 0,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    detail={"emotion": emotion},
                )
            if emotion not in valid_emotions or emotion == "neutral":
                return ""
            return emotion
        except Exception as e:
            logger.error("Emotion inference failed: %s", e)
            raise
