# LLM Query Rewriter
# Resolves pronouns and adds conversation context for better retrieval.

import logging

from langsmith import traceable

from app.config import settings

logger = logging.getLogger(__name__)

QUERY_REWRITE_PROMPT = """You are a search query optimizer for an ADHD parenting coach knowledge base.

The knowledge base contains practical documents: behavioral strategies (step-by-step), parenting guidance, and factual summaries about ADHD. It does NOT contain neurobiological explanations or clinical mechanisms.

Given a parent's current message and recent conversation context, rewrite the query to match how the knowledge base is organized.

Current query:
<query>
{query}
</query>

Recent conversation:
<conversation>
{conversation_history}
</conversation>

Family profile: {family_profile}

Rules:
- Output ONLY the rewritten query, nothing else
- Keep it concise (under 30 words)
- Resolve pronouns (e.g., "he" -> the child's name or "my child")
- Add relevant context from the conversation (e.g., child's age, specific challenge)
- Strip emotional language — focus on the information need, not the parent's feelings
- CRITICAL: For "why" questions, translate to the practical topic the parent needs. "Why can't my child X?" becomes a search for strategies/facts about X. The knowledge base has strategies and facts, not explanations of brain chemistry.
- If the query is already self-contained, return it unchanged

Examples:
- Query: "What do I do when he won't stop?" | Context: discussing homework meltdowns, child age 8 -> "strategies for 8 year old ADHD homework meltdowns refusing to stop"
- Query: "I'm SO frustrated, nothing works for bedtime" | Context: child age 6 -> "bedtime strategies ADHD 6 year old not working alternatives"
- Query: "Does weather affect his emotions and focus?" | Context: child with ADHD -> "weather environmental factors ADHD child emotions focus attention"
- Query: "Why can't my child just do things without a reward?" | Context: child age 9 -> "ADHD motivation intrinsic vs extrinsic rewards goal setting strategies"
- Query: "Why does it feel like his focus got so much worse in third grade?" | Context: child age 8 -> "executive function development ADHD school age academic demands"
- Query: "Why won't she just sit still and listen?" | Context: child age 6, hyperactive-impulsive -> "ADHD hyperactivity impulse control strategies self-regulation 6 year old\""""


class QueryRewriter:
    # Rewrites queries using conversation context

    def __init__(self, gemini_client=None):
        self._gemini = gemini_client

    # Rewrite query with last 3 turns + family profile. Returns original on error.
    @traceable(name="query_rewriter.rewrite", run_type="llm")
    async def rewrite(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        family_profile: dict | None = None,
    ) -> str:
        if not self._gemini:
            return query

        if not conversation_history:
            conversation_history = []

        try:
            recent_turns = conversation_history[-3:]
            history_lines = []
            for turn in recent_turns:
                role = turn.get("role")
                content = turn.get("content", "")
                if role == "user":
                    history_lines.append(f"Parent: {content}")
                elif role == "assistant":
                    history_lines.append(f"Coach: {content}")
            history_text = "\n".join(history_lines)

            profile_text = ""
            if family_profile:
                parts = []
                if family_profile.get("child_age"):
                    parts.append(f"Child age: {family_profile['child_age']}")
                if family_profile.get("challenge_areas"):
                    parts.append(f"Challenges: {', '.join(family_profile['challenge_areas'])}")
                profile_text = "; ".join(parts)

            prompt = QUERY_REWRITE_PROMPT.format(
                query=query,
                conversation_history=history_text,
                family_profile=profile_text or "Not yet gathered",
            )

            rewritten = await self._gemini.generate(
                prompt, temperature=0.0, max_output_tokens=1024,
                timeout=settings.RAG_EMBED_TIMEOUT_S,
            )
            rewritten = rewritten.strip().strip('"').strip("'")

            if rewritten and len(rewritten) < 500:
                logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
                return rewritten

            return query

        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return query
