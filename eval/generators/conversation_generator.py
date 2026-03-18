"""Generate synthetic conversations with known ground-truth fact annotations.

Uses a Generator-Critic architecture:
  1. Generator: produces a realistic parent-coach conversation that naturally
     surfaces the persona's ground-truth facts.
  2. Critic: verifies no fact was contradicted and all facts were surfaced.

Output is used to evaluate the memory system's fact extraction precision/recall
independently of retrieval quality.
"""
from __future__ import annotations

import logging
import uuid

from eval.config import MAX_CRITIC_RETRIES
from eval.generators.llm import GenClient

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """\
Generate a realistic conversation between an ADHD coaching chatbot and a parent.

The parent has this profile (do NOT reveal all facts upfront — they must emerge naturally):
{persona_facts_formatted}

Conversation requirements:
- 8-14 turns (user + assistant alternating, starting with user)
- The parent should reveal their profile facts organically through their questions and descriptions
- Facts should emerge in context (e.g., mention working nights rather than "I work at night")
- The assistant should ask clarifying questions and give practical ADHD coaching advice
- Include realistic emotional texture (frustration, hope, uncertainty)
- Do NOT have the parent list their facts as a bullet list

Return a JSON object:
{{
  "turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}
"""

_CRITIC_PROMPT = """\
You are validating a synthetic conversation against a user persona.

Persona facts that should have emerged naturally in the conversation:
{persona_facts_formatted}

Conversation:
{conversation_text}

For each persona fact, determine:
1. surfaced: did this fact appear in the conversation? (true/false)
2. contradicted: did any utterance CONTRADICT this fact? (true/false)
3. turn: which user turn (1-indexed) first surfaced this fact? (null if not surfaced)
4. verbatim_quote: the exact phrase from the conversation that reveals this fact (null if not surfaced)

Return JSON:
{{
  "facts": [
    {{
      "fact_key": "...",
      "surfaced": true/false,
      "contradicted": false,
      "turn": <int or null>,
      "verbatim_quote": "<string or null>"
    }}
  ],
  "valid": true/false,
  "issues": "describe any contradictions or major problems, or empty string"
}}

A conversation is valid if: no fact is contradicted AND at least 70% of facts are surfaced.
"""


def _format_facts(persona: dict) -> str:
    facts = persona.get("facts", {})
    lines = []
    for key, value in facts.items():
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(value)}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _conversation_text(turns: list[dict]) -> str:
    lines = []
    for i, turn in enumerate(turns):
        role = "Parent" if turn["role"] == "user" else "Coach"
        lines.append(f"[Turn {i + 1}] {role}: {turn['content']}")
    return "\n".join(lines)


async def generate_conversation(
    persona: dict,
    llm: GenClient,
) -> dict | None:
    """Generate a single validated conversation for a persona.

    Returns None if critic validation fails after MAX_CRITIC_RETRIES attempts.
    """
    persona_facts_formatted = _format_facts(persona)

    for attempt in range(MAX_CRITIC_RETRIES):
        # Generator
        gen_prompt = _GENERATOR_PROMPT.format(
            persona_facts_formatted=persona_facts_formatted
        )
        gen_result = await llm.json(gen_prompt, temperature=0.8, max_tokens=3000)

        turns = gen_result.get("turns", [])
        if not turns or len(turns) < 4:
            logger.warning("Persona %s attempt %d: too few turns", persona["id"], attempt + 1)
            continue

        # Critic
        conv_text = _conversation_text(turns)
        critic_prompt = _CRITIC_PROMPT.format(
            persona_facts_formatted=persona_facts_formatted,
            conversation_text=conv_text,
        )
        critic_result = await llm.json(critic_prompt, temperature=0.0, max_tokens=4096)

        is_valid = critic_result.get("valid", False)
        if not is_valid:
            logger.info(
                "Persona %s attempt %d failed critic: %s",
                persona["id"], attempt + 1, critic_result.get("issues", "")
            )
            continue

        # Build ground-truth annotations from critic output
        ground_truth = []
        for fact_result in critic_result.get("facts", []):
            if fact_result.get("surfaced"):
                ground_truth.append({
                    "fact_key": fact_result["fact_key"],
                    "expected_value": persona["facts"].get(fact_result["fact_key"]),
                    "turn": fact_result.get("turn"),
                    "verbatim_quote": fact_result.get("verbatim_quote"),
                })

        return {
            "id": f"conv_{uuid.uuid4().hex[:8]}",
            "persona_id": persona["id"],
            "persona_description": persona.get("description", ""),
            "persona_facts": persona["facts"],
            "turns": turns,
            "ground_truth_extractions": ground_truth,
            "critic_issues": critic_result.get("issues", ""),
        }

    logger.error("Persona %s: failed after %d critic attempts", persona["id"], MAX_CRITIC_RETRIES)
    return None
