"""Generate questions from factual statements across four types.

Types map to distinct retrieval and reasoning failure modes:
  fact_single    — direct lookup; tests basic retrieval precision
  reasoning      — requires inference beyond what's stated; tests generation quality
  multi_context  — requires combining info across chunks; tests recall breadth
  out_of_scope   — related topic not in corpus; tests boundary detection
"""
from __future__ import annotations

import logging
import random

from eval.generators.llm import GenClient

logger = logging.getLogger(__name__)

_PROMPTS: dict[str, str] = {
    "fact_single": """\
Write a realistic question that a parent of a child with ADHD might ask a coaching chatbot,
where the answer is directly stated in the following fact.

Fact: {statement}

Requirements:
- The question must be self-contained (no references to "this document" or "the above")
- The question must NOT contain the answer
- The question should sound like natural language, not a quiz question
- One sentence only

Return a JSON object: {{"question": "...", "reference_answer": "..."}}
""",

    "reasoning": """\
Write a question that requires REASONING or INFERENCE to answer, inspired by this fact.
The answer should not be directly stated — the responder must draw a logical conclusion.

Fact: {statement}

Requirements:
- The question must be self-contained
- Answering correctly requires understanding the underlying mechanism, not just recalling the fact
- Sound like something a curious or frustrated parent would genuinely ask
- One sentence only

Return a JSON object: {{"question": "...", "reference_answer": "..."}}
""",

    "multi_context": """\
Write a question about ADHD parenting that would require information from MULTIPLE sources
to answer fully. Use this fact as one of the required pieces, but the complete answer
needs additional context.

Fact: {statement}

Requirements:
- The question must be self-contained
- A complete answer genuinely requires synthesizing across at least 2 topics
- Sound like a realistic parent question
- One sentence only

Return a JSON object: {{"question": "...", "reference_answer": "...", "additional_context_needed": "brief description of what else is needed"}}
""",

    "out_of_scope": """\
Write a question related to ADHD or parenting that is NOT answerable from this fact,
and likely NOT covered in a general ADHD coaching knowledge base.

Fact (for thematic inspiration only, NOT the answer): {statement}

Examples of out-of-scope questions:
- Questions requiring medical diagnosis or medication dosing
- Questions about specific clinical treatment protocols
- Questions that require access to the child's school records

Requirements:
- The question must be realistic — something a parent might actually ask
- It should be clearly outside what a coaching chatbot should answer
- One sentence only

Return a JSON object: {{"question": "...", "why_out_of_scope": "brief explanation"}}
""",

    "procedure": """\
Write a "how do I" or "what steps" question that asks for a concrete procedure or sequence
of actions, where the answer is grounded in this fact.

Fact: {statement}

Requirements:
- The question must ask for a process, sequence, or how-to (not just a yes/no or a single fact)
- The question must be self-contained
- Sound like a parent asking for actionable guidance
- One sentence only

Return a JSON object: {{"question": "...", "reference_answer": "..."}}
""",

    "comparative": """\
Write a question that asks a parent to COMPARE or CHOOSE between two approaches related
to managing ADHD. Use this fact as context for one of the approaches.

Fact: {statement}

Requirements:
- The question should frame a realistic trade-off or choice a parent faces
- It must be self-contained and not reference "this document"
- Sound like genuine parental uncertainty, not a quiz
- One sentence only

Return a JSON object: {{"question": "...", "reference_answer": "..."}}
""",
}


async def generate_question(
    statement: str,
    question_type: str,
    chunk: dict,
    llm: GenClient,
) -> dict | None:
    """Generate one question of the given type from a factual statement.

    Returns a dict with at minimum: question, question_type, source_statement,
    source_doc_id. Returns None if generation fails or output is invalid.
    """
    prompt = _PROMPTS[question_type].format(statement=statement)
    result = await llm.json(prompt, temperature=0.7, max_tokens=512)

    if not isinstance(result, dict) or "question" not in result:
        logger.warning("Question generator returned invalid output for type %s", question_type)
        return None

    question = result.get("question", "").strip()
    if len(question) < 10:
        return None

    return {
        "question": question,
        "question_type": question_type,
        "reference_answer": result.get("reference_answer", ""),
        "additional_context_needed": result.get("additional_context_needed", ""),
        "why_out_of_scope": result.get("why_out_of_scope", ""),
        "source_statement": statement,
        "source_doc_id": chunk["document_id"],
        "source_doc_name": chunk["document_name"],
        "expected_doc_ids": [chunk["document_id"]],
    }


def sample_question_type(weights: dict[str, float]) -> str:
    types = list(weights.keys())
    probs = [weights[t] for t in types]
    return random.choices(types, weights=probs, k=1)[0]
