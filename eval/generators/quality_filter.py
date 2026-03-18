"""Three-stage quality filter for generated questions.

Stage 1 — Answerability (LLM judge):
  Discard questions that cannot be answered from the source document.
  Catches hallucinated questions where the LLM invented content.

Stage 2 — Self-containment (LLM judge):
  Discard questions that reference "the document", "the above", "this text", etc.
  Such questions are unusable in a real retrieval eval.

Stage 3 — Diversity (embedding deduplication):
  Discard questions that are semantically near-duplicates of already-kept questions.
  Threshold: cosine similarity > DIVERSITY_SIM_THRESHOLD.
"""
from __future__ import annotations

import logging
import math

from eval.config import ANSWERABILITY_MIN_SCORE, DIVERSITY_SIM_THRESHOLD
from eval.generators.llm import GenClient

logger = logging.getLogger(__name__)

_ANSWERABILITY_PROMPT = """\
You are evaluating a question-answer pair for inclusion in a RAG evaluation dataset.

Source document text:
{chunk_text}

Question: {question}
Reference answer: {reference_answer}

Score ANSWERABILITY on a 1-3 scale:
  3 = The source document clearly contains enough information to answer the question
  2 = The source document is relevant but only partially addresses the question
  1 = The question cannot be answered from the source document (hallucinated content)

Score SELF_CONTAINED as true/false:
  true  = the question makes sense standalone, with no reference to "the document", "the above", "this text", etc.
  false = the question depends on context not available to the user

Return JSON with exactly these two fields: {{"answerability": <1|2|3>, "self_contained": <true|false>}}
"""


async def answerability_score(
    question: dict,
    chunk_text: str,
    llm: GenClient,
) -> dict:
    """Score answerability and self-containment for a single question."""
    prompt = _ANSWERABILITY_PROMPT.format(
        chunk_text=chunk_text,
        question=question["question"],
        reference_answer=question.get("reference_answer", ""),
    )
    result = await llm.json(prompt, temperature=0.0, max_tokens=2048)
    return {
        "answerability": int(result.get("answerability", 1)),
        "self_contained": bool(result.get("self_contained", False)),
        "filter_reason": "",
    }


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def filter_questions(
    questions: list[dict],
    chunk_text_by_doc_id: dict[str, str],
    llm: GenClient,
) -> tuple[list[dict], list[dict]]:
    """Apply all three filter stages. Returns (kept, discarded).

    Adds quality_scores dict to each kept question:
      answerability, self_contained, difficulty_flag
    """
    kept: list[dict] = []
    discarded: list[dict] = []
    kept_embeddings: list[list[float]] = []

    for q in questions:
        doc_id = q.get("source_doc_id", "")
        chunk_text = chunk_text_by_doc_id.get(doc_id, "")
        q_type = q.get("question_type", "")

        # Stage 1 & 2: answerability + self-containment
        # Skip answerability check for out_of_scope (those SHOULD not be answerable from the doc)
        if q_type != "out_of_scope":
            scores = await answerability_score(q, chunk_text, llm)
            if scores["answerability"] < ANSWERABILITY_MIN_SCORE:
                q["discard_reason"] = f"answerability={scores['answerability']}: {scores['filter_reason']}"
                discarded.append(q)
                continue
            if not scores["self_contained"]:
                q["discard_reason"] = f"not self-contained: {scores['filter_reason']}"
                discarded.append(q)
                continue
            q["quality_scores"] = scores
        else:
            q["quality_scores"] = {"answerability": "n/a", "self_contained": True}

        # Stage 3: diversity deduplication
        q_text = q["question"]
        q_embedding = await llm.embed(q_text)

        is_duplicate = any(
            _cosine_sim(q_embedding, kept_emb) > DIVERSITY_SIM_THRESHOLD
            for kept_emb in kept_embeddings
        )
        if is_duplicate:
            q["discard_reason"] = "near-duplicate of an existing question"
            discarded.append(q)
            continue

        kept_embeddings.append(q_embedding)
        kept.append(q)

    return kept, discarded
