"""Extract factual statements from document chunks.

Statements are the atomic ground-truth units used for question generation.
Generating questions FROM statements (rather than directly from chunks)
eliminates answerability issues — the answer exists before the question
is written.
"""
from __future__ import annotations

import logging

from eval.generators.llm import GenClient

logger = logging.getLogger(__name__)

_PROMPT = """\
You are extracting factual statements from an ADHD coaching knowledge document.

Document name: {name}
Document text:
{text}

Extract {n} distinct factual statements from this document. Each statement must:
- Be self-contained (readable without access to the document)
- Capture a single fact, recommendation, or behavioral insight
- Be specific enough that a non-trivial question can be built from it
- Cover different sections or aspects of the document (not all from the first paragraph)
- NOT be a generic truism (e.g., "ADHD is a common condition" is too vague)

Return a JSON array of strings. Example format:
["Children with ADHD often experience time-blindness, making it hard to estimate how long tasks will take.",
 "The Pomodoro technique uses 25-minute focused work intervals followed by 5-minute breaks to reduce overwhelm."]
"""


async def extract_statements(
    chunk: dict,
    llm: GenClient,
    n: int = 4,
) -> list[str]:
    """Return n factual statements extracted from chunk text."""
    prompt = _PROMPT.format(
        name=chunk["document_name"],
        text=chunk["text"],
        n=n,
    )
    result = await llm.json(prompt, temperature=0.3, max_tokens=2048)

    # Model sometimes wraps the array: {"statements": [...]} or {"items": [...]}
    if isinstance(result, dict):
        for key in ("statements", "items", "facts", "extractions"):
            if isinstance(result.get(key), list):
                result = result[key]
                break

    if isinstance(result, list):
        statements = [s for s in result if isinstance(s, str) and len(s.strip()) > 20]
        return statements[:n]

    logger.warning("Statement extractor returned unexpected type for doc %s", chunk["document_id"])
    return []
