"""CLI: build both golden datasets from scratch.

Usage (from project root):
    python -m eval.generators.dataset_builder [--retrieval] [--memory] [--all]

Generates:
    eval/data/golden_retrieval.json   — (question, expected_doc_id, ...) pairs
    eval/data/golden_memory.json      — synthetic conversations with ground-truth facts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid

from eval.config import (
    CONVERSATIONS_PER_PERSONA,
    DATA_DIR,
    GOLDEN_MEMORY_PATH,
    GOLDEN_RETRIEVAL_PATH,
    QUESTION_TYPE_WEIGHTS,
    QUESTIONS_PER_STATEMENT,
    STATEMENTS_PER_CHUNK,
)
from eval.generators.conversation_generator import generate_conversation
from eval.generators.corpus import load_chunks
from eval.generators.llm import GenClient
from eval.generators.quality_filter import filter_questions
from eval.generators.question_generator import generate_question, sample_question_type
from eval.generators.statement_extractor import extract_statements

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def build_retrieval_dataset() -> list[dict]:
    """Generate golden retrieval dataset.

    Pipeline per chunk:
      1. Extract STATEMENTS_PER_CHUNK factual statements
      2. For each statement, generate QUESTIONS_PER_STATEMENT questions
         with question type sampled from QUESTION_TYPE_WEIGHTS
      3. Quality filter (answerability + diversity)
      4. Assign stable IDs
    """
    llm = GenClient()
    chunks = load_chunks()
    logger.info("Loaded %d chunks from knowledge base", len(chunks))

    chunk_text_by_doc_id = {c["document_id"]: c["text"] for c in chunks}
    raw_questions: list[dict] = []

    for i, chunk in enumerate(chunks):
        logger.info("[%d/%d] Extracting statements from: %s", i + 1, len(chunks), chunk["document_id"])
        statements = await extract_statements(chunk, llm, n=STATEMENTS_PER_CHUNK)

        if not statements:
            logger.warning("No statements extracted for %s — skipping", chunk["document_id"])
            continue

        for statement in statements:
            for _ in range(QUESTIONS_PER_STATEMENT):
                q_type = sample_question_type(QUESTION_TYPE_WEIGHTS)
                question = await generate_question(statement, q_type, chunk, llm)
                if question:
                    raw_questions.append(question)

    logger.info("Generated %d raw questions — running quality filter", len(raw_questions))
    kept, discarded = await filter_questions(raw_questions, chunk_text_by_doc_id, llm)
    logger.info("After filtering: %d kept, %d discarded", len(kept), len(discarded))

    # Assign stable IDs and finalize
    dataset = []
    for q in kept:
        q["id"] = f"q_{uuid.uuid4().hex[:8]}"
        dataset.append(q)

    # Summary by question type
    type_counts: dict[str, int] = {}
    for q in dataset:
        t = q.get("question_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info("Question type distribution: %s", type_counts)

    GOLDEN_RETRIEVAL_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    logger.info("Saved %d questions to %s", len(dataset), GOLDEN_RETRIEVAL_PATH)
    return dataset


async def build_memory_dataset() -> list[dict]:
    """Generate golden memory dataset.

    For each persona, generate CONVERSATIONS_PER_PERSONA validated conversations
    using the Generator-Critic pattern.
    """
    llm = GenClient()
    personas_path = DATA_DIR / "personas.json"
    with open(personas_path) as f:
        personas = json.load(f)

    logger.info("Loaded %d personas", len(personas))
    conversations: list[dict] = []

    for persona in personas:
        logger.info("Generating conversations for persona: %s", persona["id"])
        for attempt in range(CONVERSATIONS_PER_PERSONA):
            logger.info("  Conversation %d/%d", attempt + 1, CONVERSATIONS_PER_PERSONA)
            conv = await generate_conversation(persona, llm)
            if conv:
                conversations.append(conv)
            else:
                logger.warning("  Failed to generate valid conversation for %s", persona["id"])
            # Small delay to avoid rate limits
            await asyncio.sleep(1)

    logger.info("Generated %d conversations total", len(conversations))

    # Summary
    total_facts = sum(len(c["ground_truth_extractions"]) for c in conversations)
    logger.info("Total ground-truth fact annotations: %d", total_facts)

    GOLDEN_MEMORY_PATH.write_text(json.dumps(conversations, indent=2, ensure_ascii=False))
    logger.info("Saved %d conversations to %s", len(conversations), GOLDEN_MEMORY_PATH)
    return conversations


async def main(build_retrieval: bool, build_memory: bool) -> None:
    start = time.monotonic()

    if build_retrieval:
        logger.info("=== Building retrieval dataset ===")
        await build_retrieval_dataset()

    if build_memory:
        logger.info("=== Building memory dataset ===")
        await build_memory_dataset()

    elapsed = time.monotonic() - start
    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build golden evaluation datasets")
    parser.add_argument("--retrieval", action="store_true", help="Build retrieval golden dataset")
    parser.add_argument("--memory", action="store_true", help="Build memory golden dataset")
    parser.add_argument("--all", action="store_true", help="Build both datasets")
    args = parser.parse_args()

    if not (args.retrieval or args.memory or args.all):
        parser.print_help()
        raise SystemExit(1)

    asyncio.run(main(
        build_retrieval=args.retrieval or args.all,
        build_memory=args.memory or args.all,
    ))
