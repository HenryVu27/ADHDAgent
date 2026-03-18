"""CLI: build both golden datasets from scratch.

Usage (from project root):
    python -m eval.generators.dataset_builder [--retrieval] [--memory] [--all]
    python -m eval.generators.dataset_builder --all --sample  # small representative subset

Generates:
    eval/data/golden_retrieval.json   — (question, expected_doc_id, ...) pairs
    eval/data/golden_memory.json      — synthetic conversations with ground-truth facts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
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

# Sample mode: 4 docs per knowledge file (stratified), 2 questions/statement, 1 statement/chunk
_SAMPLE_DOCS_PER_FILE = 4
_SAMPLE_STATEMENTS_PER_CHUNK = 3
_SAMPLE_QUESTIONS_PER_STATEMENT = 2
_SAMPLE_PERSONAS = 10
_SAMPLE_CONVERSATIONS_PER_PERSONA = 2


def _sample_chunks_stratified(chunks: list[dict], docs_per_file: int) -> list[dict]:
    """Pick docs_per_file chunks from each source file, covering all doc types."""
    by_file: dict[str, list[dict]] = {}
    for c in chunks:
        # Group by document_type as a proxy for knowledge file
        key = c.get("document_type", "unknown")
        by_file.setdefault(key, []).append(c)

    sampled = []
    for group in by_file.values():
        random.shuffle(group)
        sampled.extend(group[:docs_per_file])
    return sampled


def _sample_personas_stratified(personas: list[dict], n: int) -> list[dict]:
    """Pick n personas covering diverse diagnosis_status and adhd_subtype values."""
    by_status: dict[str, list[dict]] = {}
    for p in personas:
        key = p["facts"].get("diagnosis_status", "unknown")
        by_status.setdefault(key, []).append(p)

    # Round-robin across status groups until we have n
    selected = []
    groups = list(by_status.values())
    for group in groups:
        random.shuffle(group)
    i = 0
    while len(selected) < n and any(groups):
        group = groups[i % len(groups)]
        if group:
            selected.append(group.pop())
        i += 1
    return selected[:n]


async def build_retrieval_dataset(sample: bool = False) -> list[dict]:
    """Generate golden retrieval dataset.

    Pipeline per chunk:
      1. Extract statements (STATEMENTS_PER_CHUNK or _SAMPLE_STATEMENTS_PER_CHUNK)
      2. For each statement, generate questions with type sampled from QUESTION_TYPE_WEIGHTS
      3. Quality filter (answerability + diversity)
      4. Assign stable IDs

    Saves incrementally after each chunk so partial results survive interruption.
    """
    llm = GenClient()
    all_chunks = load_chunks()
    logger.info("Loaded %d chunks from knowledge base", len(all_chunks))

    if sample:
        chunks = _sample_chunks_stratified(all_chunks, _SAMPLE_DOCS_PER_FILE)
        statements_per_chunk = _SAMPLE_STATEMENTS_PER_CHUNK
        questions_per_statement = _SAMPLE_QUESTIONS_PER_STATEMENT
        logger.info("SAMPLE mode: using %d chunks", len(chunks))
    else:
        chunks = all_chunks
        statements_per_chunk = STATEMENTS_PER_CHUNK
        questions_per_statement = QUESTIONS_PER_STATEMENT

    chunk_text_by_doc_id = {c["document_id"]: c["text"] for c in all_chunks}
    raw_questions: list[dict] = []

    for i, chunk in enumerate(chunks):
        logger.info("[%d/%d] Extracting statements from: %s", i + 1, len(chunks), chunk["document_id"])
        statements = await extract_statements(chunk, llm, n=statements_per_chunk)

        if not statements:
            logger.warning("No statements extracted for %s — skipping", chunk["document_id"])
            continue

        for statement in statements:
            for _ in range(questions_per_statement):
                q_type = sample_question_type(QUESTION_TYPE_WEIGHTS)
                question = await generate_question(statement, q_type, chunk, llm)
                if question:
                    raw_questions.append(question)

        # Incremental save after each chunk
        _save_raw(raw_questions, GOLDEN_RETRIEVAL_PATH)

    logger.info("Generated %d raw questions — running quality filter", len(raw_questions))
    kept, discarded = await filter_questions(raw_questions, chunk_text_by_doc_id, llm)
    logger.info("After filtering: %d kept, %d discarded", len(kept), len(discarded))

    dataset = []
    for q in kept:
        q["id"] = f"q_{uuid.uuid4().hex[:8]}"
        dataset.append(q)

    type_counts: dict[str, int] = {}
    for q in dataset:
        t = q.get("question_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info("Question type distribution: %s", type_counts)

    GOLDEN_RETRIEVAL_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    logger.info("Saved %d questions to %s", len(dataset), GOLDEN_RETRIEVAL_PATH)
    return dataset


def _save_raw(items: list[dict], path) -> None:
    """Write current items to disk (unfiltered checkpoint)."""
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False))


async def build_memory_dataset(sample: bool = False) -> list[dict]:
    """Generate golden memory dataset.

    For each persona, generate conversations using the Generator-Critic pattern.
    Saves incrementally after each conversation so partial results survive interruption.
    """
    llm = GenClient()
    personas_path = DATA_DIR / "personas.json"
    with open(personas_path) as f:
        all_personas = json.load(f)

    if sample:
        personas = _sample_personas_stratified(all_personas, _SAMPLE_PERSONAS)
        conversations_per_persona = _SAMPLE_CONVERSATIONS_PER_PERSONA
        logger.info("SAMPLE mode: using %d personas, %d convos each", len(personas), conversations_per_persona)
    else:
        personas = all_personas
        conversations_per_persona = CONVERSATIONS_PER_PERSONA

    logger.info("Loaded %d personas", len(personas))
    conversations: list[dict] = []

    for persona in personas:
        logger.info("Generating conversations for persona: %s", persona["id"])
        for attempt in range(conversations_per_persona):
            logger.info("  Conversation %d/%d", attempt + 1, conversations_per_persona)
            conv = await generate_conversation(persona, llm)
            if conv:
                conversations.append(conv)
                # Incremental save after each successful conversation
                GOLDEN_MEMORY_PATH.write_text(json.dumps(conversations, indent=2, ensure_ascii=False))
            else:
                logger.warning("  Failed to generate valid conversation for %s", persona["id"])
            await asyncio.sleep(1)

    logger.info("Generated %d conversations total", len(conversations))
    total_facts = sum(len(c["ground_truth_extractions"]) for c in conversations)
    logger.info("Total ground-truth fact annotations: %d", total_facts)

    GOLDEN_MEMORY_PATH.write_text(json.dumps(conversations, indent=2, ensure_ascii=False))
    logger.info("Saved %d conversations to %s", len(conversations), GOLDEN_MEMORY_PATH)
    return conversations


async def main(build_retrieval: bool, build_memory: bool, sample: bool) -> None:
    start = time.monotonic()

    if build_retrieval:
        logger.info("=== Building retrieval dataset (sample=%s) ===", sample)
        await build_retrieval_dataset(sample=sample)

    if build_memory:
        logger.info("=== Building memory dataset (sample=%s) ===", sample)
        await build_memory_dataset(sample=sample)

    elapsed = time.monotonic() - start
    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build golden evaluation datasets")
    parser.add_argument("--retrieval", action="store_true", help="Build retrieval golden dataset")
    parser.add_argument("--memory", action="store_true", help="Build memory golden dataset")
    parser.add_argument("--all", action="store_true", help="Build both datasets")
    parser.add_argument("--sample", action="store_true",
                        help="Build a small representative subset (~20 docs, 10 personas)")
    args = parser.parse_args()

    if not (args.retrieval or args.memory or args.all):
        parser.print_help()
        raise SystemExit(1)

    asyncio.run(main(
        build_retrieval=args.retrieval or args.all,
        build_memory=args.memory or args.all,
        sample=args.sample,
    ))
