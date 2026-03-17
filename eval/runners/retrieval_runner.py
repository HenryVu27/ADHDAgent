"""Evaluate the production retrieval pipeline against the golden dataset.

Imports from app/rag/ intentionally — we are testing the real pipeline,
not a mock. Uses a fresh in-memory Qdrant index so production state is
never touched.

Usage (from project root):
    python -m eval.runners.retrieval_runner [--k 1 3 5] [--types fact_single reasoning]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict

from app.llm.client import GeminiClient
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import FastEmbedReranker
from app.rag.retriever import HybridRetriever
from eval.config import GOLDEN_RETRIEVAL_PATH, RESULTS_DIR
from eval.metrics.retrieval import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run(
    ks: list[int],
    filter_types: list[str] | None,
) -> dict:
    if not GOLDEN_RETRIEVAL_PATH.exists():
        logger.error("Golden dataset not found at %s — run dataset_builder first", GOLDEN_RETRIEVAL_PATH)
        sys.exit(1)

    with open(GOLDEN_RETRIEVAL_PATH) as f:
        dataset = json.load(f)

    if filter_types:
        dataset = [q for q in dataset if q.get("question_type") in filter_types]

    logger.info("Loaded %d questions from golden dataset", len(dataset))

    # Build production retrieval stack with a fresh in-memory Qdrant index
    gemini = GeminiClient()
    store = KnowledgeStore()
    await store.build_index(gemini)

    reranker = FastEmbedReranker()
    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=gemini,
        query_rewriter=QueryRewriter(gemini),
        reranker=reranker,
    )

    logger.info("Retrieval pipeline ready — running eval on %d questions", len(dataset))

    per_query: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_ids = set(item.get("expected_doc_ids", []))
        q_type = item.get("question_type", "unknown")

        if not expected_ids:
            continue

        response = await retriever.retrieve(question)
        retrieved_ids = [r.document_id for r in response.results]

        scores = {
            "question_type": q_type,
            "mrr": mrr(retrieved_ids, expected_ids),
        }
        for k in ks:
            scores[f"recall@{k}"] = recall_at_k(retrieved_ids, expected_ids, k)
            scores[f"precision@{k}"] = precision_at_k(retrieved_ids, expected_ids, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, expected_ids, k)

        per_query.append(scores)
        per_type[q_type].append(scores)

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(dataset))

    # Aggregate
    overall = aggregate(per_query)
    by_type = {t: aggregate(scores) for t, scores in per_type.items()}

    results = {
        "n_questions": len(per_query),
        "k_values": ks,
        "overall": overall,
        "by_question_type": by_type,
    }

    _print_report(results, ks)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"retrieval_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", out_path)

    return results


def _print_report(results: dict, ks: list[int]) -> None:
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL  |  n={results['n_questions']}")
    print("=" * 60)

    def _row(label: str, scores: dict) -> None:
        mrr_val = scores.get("mrr", 0)
        cols = [f"MRR={mrr_val:.3f}"]
        for k in ks:
            r = scores.get(f"recall@{k}", 0)
            p = scores.get(f"precision@{k}", 0)
            n = scores.get(f"ndcg@{k}", 0)
            cols.append(f"R@{k}={r:.3f} P@{k}={p:.3f} NDCG@{k}={n:.3f}")
        print(f"  {label:<22} | {' | '.join(cols)}")

    _row("OVERALL", results["overall"])
    print("-" * 60)
    for q_type, scores in sorted(results["by_question_type"].items()):
        _row(q_type, scores)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval evaluation")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], metavar="K")
    parser.add_argument("--types", nargs="+", default=None, metavar="TYPE",
                        help="Filter to specific question types (e.g. fact_single reasoning)")
    args = parser.parse_args()
    asyncio.run(run(ks=args.k, filter_types=args.types))
