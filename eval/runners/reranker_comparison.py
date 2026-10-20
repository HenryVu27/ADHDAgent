"""Compare cross-encoder vs ColBERT reranking on the golden retrieval dataset.

Builds 2 indexes (tfidf, tfidf+colbert), runs both configs, prints side-by-side.

Usage:
    python -m eval.runners.reranker_comparison [--k 1 3 5]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict

from app.config import settings
from app.llm.client import GeminiClient
from app.models.schemas import SessionState
from app.rag.colbert_index import ColBERTIndex
from app.rag.knowledge_store import KnowledgeStore
from app.rag.reranker import FastEmbedReranker
from app.rag.retriever import HybridRetriever
from eval.config import GOLDEN_RETRIEVAL_PATH, RESULTS_DIR
from eval.metrics.retrieval import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _run_eval(
    retriever: HybridRetriever,
    dataset: list[dict],
    ks: list[int],
) -> dict:
    per_query: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)

    stub_state = SessionState(session_id="eval", conversation_history=[
        {"role": "user", "content": "stub"}
    ])

    for i, item in enumerate(dataset):
        question = item["question"]
        expected_ids = set(item.get("expected_doc_ids", []))
        q_type = item.get("question_type", "unknown")
        if not expected_ids:
            continue

        response = await retriever.retrieve(question, state=stub_state)
        retrieved_ids = [r.document_id for r in response.results]

        scores: dict = {"question_type": q_type, "mrr": mrr(retrieved_ids, expected_ids)}
        for k in ks:
            scores[f"recall@{k}"] = recall_at_k(retrieved_ids, expected_ids, k)
            scores[f"precision@{k}"] = precision_at_k(retrieved_ids, expected_ids, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, expected_ids, k)

        per_query.append(scores)
        per_type[q_type].append(scores)

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d", i + 1, len(dataset))

    return {
        "overall": aggregate(per_query),
        "by_question_type": {t: aggregate(s) for t, s in per_type.items()},
        "n_questions": len(per_query),
    }


async def run(ks: list[int]) -> dict:
    if not GOLDEN_RETRIEVAL_PATH.exists():
        logger.error("Golden dataset not found at %s", GOLDEN_RETRIEVAL_PATH)
        sys.exit(1)

    with open(GOLDEN_RETRIEVAL_PATH) as f:
        dataset = json.load(f)
    # Skip out_of_scope — they have no expected docs
    dataset = [q for q in dataset if q.get("expected_doc_ids")]
    logger.info("Loaded %d questions (with expected docs)", len(dataset))

    gemini = GeminiClient()

    # --- Build indexes ---
    # 1) Standard index (no ColBERT) — shared by cross-encoder config
    logger.info("Building standard index (dense + sparse)...")
    store_standard = KnowledgeStore(collection_name="eval_standard")
    await store_standard.build_index(gemini)

    # 2) ColBERT-enabled index
    logger.info("Loading ColBERT model...")
    colbert = ColBERTIndex()
    logger.info("Building ColBERT index (dense + sparse + colbert)...")
    store_colbert = KnowledgeStore(collection_name="eval_colbert")
    await store_colbert.build_index(gemini, colbert_index=colbert)

    reranker = FastEmbedReranker()

    # --- Config 1: Cross-encoder reranking (no ColBERT) ---
    logger.info("=" * 60)
    logger.info("Running: cross-encoder reranking")
    retriever_ce = HybridRetriever(
        knowledge_store=store_standard,
        gemini_client=gemini,
        reranker=reranker,
    )
    results_ce = await _run_eval(retriever_ce, dataset, ks)
    logger.info("Cross-encoder done — MRR=%.3f", results_ce["overall"].get("mrr", 0))

    # --- Config 2: ColBERT late-interaction (no cross-encoder reranker) ---
    logger.info("=" * 60)
    logger.info("Running: ColBERT late-interaction")
    retriever_colbert = HybridRetriever(
        knowledge_store=store_colbert,
        gemini_client=gemini,
        colbert_index=colbert,
    )
    results_colbert = await _run_eval(retriever_colbert, dataset, ks)
    logger.info("ColBERT done — MRR=%.3f", results_colbert["overall"].get("mrr", 0))

    all_results = {
        "cross_encoder": results_ce,
        "colbert": results_colbert,
    }

    _print_report(all_results, ks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"reranker_comparison_{int(time.time())}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _print_report(results: dict[str, dict], ks: list[int]) -> None:
    header_cols = ["MRR"] + [f"R@{k}" for k in ks] + [f"NDCG@{k}" for k in ks]
    col_w = 8
    name_w = 16

    print("\n" + "=" * (name_w + len(header_cols) * (col_w + 3) + 2))
    print("RERANKER COMPARISON")
    print("=" * (name_w + len(header_cols) * (col_w + 3) + 2))
    print(f"{'Config':<{name_w}} | " + " | ".join(f"{h:>{col_w}}" for h in header_cols))
    print("-" * (name_w + len(header_cols) * (col_w + 3) + 2))

    for name, data in results.items():
        overall = data.get("overall", {})
        vals = [overall.get("mrr", 0)]
        for k in ks:
            vals.append(overall.get(f"recall@{k}", 0))
        for k in ks:
            vals.append(overall.get(f"ndcg@{k}", 0))
        row = " | ".join(f"{v:>{col_w}.4f}" for v in vals)
        print(f"{name:<{name_w}} | {row}")

    print("=" * (name_w + len(header_cols) * (col_w + 3) + 2))

    # Per-type breakdown
    all_types = set()
    for data in results.values():
        all_types.update(data.get("by_question_type", {}).keys())

    for q_type in sorted(all_types):
        print(f"\n  {q_type}:")
        for name, data in results.items():
            type_scores = data.get("by_question_type", {}).get(q_type, {})
            mrr_val = type_scores.get("mrr", 0)
            r1 = type_scores.get("recall@1", 0) if 1 in ks else 0
            print(f"    {name:<16} MRR={mrr_val:.4f}  R@1={r1:.4f}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare cross-encoder vs ColBERT reranking")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], metavar="K")
    args = parser.parse_args()
    asyncio.run(run(ks=args.k))
