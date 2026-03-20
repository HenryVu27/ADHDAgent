"""Compare retrieval quality across chunking strategies.

Builds all three collections (none, recursive_contextual, semantic),
runs the same queries against each, and reports Recall@k and MRR.

Requires GEMINI_API_KEY. Run: ./adhd312/bin/python eval/runners/chunking_comparison.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Test queries with expected document IDs
EVAL_QUERIES = [
    {"query": "How can I help my child with homework?", "expected": ["schoolbased_interventions_for_adhd"]},
    {"query": "daily report card for school", "expected": ["schoolbased_interventions_for_adhd"]},
    {"query": "managing emotions and frustration", "expected": []},
    {"query": "positive reinforcement techniques", "expected": ["parent_training_in_behavior_therapy"]},
    {"query": "what is ADHD", "expected": []},
]


async def evaluate_strategy(strategy: str, queries: list[dict], top_k: int = 5) -> dict:
    """Build index for a strategy and evaluate retrieval quality."""
    os.environ["RAG_CHUNKING_STRATEGY"] = strategy
    os.environ["RAG_CONTEXTUAL_HEADERS"] = "false"  # skip headers for fair comparison

    # Re-import to pick up new settings
    import importlib
    import app.config
    importlib.reload(app.config)

    from app.config import settings
    from app.llm.client import GeminiClient
    from app.rag.knowledge_store import KnowledgeStore
    from app.rag.retriever import HybridRetriever

    gemini = GeminiClient()
    store = KnowledgeStore()
    await store.build_index(gemini)

    retriever = HybridRetriever(knowledge_store=store, gemini_client=gemini)

    results = {"strategy": strategy, "queries": [], "recall_at_k": 0.0, "mrr": 0.0}
    total_recall = 0.0
    total_rr = 0.0
    evaluated = 0

    for q in queries:
        if not q["expected"]:
            continue
        evaluated += 1
        response = await retriever.retrieve(q["query"], top_k=top_k, skip_rewrite=True)
        retrieved_ids = [r.document_id for r in response.results]

        # Recall@k
        hits = sum(1 for eid in q["expected"] if eid in retrieved_ids)
        recall = hits / len(q["expected"])
        total_recall += recall

        # MRR
        rr = 0.0
        for eid in q["expected"]:
            if eid in retrieved_ids:
                rank = retrieved_ids.index(eid) + 1
                rr = max(rr, 1.0 / rank)
        total_rr += rr

        results["queries"].append({
            "query": q["query"],
            "expected": q["expected"],
            "retrieved": retrieved_ids[:top_k],
            "recall": recall,
            "rr": rr,
        })

    if evaluated > 0:
        results["recall_at_k"] = total_recall / evaluated
        results["mrr"] = total_rr / evaluated

    logger.info(
        "Strategy=%s  Recall@%d=%.3f  MRR=%.3f",
        strategy, top_k, results["recall_at_k"], results["mrr"],
    )
    return results


async def main():
    strategies = ["none", "recursive_contextual", "semantic"]
    all_results = []

    for strategy in strategies:
        logger.info("--- Evaluating strategy: %s ---", strategy)
        result = await evaluate_strategy(strategy, EVAL_QUERIES)
        all_results.append(result)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Strategy':<25} {'Recall@5':<12} {'MRR':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['recall_at_k']:<12.3f} {r['mrr']:<12.3f}")
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "results" / "chunking_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
