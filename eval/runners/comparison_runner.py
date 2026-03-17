"""Run all 6 ablation pipeline configs against the golden retrieval dataset.

Builds 4 distinct KnowledgeStore indexes once (keyed by sparse_mode x colbert),
then wires each config and runs the eval loop. Prints a side-by-side table.

Usage (from project root):
    python -m eval.runners.comparison_runner [--k 1 3 5] [--types fact_single reasoning]
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
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import FastEmbedReranker
from app.rag.retriever import HybridRetriever
from eval.config import GOLDEN_RETRIEVAL_PATH, RESULTS_DIR
from eval.metrics.retrieval import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Collection names — one per (sparse_mode, use_colbert) combination
_COLLECTION_NAMES: dict[tuple[str, bool], str] = {
    ("tfidf", False): "rag_tfidf",
    ("bm25",  False): "rag_bm25",
    ("tfidf", True):  "rag_tfidf_colbert",
    ("bm25",  True):  "rag_bm25_colbert",
}


def _index_key(cfg: PipelineConfig) -> tuple[str, bool]:
    return (cfg.sparse_mode, cfg.use_colbert)


async def _build_indexes(
    gemini: GeminiClient,
    configs: list[PipelineConfig],
) -> tuple[dict[tuple[str, bool], KnowledgeStore], "ColBERTIndex | None"]:
    """Build one KnowledgeStore per unique (sparse_mode, use_colbert) combination.

    Returns (stores, colbert_instance). colbert_instance is None if no config
    uses ColBERT. The same instance is reused across all colbert-enabled configs
    to avoid loading the 500 MB model multiple times.
    """
    from app.rag.colbert_index import ColBERTIndex

    needed_keys = {_index_key(cfg) for cfg in configs}
    stores: dict[tuple[str, bool], KnowledgeStore] = {}

    # Instantiate ColBERTIndex once — shared across colbert-enabled configs
    colbert: ColBERTIndex | None = None
    if any(key[1] for key in needed_keys):
        logger.info("Loading ColBERT model (first run downloads ~500 MB)...")
        colbert = ColBERTIndex()

    for key in needed_keys:
        sparse_mode, use_colbert = key
        collection_name = _COLLECTION_NAMES[key]
        logger.info("Building index: sparse=%s colbert=%s collection=%s",
                    sparse_mode, use_colbert, collection_name)
        store = KnowledgeStore(sparse_mode=sparse_mode, collection_name=collection_name)
        await store.build_index(gemini, colbert_index=colbert if use_colbert else None)
        stores[key] = store

    return stores, colbert


async def _run_config(
    retriever: HybridRetriever,
    dataset: list[dict],
    ks: list[int],
) -> dict:
    """Run the eval loop for one pipeline config. Returns aggregated results."""
    per_query: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)

    # Minimal stub state so query rewriter condition is satisfied when enabled
    stub_state = SessionState(session_id="eval", conversation_history=[
        {"role": "user", "content": "stub"}
    ])

    for item in dataset:
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

    return {
        "overall": aggregate(per_query),
        "by_question_type": {t: aggregate(s) for t, s in per_type.items()},
        "n_questions": len(per_query),
    }


async def run(ks: list[int], filter_types: list[str] | None) -> dict:
    if not GOLDEN_RETRIEVAL_PATH.exists():
        logger.error("Golden dataset not found at %s — run dataset_builder first",
                     GOLDEN_RETRIEVAL_PATH)
        sys.exit(1)

    with open(GOLDEN_RETRIEVAL_PATH) as f:
        dataset = json.load(f)
    if filter_types:
        dataset = [q for q in dataset if q.get("question_type") in filter_types]
    logger.info("Loaded %d questions from golden dataset", len(dataset))

    # Force query rewriting to be usable (runner controls on/off via QueryRewriter instance)
    settings.RAG_USE_QUERY_REWRITE = True

    gemini = GeminiClient()
    reranker = FastEmbedReranker()
    query_rewriter = QueryRewriter(gemini)
    stores, shared_colbert = await _build_indexes(gemini, ABLATION_CONFIGS)

    all_results: dict[str, dict] = {}

    for cfg in ABLATION_CONFIGS:
        logger.info("Running config: %s", cfg.name)
        store = stores[_index_key(cfg)]

        # Reuse the single ColBERTIndex instance loaded during _build_indexes.
        # Do NOT instantiate a new ColBERTIndex here — that would load the 500 MB
        # model a second time.
        colbert = shared_colbert if cfg.use_colbert else None

        retriever = HybridRetriever(
            knowledge_store=store,
            gemini_client=gemini,
            query_rewriter=query_rewriter if cfg.use_query_rewriter else None,
            reranker=reranker if cfg.use_reranker else None,
            colbert_index=colbert,
        )

        result = await _run_config(retriever, dataset, ks)
        all_results[cfg.name] = result
        logger.info("Config '%s' done — MRR=%.3f", cfg.name, result["overall"].get("mrr", 0))

    _print_report(all_results, ks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"comparison_{int(time.time())}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _print_report(results: dict[str, dict], ks: list[int]) -> None:
    header_cols = ["MRR"] + [f"R@{k}" for k in ks] + [f"NDCG@{k}" for k in ks]
    col_w = 7
    name_w = 14

    print("\n" + "=" * (name_w + len(header_cols) * (col_w + 3) + 2))
    print(f"{'Pipeline':<{name_w}} | " + " | ".join(f"{h:>{col_w}}" for h in header_cols))
    print("-" * (name_w + len(header_cols) * (col_w + 3) + 2))

    for name, data in results.items():
        overall = data.get("overall", {})
        vals = [overall.get("mrr", 0)]
        for k in ks:
            vals.append(overall.get(f"recall@{k}", 0))
        for k in ks:
            vals.append(overall.get(f"ndcg@{k}", 0))
        row = " | ".join(f"{v:>{col_w}.3f}" for v in vals)
        print(f"{name:<{name_w}} | {row}")

    print("=" * (name_w + len(header_cols) * (col_w + 3) + 2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RAG pipeline ablation configs")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], metavar="K")
    parser.add_argument("--types", nargs="+", default=None, metavar="TYPE")
    args = parser.parse_args()
    asyncio.run(run(ks=args.k, filter_types=args.types))
