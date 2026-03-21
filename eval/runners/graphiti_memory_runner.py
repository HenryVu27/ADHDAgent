"""Evaluate Graphiti memory extraction against the golden conversation dataset.

Ingests each golden conversation into Graphiti via add_episode(), then queries
the graph for extracted entities and compares against the same ground truth
used by the baseline memory_runner.py.

This gives an apples-to-apples comparison: same golden set, same metrics,
different extraction backend (Graphiti graph vs standalone LLM prompt).

Requires: GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in .env

Usage (from project root):
    python -m eval.runners.graphiti_memory_runner [--fields child_name child_age ...]
    python -m eval.runners.graphiti_memory_runner --cleanup  # remove eval data from Neo4j
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.config import GOLDEN_MEMORY_PATH, RESULTS_DIR
from eval.metrics.memory import aggregate, extraction_scores

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Eval uses a dedicated group_id prefix to avoid polluting real user data
_EVAL_GROUP_PREFIX = "eval_memory_"


async def _create_graphiti_client():
    """Create a Graphiti client using env vars (not app config)."""
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.gemini_client import GeminiClient
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

    api_key = os.getenv("GEMINI_API_KEY", "")
    neo4j_uri = os.getenv("NEO4J_URI", "")
    neo4j_user = os.getenv("NEO4J_USER", "")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")

    if not all([api_key, neo4j_uri, neo4j_password]):
        logger.error("Missing required env vars: GEMINI_API_KEY, NEO4J_URI, NEO4J_PASSWORD")
        sys.exit(1)

    # Use the same model as the production Graphiti config
    llm_model = os.getenv("GRAPHITI_LLM_MODEL", os.getenv("GEMINI_UTILITY_MODEL", "gemini-2.5-flash"))
    embedding_model = os.getenv("GRAPHITI_EMBEDDING_MODEL", os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))

    client = Graphiti(
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        llm_client=GeminiClient(
            config=LLMConfig(api_key=api_key, model=llm_model)
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(api_key=api_key, embedding_model=embedding_model)
        ),
    )
    await client.build_indices_and_constraints()
    logger.info("Graphiti client initialized (neo4j=%s, llm=%s)", neo4j_uri, llm_model)
    return client


async def ingest_conversation(client, conversation: dict, group_id: str) -> None:
    """Ingest all turns of a conversation into Graphiti as episodes."""
    from graphiti_core.nodes import EpisodeType
    from app.agent.graphiti_client import ADHD_ENTITY_TYPES, ADHD_EDGE_TYPES

    turns = conversation["turns"]

    # Pair user/assistant turns
    for i in range(0, len(turns) - 1, 2):
        if turns[i]["role"] != "user":
            continue
        user_msg = turns[i]["content"]
        assistant_msg = turns[i + 1]["content"] if i + 1 < len(turns) else ""

        episode_body = f"Parent: {user_msg}\nCoach: {assistant_msg}"
        turn_num = i // 2 + 1

        await client.add_episode(
            name=f"eval_{conversation['id']}_turn_{turn_num}",
            episode_body=episode_body,
            source=EpisodeType.message,
            source_description=f"Eval conversation {conversation['id']}",
            reference_time=datetime.now(),
            group_id=group_id,
            entity_types=ADHD_ENTITY_TYPES,
            edge_types=ADHD_EDGE_TYPES,
        )
        logger.debug("  Ingested turn %d for %s", turn_num, conversation["id"])


async def extract_profile_from_graph(client, group_id: str) -> dict:
    """Query Graphiti for profile-like facts extracted from conversation.

    Uses two approaches:
    1. Search for entity facts via graphiti.search() with profile-oriented queries
    2. Direct Cypher query for entity attributes

    Returns a dict matching the golden dataset fields.
    """
    profile: dict = {}

    # Approach 1: Semantic search for specific profile fields
    field_queries = {
        "child_name": "What is the child's name?",
        "child_age": "How old is the child?",
        "diagnosis_status": "Has the child been diagnosed with ADHD?",
        "adhd_subtype": "What ADHD subtype does the child have?",
        "challenge_areas": "What challenges does the child face?",
        "attempted_strategies": "What strategies has the parent tried?",
        "hardest_situations": "What are the hardest situations for the family?",
        "good_day_description": "What does a good day look like for the family?",
    }

    list_fields = {"challenge_areas", "attempted_strategies", "hardest_situations"}

    for field, query in field_queries.items():
        try:
            edges = await client.search(
                query,
                group_ids=[group_id],
                num_results=5,
            )
            if not edges:
                continue

            facts = [
                getattr(edge, "fact", "")
                for edge in edges
                if getattr(edge, "fact", "") and getattr(edge, "invalid_at", None) is None
            ]

            if not facts:
                continue

            if field in list_fields:
                # Collect all relevant facts as list items
                profile[field] = facts
            else:
                # Use the top fact for scalar fields
                profile[field] = facts[0]

        except Exception as e:
            logger.warning("Search failed for field %s: %s", field, e)

    return profile


async def cleanup_eval_data(client) -> None:
    """Remove all eval-prefixed episodes and their entities from Neo4j."""
    logger.info("Cleaning up eval data (group_id prefix: %s)...", _EVAL_GROUP_PREFIX)
    try:
        query = """
        MATCH (n)
        WHERE n.group_id STARTS WITH $prefix
        DETACH DELETE n
        """
        await client.driver.execute_query(query, prefix=_EVAL_GROUP_PREFIX)
        logger.info("Eval data cleaned up")
    except Exception as e:
        logger.error("Cleanup failed: %s", e)


async def run(filter_fields: list[str] | None, do_cleanup: bool = False) -> dict:
    if not GOLDEN_MEMORY_PATH.exists():
        logger.error("Memory dataset not found at %s", GOLDEN_MEMORY_PATH)
        sys.exit(1)

    client = await _create_graphiti_client()

    if do_cleanup:
        await cleanup_eval_data(client)
        await client.close()
        return {}

    with open(GOLDEN_MEMORY_PATH) as f:
        dataset = json.load(f)

    logger.info("Loaded %d conversations from golden dataset", len(dataset))

    per_conversation: list[dict] = []
    failures: list[dict] = []
    total_ingest_ms = 0
    total_extract_ms = 0

    for i, conv in enumerate(dataset):
        conv_id = conv["id"]
        group_id = f"{_EVAL_GROUP_PREFIX}{conv_id}"

        logger.info("[%d/%d] Processing conversation %s (persona: %s)",
                    i + 1, len(dataset), conv_id, conv.get("persona_id", "?"))

        ground_truth = conv.get("ground_truth_extractions", [])
        if filter_fields:
            ground_truth = [g for g in ground_truth if g["fact_key"] in filter_fields]

        if not ground_truth:
            logger.warning("No ground truth facts for conversation %s -- skipping", conv_id)
            continue

        # Phase 1: Ingest conversation into Graphiti
        try:
            t0 = time.monotonic()
            await ingest_conversation(client, conv, group_id)
            ingest_ms = (time.monotonic() - t0) * 1000
            total_ingest_ms += ingest_ms
            logger.info("  Ingested in %.0fms", ingest_ms)
        except Exception as e:
            logger.error("  Ingestion failed: %s", e)
            failures.append({"conversation_id": conv_id, "phase": "ingest", "error": str(e)})
            continue

        # Phase 2: Extract profile from graph
        try:
            t0 = time.monotonic()
            extracted = await extract_profile_from_graph(client, group_id)
            extract_ms = (time.monotonic() - t0) * 1000
            total_extract_ms += extract_ms
            logger.info("  Extracted in %.0fms: %s", extract_ms, list(extracted.keys()))
        except Exception as e:
            logger.error("  Extraction failed: %s", e)
            failures.append({"conversation_id": conv_id, "phase": "extract", "error": str(e)})
            continue

        # Phase 3: Score against ground truth
        scores = extraction_scores(extracted, ground_truth)
        scores["conversation_id"] = conv_id
        scores["persona_id"] = conv.get("persona_id", "")
        scores["extracted"] = extracted
        scores["ground_truth"] = ground_truth
        scores["ingest_ms"] = round(ingest_ms)
        scores["extract_ms"] = round(extract_ms)
        per_conversation.append(scores)

        logger.info("  P=%.3f R=%.3f F1=%.3f  |  fields: %s",
                    scores["precision"], scores["recall"], scores["f1"],
                    {k: "HIT" if v else "miss" for k, v in scores["per_field"].items()})

    summary = aggregate(per_conversation)
    summary["n_failures"] = len(failures)
    summary["total_ingest_ms"] = round(total_ingest_ms)
    summary["total_extract_ms"] = round(total_extract_ms)
    summary["avg_ingest_ms"] = round(total_ingest_ms / max(len(per_conversation), 1))
    summary["avg_extract_ms"] = round(total_extract_ms / max(len(per_conversation), 1))

    _print_report(summary, per_conversation)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"graphiti_memory_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "per_conversation": per_conversation,
        "failures": failures,
    }, indent=2))
    logger.info("Results saved to %s", out_path)

    await client.close()
    return summary


def _print_report(summary: dict, per_conversation: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"GRAPHITI MEMORY EVAL  |  n={summary.get('n_conversations', 0)}  failures={summary.get('n_failures', 0)}")
    print("=" * 60)
    print(f"  Precision : {summary.get('precision', 0):.4f}")
    print(f"  Recall    : {summary.get('recall', 0):.4f}")
    print(f"  F1        : {summary.get('f1', 0):.4f}")
    print(f"  Avg ingest  : {summary.get('avg_ingest_ms', 0)}ms")
    print(f"  Avg extract : {summary.get('avg_extract_ms', 0)}ms")
    print("-" * 60)
    print("  Per-field recall:")
    for field, rate in sorted(summary.get("per_field_recall", {}).items()):
        bar = "#" * int(rate * 20)
        print(f"    {field:<30} {rate:.3f}  {bar}")
    print("-" * 60)
    if per_conversation:
        worst = sorted(per_conversation, key=lambda x: x["f1"])[:3]
        print("  Lowest F1 conversations:")
        for w in worst:
            print(f"    {w['conversation_id']} ({w['persona_id']})  F1={w['f1']:.3f}")
    print("=" * 60 + "\n")

    # Side-by-side with baseline if available
    baselines = sorted(RESULTS_DIR.glob("memory_*.json"))
    if baselines:
        latest_baseline = baselines[-1]
        try:
            baseline = json.loads(latest_baseline.read_text())["summary"]
            print("COMPARISON vs latest baseline (%s):" % latest_baseline.name)
            print(f"                {'Baseline':>10}  {'Graphiti':>10}  {'Delta':>10}")
            for metric in ("precision", "recall", "f1"):
                b = baseline.get(metric, 0)
                g = summary.get(metric, 0)
                delta = g - b
                sign = "+" if delta >= 0 else ""
                print(f"  {metric:<12} {b:>10.4f}  {g:>10.4f}  {sign}{delta:>9.4f}")
            print()
            print("  Per-field recall comparison:")
            all_fields = sorted(set(baseline.get("per_field_recall", {})) | set(summary.get("per_field_recall", {})))
            for field in all_fields:
                b = baseline.get("per_field_recall", {}).get(field, 0)
                g = summary.get("per_field_recall", {}).get(field, 0)
                delta = g - b
                sign = "+" if delta >= 0 else ""
                print(f"    {field:<30} {b:.3f} -> {g:.3f}  ({sign}{delta:.3f})")
            print("=" * 60 + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Graphiti memory extraction evaluation")
    parser.add_argument("--fields", nargs="+", default=None, metavar="FIELD",
                        help="Evaluate only specific fields (e.g. child_name diagnosis_status)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Remove eval data from Neo4j and exit")
    args = parser.parse_args()
    asyncio.run(run(filter_fields=args.fields, do_cleanup=args.cleanup))
