"""Run conversation-level coherence evaluation.

Usage: python -m eval.runners.coherence_runner [--dir eval/data/conversations]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from eval.config import RESULTS_DIR
from eval.db import get_eval_connection, save_eval_run
from eval.judges.base import load_conversations
from eval.judges.coherence_judge import CoherenceJudge
from eval.metrics.coherence import aggregate_coherence_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run(conversations_dir: Path | None = None) -> dict:
    conversations = load_conversations(conversations_dir)
    if not conversations:
        logger.error("No conversations found.")
        return {}

    judge = CoherenceJudge()
    all_scores: list[dict] = []
    per_conv_detail: list[dict] = []
    skipped: list[str] = []

    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        turns = [t for t in conv.get("turns", []) if not t.get("input_blocked")]

        result = await judge.score_conversation(
            turns=turns,
            family_profile=conv.get("metadata", {}).get("family_profile", {}),
        )

        if result is None:
            skipped.append(conv_id)
            logger.info("Skipped %s (too short: %d turns)", conv_id, len(turns))
            continue

        all_scores.append(result["scores"])
        per_conv_detail.append({"conversation_id": conv_id, **result})
        logger.info("  %s: overall=%.2f", conv_id, result["overall"])

    summary = aggregate_coherence_scores(all_scores)
    summary["n_skipped"] = len(skipped)

    _print_report(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"coherence_{int(time.time())}"
    out_path = RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps({"summary": summary, "detail": per_conv_detail, "skipped": skipped}, indent=2))
    logger.info("Results saved to %s", out_path)

    conn = await get_eval_connection()
    try:
        await save_eval_run(conn=conn, run_id=run_id, eval_type="coherence", pipeline_variant="default",
                            summary=summary, detail=per_conv_detail,
                            metadata={"n_conversations": len(conversations), "skipped": skipped})
    finally:
        await conn.close()

    return summary


def _print_report(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"COHERENCE EVAL  |  n={summary.get('n_conversations', 0)}  skipped={summary.get('n_skipped', 0)}")
    print("=" * 60)
    for dim in summary.get("mean", {}):
        mean = summary["mean"].get(dim, 0)
        minimum = summary.get("min", {}).get(dim, 0)
        print(f"  {dim:<30} mean={mean:.2f}  min={minimum}")
    print("-" * 60)
    print(f"  {'Overall':<30} {summary.get('overall', 0):.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run coherence evaluation")
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(run(conversations_dir=args.dir))
