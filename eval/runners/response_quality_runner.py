"""Run response quality evaluation on recorded conversations.

Usage (from project root):
    python -m eval.runners.response_quality_runner [--dir eval/data/conversations]
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
from eval.judges.response_judge import ResponseJudge
from eval.metrics.response_quality import aggregate_turn_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run(conversations_dir: Path | None = None) -> dict:
    conversations = load_conversations(conversations_dir)
    if not conversations:
        logger.error("No conversations found. Record some first with recorder.py")
        return {}

    judge = ResponseJudge()
    all_turn_scores: list[dict] = []
    per_turn_detail: list[dict] = []
    failed_turns: list[dict] = []

    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        logger.info("Evaluating conversation: %s", conv_id)

        history: list[dict] = []
        for turn in conv.get("turns", []):
            if turn.get("input_blocked"):
                continue

            result = await judge.score_turn(
                user_message=turn["user_message"],
                assistant_response=turn["assistant_response"],
                tool_calls=turn.get("tool_calls", []),
                conversation_history=history,
            )

            if result:
                all_turn_scores.append(result["scores"])
                per_turn_detail.append({
                    "conversation_id": conv_id,
                    "turn": turn["turn"],
                    **result,
                })
            else:
                failed_turns.append({"conversation_id": conv_id, "turn": turn["turn"]})

            # Accumulate history for subsequent turns
            history.append({"role": "user", "content": turn["user_message"]})
            history.append({"role": "assistant", "content": turn["assistant_response"]})

    summary = aggregate_turn_scores(all_turn_scores)
    summary["n_failed"] = len(failed_turns)

    _print_report(summary)

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"response_quality_{int(time.time())}"
    out_path = RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps({"summary": summary, "detail": per_turn_detail, "failures": failed_turns}, indent=2))
    logger.info("Results saved to %s", out_path)

    # Save to eval DB
    conn = await get_eval_connection()
    try:
        await save_eval_run(
            conn=conn,
            run_id=run_id,
            eval_type="response_quality",
            pipeline_variant="default",
            summary=summary,
            detail=per_turn_detail,
            metadata={"n_conversations": len(conversations), "failed_turns": failed_turns},
        )
    finally:
        await conn.close()

    return summary


def _print_report(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"RESPONSE QUALITY EVAL  |  n={summary.get('n_turns', 0)}  failed={summary.get('n_failed', 0)}")
    print("=" * 60)
    print(f"{'Dimension':<25} {'Mean':>6} {'Median':>8} {'Min':>6} ")
    print("-" * 60)
    for dim in summary.get("mean", {}):
        mean = summary["mean"].get(dim, 0)
        median = summary.get("median", {}).get(dim, 0)
        minimum = summary.get("min", {}).get(dim, 0)
        print(f"  {dim:<23} {mean:>6.2f} {median:>8.1f} {minimum:>6}")
    print("-" * 60)
    print(f"  {'Overall':<23} {summary.get('overall', 0):>6.2f}")
    print(f"  Flagged turns: {summary.get('flagged_turns', 0)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run response quality evaluation")
    parser.add_argument("--dir", type=Path, default=None, help="Conversations directory")
    args = parser.parse_args()
    asyncio.run(run(conversations_dir=args.dir))
