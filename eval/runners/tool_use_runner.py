"""Run tool use evaluation on recorded conversations.

Usage: python -m eval.runners.tool_use_runner [--dir eval/data/conversations]
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
from eval.judges.tool_use_judge import ToolUseJudge
from eval.metrics.tool_use import aggregate_tool_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run(conversations_dir: Path | None = None) -> dict:
    conversations = load_conversations(conversations_dir)
    if not conversations:
        logger.error("No conversations found.")
        return {}

    judge = ToolUseJudge()
    all_turn_scores: list[dict] = []
    per_turn_detail: list[dict] = []
    failed_turns: list[dict] = []

    for conv in conversations:
        conv_id = conv.get("conversation_id", "unknown")
        logger.info("Evaluating tool use: %s", conv_id)

        history: list[dict] = []
        for turn in conv.get("turns", []):
            if turn.get("input_blocked"):
                continue

            result = await judge.evaluate_turn(
                user_message=turn["user_message"],
                assistant_response=turn["assistant_response"],
                tool_calls=turn.get("tool_calls", []),
                conversation_history=history,
            )

            if result:
                all_turn_scores.append(result)
                per_turn_detail.append({"conversation_id": conv_id, "turn": turn["turn"], **result})
            else:
                failed_turns.append({"conversation_id": conv_id, "turn": turn["turn"]})

            history.append({"role": "user", "content": turn["user_message"]})
            history.append({"role": "assistant", "content": turn["assistant_response"]})

    summary = aggregate_tool_scores(all_turn_scores)
    summary["n_failed"] = len(failed_turns)

    _print_report(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"tool_use_{int(time.time())}"
    out_path = RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps({"summary": summary, "detail": per_turn_detail, "failures": failed_turns}, indent=2))
    logger.info("Results saved to %s", out_path)

    conn = await get_eval_connection()
    try:
        await save_eval_run(conn=conn, run_id=run_id, eval_type="tool_use", pipeline_variant="default",
                            summary=summary, detail=per_turn_detail,
                            metadata={"n_conversations": len(conversations), "failed_turns": failed_turns})
    finally:
        await conn.close()

    return summary


def _print_report(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"TOOL USE EVAL  |  n={summary.get('n_turns', 0)}  failed={summary.get('n_failed', 0)}")
    print("=" * 60)
    print(f"  Precision          : {summary.get('mean_precision', 0):.4f}")
    print(f"  Recall             : {summary.get('mean_recall', 0):.4f}")
    print(f"  F1                 : {summary.get('mean_f1', 0):.4f}")
    print(f"  Argument Accuracy  : {summary.get('mean_argument_accuracy', 0):.2f}")
    print(f"  Result Utilization : {summary.get('mean_result_utilization', 0):.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tool use evaluation")
    parser.add_argument("--dir", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(run(conversations_dir=args.dir))
