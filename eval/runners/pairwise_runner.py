"""Run pairwise comparison between two pipeline variants.

Expects two directories of recorded conversations with matching conversation IDs.

Usage:
    python -m eval.runners.pairwise_runner --dir-a eval/data/conversations/variant_a --dir-b eval/data/conversations/variant_b
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
from eval.judges.pairwise_judge import PairwiseJudge
from eval.metrics.pairwise import aggregate_pairwise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _match_conversations(convs_a: list[dict], convs_b: list[dict]) -> list[tuple[dict, dict]]:
    """Match conversations by conversation_id."""
    b_by_id = {c["conversation_id"]: c for c in convs_b}
    pairs = []
    for a in convs_a:
        b = b_by_id.get(a["conversation_id"])
        if b:
            pairs.append((a, b))
    return pairs


async def run(dir_a: Path, dir_b: Path, variant_a: str = "A", variant_b: str = "B") -> dict:
    convs_a = load_conversations(dir_a)
    convs_b = load_conversations(dir_b)

    pairs = _match_conversations(convs_a, convs_b)
    if not pairs:
        logger.error("No matching conversation IDs between the two directories.")
        return {}

    logger.info("Matched %d conversation pairs", len(pairs))

    judge = PairwiseJudge()
    all_judgments: list[dict] = []
    per_turn_detail: list[dict] = []

    for conv_a, conv_b in pairs:
        conv_id = conv_a["conversation_id"]
        turns_a = {t["turn"]: t for t in conv_a.get("turns", [])}
        turns_b = {t["turn"]: t for t in conv_b.get("turns", [])}

        common_turns = sorted(set(turns_a.keys()) & set(turns_b.keys()))

        for turn_num in common_turns:
            ta = turns_a[turn_num]
            tb = turns_b[turn_num]

            if ta.get("input_blocked") or tb.get("input_blocked"):
                continue

            result = await judge.compare_turn(
                user_message=ta["user_message"],
                response_a=ta["assistant_response"],
                response_b=tb["assistant_response"],
            )

            if result:
                all_judgments.append(result)
                per_turn_detail.append({"conversation_id": conv_id, "turn": turn_num, **result})

    summary = aggregate_pairwise(all_judgments, variant_a, variant_b)

    _print_report(summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"pairwise_{int(time.time())}"
    out_path = RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps({"summary": summary, "detail": per_turn_detail}, indent=2))
    logger.info("Results saved to %s", out_path)

    conn = await get_eval_connection()
    try:
        await save_eval_run(conn=conn, run_id=run_id, eval_type="pairwise",
                            pipeline_variant=f"{variant_a}_vs_{variant_b}",
                            summary=summary, detail=per_turn_detail,
                            metadata={"variant_a": variant_a, "variant_b": variant_b,
                                      "n_conversations": len(pairs)})
    finally:
        await conn.close()

    return summary


def _print_report(summary: dict) -> None:
    a = summary.get("variant_a", "A")
    b = summary.get("variant_b", "B")
    print("\n" + "=" * 60)
    print(f"PAIRWISE EVAL  |  n={summary.get('n_comparisons', 0)}")
    print(f"  {a} vs {b}")
    print("=" * 60)
    print(f"  Win rate {a:<10}: {summary.get('win_rate_a', 0):.1%}")
    print(f"  Win rate {b:<10}: {summary.get('win_rate_b', 0):.1%}")
    print(f"  Tie rate        : {summary.get('tie_rate', 0):.1%}")
    print(f"  Weighted {a:<7}: {summary.get('weighted_win_rate_a', 0):.1%}")
    print(f"  Weighted {b:<7}: {summary.get('weighted_win_rate_b', 0):.1%}")
    if summary.get("dimension_win_rates"):
        print("-" * 60)
        print("  Per-dimension win rates:")
        for dim, rates in summary["dimension_win_rates"].items():
            print(f"    {dim:<20} {a}={rates.get('A', 0):.0%}  {b}={rates.get('B', 0):.0%}  tie={rates.get('tie', 0):.0%}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pairwise comparison")
    parser.add_argument("--dir-a", type=Path, required=True, help="Conversations dir for variant A")
    parser.add_argument("--dir-b", type=Path, required=True, help="Conversations dir for variant B")
    parser.add_argument("--name-a", default="A", help="Display name for variant A")
    parser.add_argument("--name-b", default="B", help="Display name for variant B")
    args = parser.parse_args()
    asyncio.run(run(dir_a=args.dir_a, dir_b=args.dir_b, variant_a=args.name_a, variant_b=args.name_b))
