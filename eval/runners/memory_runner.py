"""Evaluate fact extraction against the golden conversation dataset.

The fact extraction prompt is replicated here (not imported from app/)
so we test the extraction behavior in isolation. If the production prompt
changes, the delta shows up as a metric shift — intentional signal.

Usage (from project root):
    python -m eval.runners.memory_runner [--fields child_name child_age ...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

from eval.config import GOLDEN_MEMORY_PATH, RESULTS_DIR
from eval.generators.llm import GenClient
from eval.metrics.memory import aggregate, extraction_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Replicated from app/agent/memory.py — intentional copy so eval is stable
# regardless of production prompt changes. Update here explicitly when you
# want to test a new version of the extraction logic.
_EXTRACTION_PROMPT = """\
Extract any family profile facts from this parent's message about their child with ADHD.
Return a JSON object with only the fields that are explicitly mentioned or clearly implied. Valid fields:
- child_name (string)
- child_age (string, e.g. "7" or "8-9" if ambiguous)
- diagnosis_status (string: "diagnosed", "suspected", "evaluating", "not diagnosed")
- adhd_subtype (string: "inattentive", "hyperactive-impulsive", "combined")
- challenge_areas (list of strings)
- attempted_strategies (list of strings)
- good_day_description (string)
- hardest_situations (list of strings)

If the parent corrects previously shared information, extract the CORRECTED value.
If information is ambiguous, use the parent's phrasing (e.g., "about 8 or 9" -> "8-9").
If no NEW profile facts are mentioned, return an empty object {{}}.

Already known: {known_facts}
{history_block}

Parent message:
<parent_message>
{user_message}
</parent_message>
"""


async def extract_facts_from_conversation(
    conversation: dict,
    llm: GenClient,
) -> dict:
    """Run the fact extractor over all user turns in a conversation.

    Simulates the production memory manager's incremental extraction:
    each user turn is processed with the context of prior turns and
    the accumulated profile state from previous turns.
    """
    turns = conversation["turns"]
    accumulated_profile: dict = {}

    for i, turn in enumerate(turns):
        if turn["role"] != "user":
            continue

        # Build known facts string from accumulated profile
        known_parts = []
        for k, v in accumulated_profile.items():
            if v:
                known_parts.append(f"{k}: {v}")
        known_facts = ", ".join(known_parts) if known_parts else "None yet"

        # Build recent history context (last 3 exchanges)
        history_turns = turns[max(0, i - 5):i]
        history_lines = []
        for h in history_turns:
            role_label = "Parent" if h["role"] == "user" else "Coach"
            history_lines.append(f"{role_label}: {h['content']}")
        history_block = (
            "Recent conversation context:\n" + "\n".join(history_lines)
            if history_lines else ""
        )

        prompt = _EXTRACTION_PROMPT.format(
            known_facts=known_facts,
            history_block=history_block,
            user_message=turn["content"],
        )

        extracted = await llm.json(prompt, temperature=0.0, max_tokens=4096)
        if isinstance(extracted, dict):
            for field, value in extracted.items():
                if value:
                    accumulated_profile[field] = value

    return accumulated_profile


async def run(filter_fields: list[str] | None) -> dict:
    if not GOLDEN_MEMORY_PATH.exists():
        logger.error("Memory dataset not found at %s — run dataset_builder first", GOLDEN_MEMORY_PATH)
        sys.exit(1)

    with open(GOLDEN_MEMORY_PATH) as f:
        dataset = json.load(f)

    logger.info("Loaded %d conversations from golden dataset", len(dataset))
    llm = GenClient()

    per_conversation: list[dict] = []
    failures: list[dict] = []

    for i, conv in enumerate(dataset):
        logger.info("[%d/%d] Evaluating conversation %s (persona: %s)",
                    i + 1, len(dataset), conv["id"], conv.get("persona_id", "?"))

        ground_truth = conv.get("ground_truth_extractions", [])
        if filter_fields:
            ground_truth = [g for g in ground_truth if g["fact_key"] in filter_fields]

        if not ground_truth:
            logger.warning("No ground truth facts for conversation %s — skipping", conv["id"])
            continue

        try:
            extracted = await extract_facts_from_conversation(conv, llm)
        except Exception as e:
            logger.error("Extraction failed for conversation %s: %s", conv["id"], e)
            failures.append({"conversation_id": conv["id"], "error": str(e)})
            continue

        scores = extraction_scores(extracted, ground_truth)
        scores["conversation_id"] = conv["id"]
        scores["persona_id"] = conv.get("persona_id", "")
        scores["extracted"] = extracted
        scores["ground_truth"] = ground_truth
        per_conversation.append(scores)

        logger.info("  P=%.3f R=%.3f F1=%.3f  |  fields: %s",
                    scores["precision"], scores["recall"], scores["f1"],
                    {k: "HIT" if v else "miss" for k, v in scores["per_field"].items()})

    summary = aggregate(per_conversation)
    summary["n_failures"] = len(failures)

    _print_report(summary, per_conversation)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"memory_{int(time.time())}.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "per_conversation": per_conversation,
        "failures": failures,
    }, indent=2))
    logger.info("Results saved to %s", out_path)

    return summary


def _print_report(summary: dict, per_conversation: list[dict]) -> None:
    print("\n" + "=" * 60)
    print(f"MEMORY EVAL  |  n={summary.get('n_conversations', 0)}  failures={summary.get('n_failures', 0)}")
    print("=" * 60)
    print(f"  Precision : {summary.get('precision', 0):.4f}")
    print(f"  Recall    : {summary.get('recall', 0):.4f}")
    print(f"  F1        : {summary.get('f1', 0):.4f}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory/fact extraction evaluation")
    parser.add_argument("--fields", nargs="+", default=None, metavar="FIELD",
                        help="Evaluate only specific fields (e.g. child_name diagnosis_status)")
    args = parser.parse_args()
    asyncio.run(run(filter_fields=args.fields))
