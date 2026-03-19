"""Tool use evaluation metrics: precision, recall, F1, argument accuracy, result utilization."""
from __future__ import annotations

import statistics


def compute_tool_f1(
    verdicts: list[dict],
    missed_tools: list[str],
) -> dict:
    """Compute precision/recall/F1 for tool calls in a single turn.

    verdicts: list of {"name": str, "verdict": "appropriate"|"unnecessary"}
    missed_tools: tool names that should have been called but weren't
    """
    if not verdicts and not missed_tools:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    appropriate = sum(1 for v in verdicts if v["verdict"] == "appropriate")
    total_called = len(verdicts)
    total_should_call = appropriate + len(missed_tools)

    precision = appropriate / total_called if total_called > 0 else 1.0
    recall = appropriate / total_should_call if total_should_call > 0 else 1.0

    if precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": f1,
    }


def aggregate_tool_scores(turns: list[dict]) -> dict:
    """Aggregate per-turn tool use scores.

    Each turn dict has: precision, recall, f1, argument_accuracy, result_utilization, by_tool.
    """
    if not turns:
        return {"n_turns": 0, "mean_precision": 0.0, "mean_recall": 0.0, "mean_f1": 0.0,
                "mean_argument_accuracy": 0.0, "mean_result_utilization": 0.0}

    return {
        "n_turns": len(turns),
        "mean_precision": round(statistics.mean(t["precision"] for t in turns), 4),
        "mean_recall": round(statistics.mean(t["recall"] for t in turns), 4),
        "mean_f1": round(statistics.mean(t["f1"] for t in turns), 4),
        "mean_argument_accuracy": round(statistics.mean(t["argument_accuracy"] for t in turns), 2),
        "mean_result_utilization": round(statistics.mean(t["result_utilization"] for t in turns), 2),
    }
