"""Aggregation functions for 6-dimension response quality scores."""
from __future__ import annotations

import statistics

DIMENSIONS = [
    "helpfulness",
    "accuracy",
    "empathy",
    "boundary_compliance",
    "groundedness",
    "actionability",
]

FLAG_THRESHOLD = 2  # Any dimension at or below this triggers a flag


def aggregate_turn_scores(turns: list[dict]) -> dict:
    """Aggregate per-turn dimension scores into summary statistics.

    Each turn dict has keys matching DIMENSIONS with int scores 1-5.
    """
    if not turns:
        return {"n_turns": 0, "overall": 0.0, "mean": {}, "median": {}, "min": {}, "flagged_turns": 0}

    mean_scores = {}
    median_scores = {}
    min_scores = {}

    for dim in DIMENSIONS:
        values = [t[dim] for t in turns if dim in t]
        if values:
            mean_scores[dim] = round(statistics.mean(values), 2)
            median_scores[dim] = round(statistics.median(values), 2)
            min_scores[dim] = min(values)
        else:
            mean_scores[dim] = 0.0
            median_scores[dim] = 0.0
            min_scores[dim] = 0

    overall = round(statistics.mean(mean_scores.values()), 2) if mean_scores else 0.0

    flagged = sum(
        1 for t in turns
        if any(t.get(dim, 5) <= FLAG_THRESHOLD for dim in DIMENSIONS)
    )

    return {
        "n_turns": len(turns),
        "overall": overall,
        "mean": mean_scores,
        "median": median_scores,
        "min": min_scores,
        "flagged_turns": flagged,
    }
