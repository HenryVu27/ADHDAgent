"""Conversation-level coherence score aggregation."""
from __future__ import annotations

import statistics

COHERENCE_DIMENSIONS = [
    "progressive_profiling",
    "repetition_avoidance",
    "follow_up",
    "topic_management",
]


def aggregate_coherence_scores(conversations: list[dict]) -> dict:
    """Aggregate per-conversation coherence scores."""
    if not conversations:
        return {"n_conversations": 0, "overall": 0.0, "mean": {}, "min": {}}

    mean_scores = {}
    min_scores = {}

    for dim in COHERENCE_DIMENSIONS:
        values = [c[dim] for c in conversations if dim in c]
        if values:
            mean_scores[dim] = round(statistics.mean(values), 2)
            min_scores[dim] = min(values)
        else:
            mean_scores[dim] = 0.0
            min_scores[dim] = 0

    overall = round(statistics.mean(mean_scores.values()), 2) if mean_scores else 0.0

    return {
        "n_conversations": len(conversations),
        "overall": overall,
        "mean": mean_scores,
        "min": min_scores,
    }
