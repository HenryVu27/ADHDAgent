"""Pairwise comparison metrics: win rates, confidence weighting, dimension breakdown."""
from __future__ import annotations

from collections import defaultdict

CONFIDENCE_WEIGHTS = {"high": 1.0, "medium": 0.66, "low": 0.33}


def aggregate_pairwise(
    judgments: list[dict],
    variant_a: str,
    variant_b: str,
) -> dict:
    """Aggregate pairwise judgments into win rates and dimension breakdowns."""
    if not judgments:
        return {"n_comparisons": 0, "variant_a": variant_a, "variant_b": variant_b,
                "win_rate_a": 0.0, "win_rate_b": 0.0, "tie_rate": 0.0,
                "weighted_win_rate_a": 0.0, "weighted_win_rate_b": 0.0,
                "dimension_win_rates": {}}

    n = len(judgments)
    wins_a = sum(1 for j in judgments if j["winner"] == "A")
    wins_b = sum(1 for j in judgments if j["winner"] == "B")
    ties = sum(1 for j in judgments if j["winner"] == "tie")

    # Confidence-weighted win rates
    weighted_a = sum(CONFIDENCE_WEIGHTS.get(j.get("confidence", "medium"), 0.66)
                     for j in judgments if j["winner"] == "A")
    weighted_b = sum(CONFIDENCE_WEIGHTS.get(j.get("confidence", "medium"), 0.66)
                     for j in judgments if j["winner"] == "B")
    total_weight = sum(CONFIDENCE_WEIGHTS.get(j.get("confidence", "medium"), 0.66)
                       for j in judgments)

    # Dimension-level win rates
    dim_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"A": 0, "B": 0, "tie": 0})
    for j in judgments:
        for dim, winner in j.get("dimension_winners", {}).items():
            dim_counts[dim][winner] += 1

    dim_win_rates = {}
    for dim, counts in dim_counts.items():
        dim_total = counts["A"] + counts["B"] + counts["tie"]
        if dim_total > 0:
            dim_win_rates[dim] = {
                "A": round(counts["A"] / dim_total, 4),
                "B": round(counts["B"] / dim_total, 4),
                "tie": round(counts["tie"] / dim_total, 4),
            }

    return {
        "n_comparisons": n,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "win_rate_a": round(wins_a / n, 4),
        "win_rate_b": round(wins_b / n, 4),
        "tie_rate": round(ties / n, 4),
        "weighted_win_rate_a": round(weighted_a / total_weight, 4) if total_weight > 0 else 0.0,
        "weighted_win_rate_b": round(weighted_b / total_weight, 4) if total_weight > 0 else 0.0,
        "dimension_win_rates": dim_win_rates,
    }
