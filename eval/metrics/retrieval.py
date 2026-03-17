"""Standard IR metrics for retrieval evaluation.

All functions operate on lists of retrieved document IDs (ranked, most
relevant first) and a set of relevant document IDs (ground truth).
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k results."""
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Assumes binary relevance (relevant = gain 1, not relevant = gain 0).
    """
    def dcg(ranked: list[str], rel: set[str], k: int) -> float:
        return sum(
            (1.0 / math.log2(rank + 1))
            for rank, doc_id in enumerate(ranked[:k], start=1)
            if doc_id in rel
        )

    actual_dcg = dcg(retrieved, relevant, k)
    # Ideal DCG: relevant docs ranked first
    ideal_ranked = list(relevant) + [d for d in retrieved if d not in relevant]
    ideal_dcg = dcg(ideal_ranked, relevant, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def aggregate(per_query_scores: list[dict]) -> dict:
    """Compute mean over all per-query metric dicts."""
    if not per_query_scores:
        return {}
    keys = per_query_scores[0].keys()
    return {
        k: round(sum(s[k] for s in per_query_scores) / len(per_query_scores), 4)
        for k in keys
        if isinstance(per_query_scores[0][k], (int, float))
    }
