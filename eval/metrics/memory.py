"""Metrics for memory / fact extraction evaluation.

Operates on the structured extraction output fields used by the production
fact extractor: child_name, child_age, diagnosis_status, adhd_subtype,
challenge_areas, attempted_strategies, good_day_description, hardest_situations.
"""
from __future__ import annotations

from typing import Any

_LIST_FIELDS = {"challenge_areas", "attempted_strategies", "hardest_situations"}
_SCALAR_FIELDS = {"child_name", "child_age", "diagnosis_status", "adhd_subtype", "good_day_description"}


def _normalize(value: Any) -> set[str]:
    """Normalize a field value to a comparable set of lowercase strings."""
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(v).strip().lower() for v in value if v}
    return {str(value).strip().lower()}


def field_match(extracted_value: Any, expected_value: Any, field: str) -> bool:
    """Return True if extracted value matches expected value for a field.

    For scalar fields: case-insensitive string match.
    For list fields: at least one extracted item matches an expected item.
    """
    if expected_value is None:
        return extracted_value is None

    extracted_norm = _normalize(extracted_value)
    expected_norm = _normalize(expected_value)

    if not extracted_norm:
        return False

    if field in _LIST_FIELDS:
        # Partial credit: at least one match
        return bool(extracted_norm & expected_norm)
    else:
        # Exact match after normalization
        return extracted_norm == expected_norm


def extraction_scores(
    extracted: dict,
    ground_truth: list[dict],
) -> dict:
    """Compute precision and recall for a single fact extraction result.

    Args:
        extracted: the dict returned by the fact extractor (field -> value)
        ground_truth: list of {fact_key, expected_value} dicts from the golden dataset

    Returns:
        {precision, recall, f1, per_field: {field: hit/miss}}
    """
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "per_field": {}}

    # Recall: how many ground-truth facts were correctly extracted?
    true_positives = 0
    per_field: dict[str, bool] = {}
    for item in ground_truth:
        field = item["fact_key"]
        expected = item["expected_value"]
        extracted_value = extracted.get(field)
        hit = field_match(extracted_value, expected, field)
        per_field[field] = hit
        if hit:
            true_positives += 1

    recall = true_positives / len(ground_truth)

    # Precision: how many extracted fields are actually correct?
    extracted_fields = {k: v for k, v in extracted.items() if v is not None and v != [] and v != ""}
    if not extracted_fields:
        precision = 0.0
    else:
        gt_by_field = {item["fact_key"]: item["expected_value"] for item in ground_truth}
        correct_extractions = sum(
            1 for field, value in extracted_fields.items()
            if field in gt_by_field and field_match(value, gt_by_field[field], field)
        )
        precision = correct_extractions / len(extracted_fields)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_field": per_field,
    }


def aggregate(per_conversation_scores: list[dict]) -> dict:
    """Compute mean precision, recall, F1 across all conversations."""
    if not per_conversation_scores:
        return {}

    precision = sum(s["precision"] for s in per_conversation_scores) / len(per_conversation_scores)
    recall = sum(s["recall"] for s in per_conversation_scores) / len(per_conversation_scores)
    f1 = sum(s["f1"] for s in per_conversation_scores) / len(per_conversation_scores)

    # Per-field hit rate across all conversations
    field_hits: dict[str, list[bool]] = {}
    for s in per_conversation_scores:
        for field, hit in s.get("per_field", {}).items():
            field_hits.setdefault(field, []).append(hit)

    per_field_rate = {
        field: round(sum(hits) / len(hits), 4)
        for field, hits in field_hits.items()
    }

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_conversations": len(per_conversation_scores),
        "per_field_recall": per_field_rate,
    }
