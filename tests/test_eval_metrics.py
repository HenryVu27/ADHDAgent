import pytest
from eval.metrics.response_quality import aggregate_turn_scores, DIMENSIONS


def test_dimensions_list():
    assert "helpfulness" in DIMENSIONS
    assert "empathy" in DIMENSIONS
    assert len(DIMENSIONS) == 6


def test_aggregate_single_turn():
    turns = [
        {"helpfulness": 4, "accuracy": 5, "empathy": 3, "boundary_compliance": 5, "groundedness": 4, "actionability": 3}
    ]
    result = aggregate_turn_scores(turns)
    assert result["mean"]["helpfulness"] == 4.0
    assert result["overall"] == pytest.approx(4.0)
    assert result["n_turns"] == 1


def test_aggregate_multiple_turns():
    turns = [
        {"helpfulness": 4, "accuracy": 5, "empathy": 3, "boundary_compliance": 5, "groundedness": 4, "actionability": 3},
        {"helpfulness": 2, "accuracy": 3, "empathy": 5, "boundary_compliance": 5, "groundedness": 2, "actionability": 5},
    ]
    result = aggregate_turn_scores(turns)
    assert result["mean"]["helpfulness"] == 3.0
    assert result["median"]["helpfulness"] == 3.0
    assert result["min"]["helpfulness"] == 2
    assert result["n_turns"] == 2
    assert "overall" in result


def test_aggregate_flags_low_scores():
    turns = [
        {"helpfulness": 1, "accuracy": 5, "empathy": 5, "boundary_compliance": 5, "groundedness": 5, "actionability": 5},
        {"helpfulness": 5, "accuracy": 5, "empathy": 5, "boundary_compliance": 5, "groundedness": 5, "actionability": 5},
    ]
    result = aggregate_turn_scores(turns)
    assert result["flagged_turns"] == 1


def test_aggregate_empty():
    result = aggregate_turn_scores([])
    assert result["n_turns"] == 0
    assert result["overall"] == 0.0
