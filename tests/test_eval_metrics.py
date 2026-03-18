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


from eval.metrics.tool_use import aggregate_tool_scores, compute_tool_f1


def test_compute_tool_f1_perfect():
    result = compute_tool_f1(
        verdicts=[
            {"name": "search_knowledge_base", "verdict": "appropriate"},
            {"name": "update_family_profile", "verdict": "appropriate"},
        ],
        missed_tools=[]
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_compute_tool_f1_with_unnecessary():
    result = compute_tool_f1(
        verdicts=[
            {"name": "search_knowledge_base", "verdict": "appropriate"},
            {"name": "update_family_profile", "verdict": "unnecessary"},
        ],
        missed_tools=[]
    )
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_compute_tool_f1_with_missed():
    result = compute_tool_f1(
        verdicts=[
            {"name": "search_knowledge_base", "verdict": "appropriate"},
        ],
        missed_tools=["update_family_profile"]
    )
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_compute_tool_f1_no_tools():
    result = compute_tool_f1(verdicts=[], missed_tools=[])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_aggregate_tool_scores():
    turns = [
        {"precision": 1.0, "recall": 0.5, "f1": 0.67, "argument_accuracy": 4, "result_utilization": 5, "by_tool": {}},
        {"precision": 0.5, "recall": 1.0, "f1": 0.67, "argument_accuracy": 3, "result_utilization": 4, "by_tool": {}},
    ]
    result = aggregate_tool_scores(turns)
    assert result["mean_precision"] == 0.75
    assert result["mean_recall"] == 0.75
    assert result["mean_argument_accuracy"] == 3.5
    assert result["n_turns"] == 2


from eval.metrics.coherence import aggregate_coherence_scores, COHERENCE_DIMENSIONS


def test_coherence_dimensions():
    assert len(COHERENCE_DIMENSIONS) == 4
    assert "progressive_profiling" in COHERENCE_DIMENSIONS


def test_aggregate_coherence_scores():
    conversations = [
        {"progressive_profiling": 4, "repetition_avoidance": 5, "follow_up": 2, "topic_management": 4},
        {"progressive_profiling": 3, "repetition_avoidance": 4, "follow_up": 4, "topic_management": 3},
    ]
    result = aggregate_coherence_scores(conversations)
    assert result["n_conversations"] == 2
    assert result["mean"]["progressive_profiling"] == 3.5
    assert result["mean"]["follow_up"] == 3.0
    assert "overall" in result


def test_aggregate_coherence_empty():
    result = aggregate_coherence_scores([])
    assert result["n_conversations"] == 0
