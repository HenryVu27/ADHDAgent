import dataclasses
import pytest
from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig


def test_eval_judge_model_is_pro():
    from eval.config import EVAL_JUDGE_MODEL
    assert EVAL_JUDGE_MODEL == "gemini-2.5-pro"


def test_pipeline_variant_inherits_pipeline_config():
    from eval.pipeline_config import PipelineVariant
    v = PipelineVariant(
        name="test-variant",
        use_reranker=True,
        use_colbert=False,
        use_query_rewriter=True,
        model="gemini-2.5-pro",
        system_prompt_version="v2",
        thinking_budget=4000,
        temperature=0.1,
    )
    assert v.name == "test-variant"
    assert v.use_reranker is True
    assert v.model == "gemini-2.5-pro"
    assert v.system_prompt_version == "v2"
    assert v.thinking_budget == 4000
    assert v.temperature == 0.1


def test_pipeline_variant_defaults():
    from eval.pipeline_config import PipelineVariant
    v = PipelineVariant(name="minimal", use_reranker=False, use_colbert=False, use_query_rewriter=False)
    assert v.model == "gemini-2.5-pro"
    assert v.system_prompt_version == "default"
    assert v.thinking_budget == 8000
    assert v.temperature == 0.0


def test_pipeline_variant_is_frozen():
    from eval.pipeline_config import PipelineVariant
    v = PipelineVariant(name="frozen", use_reranker=False, use_colbert=False, use_query_rewriter=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.model = "other"


def test_existing_ablation_configs_unchanged():
    assert len(ABLATION_CONFIGS) == 5
    assert all(isinstance(c, PipelineConfig) for c in ABLATION_CONFIGS)


def test_pipeline_config_fields():
    cfg = PipelineConfig(
        name="test",
        use_reranker=False,
        use_colbert=False,
        use_query_rewriter=False,
    )
    assert cfg.name == "test"
    assert cfg.use_reranker is False
    assert cfg.use_colbert is False
    assert cfg.use_query_rewriter is False


def test_ablation_configs_count():
    assert len(ABLATION_CONFIGS) == 5


def test_ablation_config_names():
    names = [c.name for c in ABLATION_CONFIGS]
    assert names == ["baseline", "+rewriter", "+reranker", "+colbert", "full"]


def test_baseline_has_nothing_on():
    baseline = ABLATION_CONFIGS[0]
    assert not baseline.use_reranker
    assert not baseline.use_colbert
    assert not baseline.use_query_rewriter


def test_full_has_everything_on():
    full = ABLATION_CONFIGS[4]
    assert full.use_reranker
    assert full.use_colbert
    assert full.use_query_rewriter
