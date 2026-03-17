from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig


def test_pipeline_config_fields():
    cfg = PipelineConfig(
        name="test",
        sparse_mode="tfidf",
        use_reranker=False,
        use_colbert=False,
        use_query_rewriter=False,
    )
    assert cfg.name == "test"
    assert cfg.sparse_mode == "tfidf"
    assert cfg.use_reranker is False
    assert cfg.use_colbert is False
    assert cfg.use_query_rewriter is False


def test_ablation_configs_count():
    assert len(ABLATION_CONFIGS) == 6


def test_ablation_config_names():
    names = [c.name for c in ABLATION_CONFIGS]
    assert names == ["baseline", "+rewriter", "+reranker", "+bm25", "+colbert", "full"]


def test_baseline_has_nothing_on():
    baseline = ABLATION_CONFIGS[0]
    assert baseline.sparse_mode == "tfidf"
    assert not baseline.use_reranker
    assert not baseline.use_colbert
    assert not baseline.use_query_rewriter


def test_full_has_everything_on():
    full = ABLATION_CONFIGS[5]
    assert full.sparse_mode == "bm25"
    assert full.use_reranker
    assert full.use_colbert
    assert full.use_query_rewriter
