"""Pipeline configuration dataclass and ablation config definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    sparse_mode: Literal["tfidf", "bm25"]
    use_reranker: bool
    use_colbert: bool
    use_query_rewriter: bool


# Each config varies exactly one axis from baseline.
# 'full' combines all axes and serves as a best-possible reference.
ABLATION_CONFIGS: list[PipelineConfig] = [
    PipelineConfig("baseline",  sparse_mode="tfidf", use_reranker=False, use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+rewriter",  sparse_mode="tfidf", use_reranker=False, use_colbert=False, use_query_rewriter=True),
    PipelineConfig("+reranker",  sparse_mode="tfidf", use_reranker=True,  use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+bm25",      sparse_mode="bm25",  use_reranker=False, use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+colbert",   sparse_mode="tfidf", use_reranker=False, use_colbert=True,  use_query_rewriter=False),
    PipelineConfig("full",       sparse_mode="bm25",  use_reranker=True,  use_colbert=True,  use_query_rewriter=True),
]
