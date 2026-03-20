"""Pipeline configuration dataclass and ablation config definitions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    use_reranker: bool
    use_colbert: bool
    use_query_rewriter: bool


@dataclass(frozen=True)
class PipelineVariant(PipelineConfig):
    """Full pipeline config for end-to-end comparison.
    Inherits RAG knobs from PipelineConfig. Adds agent-level knobs.
    """
    model: str = "gemini-2.5-pro"
    system_prompt_version: str = "default"
    thinking_budget: int = 8000
    temperature: float = 0.0


# Each config varies exactly one axis from baseline.
# 'full' combines all axes and serves as a best-possible reference.
ABLATION_CONFIGS: list[PipelineConfig] = [
    PipelineConfig("baseline",  use_reranker=False, use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+rewriter", use_reranker=False, use_colbert=False, use_query_rewriter=True),
    PipelineConfig("+reranker", use_reranker=True,  use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+colbert",  use_reranker=False, use_colbert=True,  use_query_rewriter=False),
    PipelineConfig("full",      use_reranker=True,  use_colbert=True,  use_query_rewriter=True),
]
