from pydantic_settings import BaseSettings


class IngestConfig(BaseSettings):
    # API keys
    ncbi_api_key: str = ""
    s2_api_key: str = ""
    openalex_email: str = ""
    gemini_api_key: str = ""

    # Pipeline
    relevance_threshold: int = 6
    batch_size: int = 500
    dedup_title_threshold: float = 0.85
    filter_batch_size: int = 20

    # Rate limiting (requests per second)
    pubmed_rps: float = 10.0
    openalex_rps: float = 10.0
    s2_rps: float = 10.0
    eric_rps: float = 5.0
    gov_rps: float = 2.0

    # Paths
    output_dir: str = "app/knowledge"
    raw_cache_dir: str = "scripts/ingest/.cache"

    # Default queries
    default_queries: list[str] = [
        "ADHD parenting strategies",
        "attention deficit hyperactivity disorder child behavior",
        "behavioral intervention pediatric ADHD",
        "executive function child intervention",
        "parent training ADHD",
        "ADHD school accommodations",
        "ADHD family support",
    ]

    model_config = {"env_file": ".env", "env_prefix": "INGEST_", "extra": "ignore"}
