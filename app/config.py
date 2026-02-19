import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # Rules engine: "python" or "clingo"
    RULES_ENGINE: str = os.getenv("RULES_ENGINE", "python")

    # RAG
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    RAG_DENSE_WEIGHT: float = float(os.getenv("RAG_DENSE_WEIGHT", "0.6"))
    RAG_SPARSE_WEIGHT: float = float(os.getenv("RAG_SPARSE_WEIGHT", "0.4"))
    RAG_RERANK_CANDIDATES: int = int(os.getenv("RAG_RERANK_CANDIDATES", "9"))
    RAG_RELEVANCE_THRESHOLD: float = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.3"))
    RAG_USE_QUERY_REWRITE: bool = os.getenv("RAG_USE_QUERY_REWRITE", "true").lower() == "true"
    RAG_USE_RERANKING: bool = os.getenv("RAG_USE_RERANKING", "true").lower() == "true"

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", ":memory:")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "adhd_knowledge")

    # App
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Feature flags
    USE_LLM_EXTRACTION: bool = os.getenv("USE_LLM_EXTRACTION", "true").lower() == "true"
    USE_LLM_SAFETY: bool = os.getenv("USE_LLM_SAFETY", "true").lower() == "true"
    USE_LLM_RESPONSES: bool = os.getenv("USE_LLM_RESPONSES", "true").lower() == "true"

    # NeMo Guardrails
    USE_NEMO_GUARDRAILS: bool = os.getenv("USE_NEMO_GUARDRAILS", "true").lower() == "true"
    NEMO_GUARDRAILS_TIMEOUT_MS: int = int(os.getenv("NEMO_GUARDRAILS_TIMEOUT_MS", "5000"))

    # Context engineering
    CONTEXT_WINDOW_TURNS: int = int(os.getenv("CONTEXT_WINDOW_TURNS", "5"))


settings = Settings()
