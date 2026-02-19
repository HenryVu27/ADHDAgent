import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # RAG
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    RAG_USE_QUERY_REWRITE: bool = os.getenv("RAG_USE_QUERY_REWRITE", "true").lower() == "true"

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", ":memory:")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "adhd_knowledge")

    # App
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Feature flags
    USE_LLM_RESPONSES: bool = os.getenv("USE_LLM_RESPONSES", "true").lower() == "true"

    # NeMo Guardrails
    NEMO_GUARDRAILS_TIMEOUT_MS: int = int(os.getenv("NEMO_GUARDRAILS_TIMEOUT_MS", "5000"))

    # Context engineering
    CONTEXT_WINDOW_TURNS: int = int(os.getenv("CONTEXT_WINDOW_TURNS", "5"))


settings = Settings()
