import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    # RAG
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    RAG_USE_QUERY_REWRITE: bool = os.getenv("RAG_USE_QUERY_REWRITE", "true").lower() == "true"
    RAG_RERANK_ENABLED: bool = os.getenv("RAG_RERANK_ENABLED", "true").lower() == "true"
    RAG_RERANK_CANDIDATES: int = int(os.getenv("RAG_RERANK_CANDIDATES", "10"))
    RAG_RERANK_MODEL: str = os.getenv("RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", ":memory:")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "adhd_knowledge")

    # App
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Agent
    AGENT_MAX_TOOL_STEPS: int = int(os.getenv("AGENT_MAX_TOOL_STEPS", "5"))

    # NeMo Guardrails
    NEMO_GUARDRAILS_TIMEOUT_MS: int = int(os.getenv("NEMO_GUARDRAILS_TIMEOUT_MS", "15000"))

    # Context engineering
    CONTEXT_WINDOW_TURNS: int = int(os.getenv("CONTEXT_WINDOW_TURNS", "6"))

    # Model routing
    GEMINI_MODEL_FAST: str = os.getenv("GEMINI_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    GEMINI_MODEL_STANDARD: str = os.getenv("GEMINI_MODEL_STANDARD", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    GEMINI_MODEL_COMPLEX: str = os.getenv("GEMINI_MODEL_COMPLEX", "gemini-3-pro-preview")
    GEMINI_MODEL_BACKGROUND: str = os.getenv("GEMINI_MODEL_BACKGROUND", os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"))
    MODEL_ROUTING_ENABLED: bool = os.getenv("MODEL_ROUTING_ENABLED", "false").lower() == "true"

    # Memory manager
    SUMMARY_INTERVAL_TURNS: int = int(os.getenv("SUMMARY_INTERVAL_TURNS", "5"))
    FACT_EXTRACTION_MIN_LENGTH: int = int(os.getenv("FACT_EXTRACTION_MIN_LENGTH", "40"))

    # SQLite persistence
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "adhd_agent.db")
    SQLITE_ENABLED: bool = os.getenv("SQLITE_ENABLED", "true").lower() == "true"

    # Observability
    ANALYZER_ENABLED: bool = os.getenv("ANALYZER_ENABLED", "true").lower() == "true"
    EVENT_BUFFER_SIZE: int = int(os.getenv("EVENT_BUFFER_SIZE", "200"))


settings = Settings()
