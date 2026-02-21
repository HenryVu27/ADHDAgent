from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # RAG
    RAG_TOP_K: int = 5
    RAG_USE_QUERY_REWRITE: bool = True
    RAG_RERANK_ENABLED: bool = True
    RAG_RERANK_CANDIDATES: int = 10
    RAG_RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # Qdrant
    QDRANT_URL: str = ":memory:"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "adhd_knowledge"

    # App
    LOG_LEVEL: str = "INFO"

    # Agent
    AGENT_MAX_TOOL_STEPS: int = 5

    # Guardrails
    GUARDRAILS_TIMEOUT_S: float = 10.0

    # Context engineering
    CONTEXT_WINDOW_TURNS: int = 6

    # Model routing
    GEMINI_MODEL_FAST: str = ""
    GEMINI_MODEL_STANDARD: str = ""
    GEMINI_MODEL_COMPLEX: str = "gemini-3-pro-preview"
    GEMINI_MODEL_BACKGROUND: str = ""
    MODEL_ROUTING_ENABLED: bool = False

    # Memory manager
    SUMMARY_INTERVAL_TURNS: int = 5
    FACT_EXTRACTION_MIN_LENGTH: int = 40

    # SQLite persistence
    SQLITE_DB_PATH: str = "adhd_agent.db"
    SQLITE_ENABLED: bool = True

    # Observability
    ANALYZER_ENABLED: bool = True
    EVENT_BUFFER_SIZE: int = 200

    @model_validator(mode="after")
    def _fill_model_tier_defaults(self) -> "Settings":
        """Model tier fields default to GEMINI_MODEL when not explicitly set."""
        if not self.GEMINI_MODEL_FAST:
            self.GEMINI_MODEL_FAST = self.GEMINI_MODEL
        if not self.GEMINI_MODEL_STANDARD:
            self.GEMINI_MODEL_STANDARD = self.GEMINI_MODEL
        if not self.GEMINI_MODEL_BACKGROUND:
            self.GEMINI_MODEL_BACKGROUND = self.GEMINI_MODEL
        return self


settings = Settings()
