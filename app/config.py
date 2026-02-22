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
    CONTEXT_MAX_ACTIVE_GOALS: int = 5
    CONTEXT_MAX_COMPLETED_GOALS: int = 2
    CONTEXT_MAX_OUTCOMES: int = 3
    CONTEXT_MAX_CHARS: int = 120000  # ~30k tokens at 4 chars/token

    # Model split: Pro for agent, Flash for utilities
    GEMINI_AGENT_MODEL: str = "gemini-3-pro-preview"
    GEMINI_UTILITY_MODEL: str = ""

    # Thinking mode budget (tokens) for the agent model
    GEMINI_THINKING_BUDGET: int = 2048

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
    def _fill_model_defaults(self) -> "Settings":
        """GEMINI_UTILITY_MODEL defaults to GEMINI_MODEL (flash) when not set."""
        if not self.GEMINI_UTILITY_MODEL:
            self.GEMINI_UTILITY_MODEL = self.GEMINI_MODEL
        return self


settings = Settings()
