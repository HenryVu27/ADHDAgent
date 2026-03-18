from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # RAG
    RAG_TOP_K: int = 5
    RAG_USE_QUERY_REWRITE: bool = True
    RAG_RERANKER: str = "cross_encoder"    # "none" | "cross_encoder" | "colbert"
    RAG_RERANK_CANDIDATES: int = 10
    RAG_RERANK_MODEL: str = "BAAI/bge-reranker-base"
    RAG_RELEVANCE_THRESHOLD: float = 0.0   # Cross-encoder score floor (bge-reranker-base: 0 = decision boundary)
    RAG_COLBERT_MODEL: str = "colbert-ir/colbertv2.0"
    RAG_EMBED_TIMEOUT_S: float = 30.0      # Timeout for embed/embed_batch API calls
    RAG_QUERY_CACHE_TTL_S: float = 300.0   # Embedding-based query cache TTL (seconds)
    RAG_QUERY_CACHE_MAX_SIZE: int = 50     # Max cached query embeddings
    RAG_QUERY_CACHE_SIMILARITY: float = 0.92  # Cosine similarity threshold for cache hit
    RAG_OUTCOME_BOOST_POSITIVE: float = 0.10
    RAG_OUTCOME_BOOST_NEGATIVE: float = -0.15
    RAG_OUTCOME_BOOST_CAP: float = 0.30
    RAG_OUTCOME_JACCARD_THRESHOLD: float = 0.3

    # Qdrant
    QDRANT_URL: str = ":memory:"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "adhd_knowledge"

    # App
    LOG_LEVEL: str = "INFO"
    CORS_ALLOWED_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:8000"]
    CHAT_TIMEOUT_S: float = 120.0
    API_KEY: str = ""           # Empty = dev mode (no auth)
    ADMIN_API_KEY: str = ""     # Empty = no admin gate

    # JWT auth
    JWT_SECRET: str = ""
    JWT_EXPIRY_HOURS: int = 24
    DEFAULT_USER_EMAIL: str = "test@test.com"
    DEFAULT_USER_PASSWORD: str = "test1234"
    RATE_LIMIT_CHAT: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_ENABLED: bool = True

    # Agent
    AGENT_MAX_TOOL_STEPS: int = 8
    WEB_SEARCH_ENABLED: bool = True
    WEB_SEARCH_TIMEOUT_S: float = 15.0

    # Guardrails
    GUARDRAILS_TIMEOUT_S: float = 10.0

    # Semantic fast path
    SEMANTIC_FAST_PATH_ENABLED: bool = True
    SEMANTIC_FAST_PATH_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SEMANTIC_FAST_PATH_THRESHOLD: float = 0.82

    # Context engineering
    CONTEXT_WINDOW_TURNS: int = 6
    CONTEXT_MAX_ACTIVE_GOALS: int = 5
    CONTEXT_MAX_COMPLETED_GOALS: int = 2
    CONTEXT_MAX_OUTCOMES: int = 3
    CONTEXT_MAX_CHARS: int = 120000  # ~30k tokens at 4 chars/token

    # Model split: Pro for agent, Flash for utilities
    GEMINI_AGENT_MODEL: str = "gemini-2.5-pro"
    GEMINI_UTILITY_MODEL: str = ""

    # Thinking mode budget (tokens) for the agent model
    GEMINI_THINKING_BUDGET: int = 2048

    # Fast model for simple messages (defaults to utility/flash model)
    GEMINI_FAST_MODEL: str = ""
    GEMINI_FAST_THINKING_BUDGET: int = 0

    # Memory manager
    SUMMARY_INTERVAL_TURNS: int = 5
    FACT_EXTRACTION_MIN_LENGTH: int = 40
    MEMORY_TIMEOUT_S: float = 30.0

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
        if not self.GEMINI_FAST_MODEL:
            self.GEMINI_FAST_MODEL = self.GEMINI_UTILITY_MODEL
        return self


settings = Settings()
