# ADHDAgent FastAPI application — ReAct agent architecture

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.middleware import APIKeyMiddleware
from app.api.observability_routes import obs_router
from app.api.rate_limit import limiter
from app.api.routes import router
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("Starting ADHDAgent...")

    # 1. Init Gemini client (None if no API key)
    gemini = None
    if settings.GEMINI_API_KEY:
        from app.llm.client import GeminiClient
        gemini = GeminiClient()
        logger.info(f"Gemini client initialized (model={settings.GEMINI_MODEL})")
    else:
        logger.warning("No GEMINI_API_KEY — running in fallback mode (no LLM)")

    # 2. Optionally load ColBERT model (before index build — needed for embed_chunks)
    colbert = None
    if settings.RAG_COLBERT_ENABLED:
        try:
            from app.rag.colbert_index import ColBERTIndex
            colbert = ColBERTIndex(settings.RAG_COLBERT_MODEL)
        except Exception as e:
            logger.error("ColBERT model load failed — ColBERT disabled: %s", e)
            colbert = None

    # 3. Load knowledge store + build Qdrant index
    from app.rag.knowledge_store import KnowledgeStore
    store = KnowledgeStore()
    if gemini:
        try:
            await store.build_index(gemini, colbert_index=colbert)
            colbert_note = " + colbert" if colbert else ""
            logger.info("Qdrant hybrid index built (dense + sparse%s)", colbert_note)
        except Exception as e:
            logger.error(f"Qdrant index build failed: {e}")
    app.state.knowledge_base = store

    # 4. Initialize guardrail gates
    from app.guardrails.validator import InputGate, OutputGate

    fast_path = None
    if gemini and settings.SEMANTIC_FAST_PATH_ENABLED:
        from app.guardrails.fast_path import SemanticFastPath
        fast_path = SemanticFastPath(
            model_name=settings.SEMANTIC_FAST_PATH_MODEL,
            threshold=settings.SEMANTIC_FAST_PATH_THRESHOLD,
        )
        fast_path.build_index()  # synchronous — blocks startup intentionally
        if fast_path._index is None:
            logger.warning(
                "SemanticFastPath: index build failed — fast path disabled (Gemini gate will run for all messages)"
            )
            fast_path = None
        else:
            logger.info(
                "SemanticFastPath initialized (model=%s, threshold=%.2f)",
                settings.SEMANTIC_FAST_PATH_MODEL,
                settings.SEMANTIC_FAST_PATH_THRESHOLD,
            )

    input_gate = InputGate(gemini_client=gemini, fast_path=fast_path) if gemini else None
    output_gate = OutputGate(gemini_client=gemini) if gemini else None
    logger.info("Guardrail gates initialized (input + output)")

    # 5. Create session store, retriever, tools
    from app.agent.tools import create_tools
    from app.rag.query_rewriter import QueryRewriter
    from app.rag.retriever import HybridRetriever

    db_conn = None
    if settings.SQLITE_ENABLED:
        from app.agent.sqlite_store import SQLiteSessionStore
        from app.db import get_async_connection, init_db_async
        db_conn = await get_async_connection(settings.SQLITE_DB_PATH)
        await init_db_async(db_conn)
        session_store = SQLiteSessionStore(db_conn)
        logger.info("Using SQLite session store (path=%s)", settings.SQLITE_DB_PATH)
    else:
        from app.agent.session_store import create_in_memory_store
        session_store = await create_in_memory_store()
        logger.info("Using in-memory session store")
    query_rewriter = QueryRewriter(gemini_client=gemini)

    reranker = None
    if settings.RAG_RERANK_ENABLED:
        from app.rag.reranker import FastEmbedReranker
        reranker = FastEmbedReranker(model_name=settings.RAG_RERANK_MODEL)
        logger.info("FastEmbed reranker enabled (model=%s, candidates=%d)", settings.RAG_RERANK_MODEL, settings.RAG_RERANK_CANDIDATES)

    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=gemini,
        query_rewriter=query_rewriter,
        reranker=reranker,
        colbert_index=colbert,
    )
    tools = create_tools(retriever=retriever, session_store=session_store)

    # 6. Create event bus (with SQLite persistence when available)
    #    Uses a separate connection to avoid contention with the session store.
    from app.agent.event_bus import EventBus
    event_bus = EventBus(buffer_size=settings.EVENT_BUFFER_SIZE, conn=db_conn)
    logger.info("EventBus initialized (buffer_size=%d)", settings.EVENT_BUFFER_SIZE)

    # 7. Create context preparation hook
    from app.agent.hooks import create_prepare_context
    prepare_context = create_prepare_context(
        session_store=session_store,
        event_bus=event_bus,
    )

    # 8. Create memory manager (optional, requires Gemini)
    from app.agent.memory import MemoryManager
    memory_manager = MemoryManager(
        session_store=session_store, gemini_client=gemini, event_bus=event_bus,
    ) if gemini else None
    if memory_manager:
        logger.info("MemoryManager initialized (summary_interval=%d)", settings.SUMMARY_INTERVAL_TURNS)

    # 9. Create conversation analyzer (optional, requires Gemini)
    analyzer = None
    if settings.ANALYZER_ENABLED and gemini:
        from app.agent.analyzer import ConversationAnalyzer
        analyzer = ConversationAnalyzer(session_store=session_store, gemini_client=gemini)
        logger.info("ConversationAnalyzer initialized")

    # 10. Build ReAct agent and orchestrator
    from app.agent.graph import build_agent
    from app.agent.orchestrator import AgentOrchestrator

    agent = build_agent(
        tools=tools,
        prepare_context=prepare_context,
        input_gate=input_gate,
        output_gate=output_gate,
    )
    orchestrator = AgentOrchestrator(
        agent=agent,
        session_store=session_store,
        memory_manager=memory_manager,
        analyzer=analyzer,
        event_bus=event_bus,
        output_gate=output_gate,
        gemini_client=gemini,
    )
    app.state.orchestrator = orchestrator
    app.state.session_store = session_store
    app.state.event_bus = event_bus
    app.state.analyzer = analyzer

    logger.info(
        "ADHDAgent ready (pro_model=%s, fast_model=%s, utility_model=%s, colbert=%s)",
        settings.GEMINI_AGENT_MODEL,
        settings.GEMINI_FAST_MODEL,
        settings.GEMINI_UTILITY_MODEL,
        "enabled" if colbert else "disabled",
    )
    yield
    if hasattr(orchestrator, 'shutdown'):
        await orchestrator.shutdown()
    if db_conn:
        await db_conn.close()
    logger.info("ADHDAgent shutting down")


app = FastAPI(
    title="ADHDAgent",
    description="ReAct ADHD coaching agent with guardrail gates",
    version="0.4.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(APIKeyMiddleware)  # Added BEFORE CORS (CORS must be outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(obs_router, prefix="/api")

# Serve React build
react_dist = Path(__file__).parent.parent / "frontend-react" / "dist"

if react_dist.exists():
    # SPA-aware static serving for React build
    if (react_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=react_dist / "assets"), name="assets")
    if (react_dist / "mascots").exists():
        app.mount("/mascots", StaticFiles(directory=react_dist / "mascots"), name="mascots")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        # Serve static files if they exist, otherwise serve index.html for client-side routing
        file_path = react_dist / full_path
        if full_path and file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(react_dist / "index.html")
