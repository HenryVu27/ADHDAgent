# ADHDAgent FastAPI application — ReAct agent architecture

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, set_knowledge_base, set_orchestrator
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

    # 2. Load knowledge store + build Qdrant index
    from app.rag.knowledge_store import KnowledgeStore
    store = KnowledgeStore()
    if gemini:
        try:
            await store.build_index(gemini)
            logger.info("Qdrant hybrid index built (dense + sparse)")
        except Exception as e:
            logger.error(f"Qdrant index build failed: {e}")
    set_knowledge_base(store)

    # 3. Initialize NeMo Guardrails (required)
    from app.guardrails.validator import GuardrailsValidator
    guardrails = GuardrailsValidator(gemini_client=gemini)
    logger.info("NeMo Guardrails initialized (input + output rails)")

    # 4. Create session store, retriever, tools
    from app.agent.session_store import InMemorySessionStore
    from app.agent.tools import create_tools
    from app.rag.query_rewriter import QueryRewriter
    from app.rag.retriever import HybridRetriever

    if settings.SQLITE_ENABLED:
        from app.agent.sqlite_store import SQLiteSessionStore
        from app.db import get_connection, init_db
        conn = get_connection(settings.SQLITE_DB_PATH)
        init_db(conn)
        session_store = SQLiteSessionStore(conn)
        logger.info("Using SQLite session store (path=%s)", settings.SQLITE_DB_PATH)
    else:
        session_store = InMemorySessionStore()
        logger.info("Using in-memory session store")
    query_rewriter = QueryRewriter(gemini_client=gemini)
    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=gemini,
        query_rewriter=query_rewriter,
    )
    tools = create_tools(retriever=retriever, session_store=session_store)

    # 5. Create hooks (guardrails + context injection)
    from app.agent.hooks import create_hooks
    pre_model_hook, post_model_hook = create_hooks(
        guardrails=guardrails,
        session_store=session_store,
    )

    # 6. Create memory manager (optional, requires Gemini)
    from app.agent.memory import MemoryManager
    memory_manager = MemoryManager(session_store=session_store, gemini_client=gemini) if gemini else None
    if memory_manager:
        logger.info("MemoryManager initialized (summary_interval=%d)", settings.SUMMARY_INTERVAL_TURNS)

    # 7. Build ReAct agent and orchestrator
    from app.agent.graph import build_agent
    from app.agent.orchestrator import AgentOrchestrator

    agent = build_agent(
        tools=tools,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
    )
    orchestrator = AgentOrchestrator(
        agent=agent,
        session_store=session_store,
        memory_manager=memory_manager,
    )
    set_orchestrator(orchestrator)

    logger.info("ADHDAgent ready (ReAct agent architecture)")
    yield
    logger.info("ADHDAgent shutting down")


app = FastAPI(
    title="ADHDAgent",
    description="ReAct ADHD coaching agent with NeMo Guardrails",
    version="0.4.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")

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
