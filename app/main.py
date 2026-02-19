# ADHDAgent FastAPI application

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agents.intake import IntakeAgent
from app.agents.orchestrator import AgentOrchestrator
from app.agents.progress import ProgressAgent
from app.agents.strategy import StrategyAgent
from app.api.routes import router, set_knowledge_base, set_orchestrator
from app.config import settings
from app.phase_manager import PhaseManager
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import HybridRetriever

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

    # 4. Create phase manager
    phase_manager = PhaseManager()

    # 5. Wire all components
    query_rewriter = QueryRewriter(gemini_client=gemini)
    retriever = HybridRetriever(
        knowledge_store=store,
        gemini_client=gemini,
        query_rewriter=query_rewriter,
    )
    intake = IntakeAgent(gemini_client=gemini)
    strategy = StrategyAgent(gemini_client=gemini)
    progress = ProgressAgent(gemini_client=gemini)

    orchestrator = AgentOrchestrator(
        guardrails=guardrails,
        phase_manager=phase_manager,
        retriever=retriever,
        intake=intake,
        strategy=strategy,
        progress=progress,
    )
    set_orchestrator(orchestrator)

    logger.info("ADHDAgent ready")
    yield
    logger.info("ADHDAgent shutting down")


app = FastAPI(
    title="ADHDAgent",
    description="Agentic ADHD coaching with NeMo Guardrails conversation safety",
    version="0.3.0",
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
