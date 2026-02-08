from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.routes import router

app = FastAPI(
    title="ADHDAgent",
    description="Agentic ADHD coaching with ASP-backed conversation guardrails",
    version="0.1.0",
)

app.include_router(router, prefix="/api")

frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(frontend_dir / "index.html")
