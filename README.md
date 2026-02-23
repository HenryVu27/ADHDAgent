# ADHDAgent

Parent-facing ADHD coaching chatbot using a ReAct agent architecture with multi-layer safety guardrails.

Input gate (crisis + jailbreak) → Context assembly → Gemini ReAct agent with tool calling → Output gate (medication + diagnosis + scope).

## Key Features

- **ReAct agent** with LangGraph tool calling and full observability
- **Hybrid RAG** - Qdrant dense + sparse vectors with RRF, tag boosting, and local cross-encoder reranking
- **LangGraph-native guardrail gates** - structured Gemini classifiers for input (crisis + jailbreak) and output (medication + diagnosis + scope)
- **4-tier memory** - SQLite persistence, rolling summaries, episodic memory, gated fact extraction
- **Model routing** - rule-based complexity classification selects Gemini model tier per turn
- **Outcome tracking** - goals, progress, strategy effectiveness measurement
- **Observability** - structured event bus, per-turn conversation analysis, observability dashboard
- **Sectioned system prompt** - structured context assembly with family profile, goals, and session history

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 3 (3-flash-preview / 3-pro-preview) via langchain-google-genai |
| Embeddings | Gemini gemini-embedding-001 via google-genai SDK |
| Vector Search | Qdrant (in-memory for dev, remote for prod) with dense + sparse + RRF |
| Reranker | FastEmbed cross-encoder (BAAI/bge-reranker-base), local ONNX inference |
| Agent | LangGraph create_react_agent (ReAct loop with tool calling) |
| Guardrails | LangGraph-native gate nodes — structured Gemini classifiers (2 calls: input + output) |
| Persistence | SQLite (default) or in-memory session store |
| API | FastAPI |
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS + shadcn/ui |

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Run
uvicorn app.main:app --reload

# Open http://localhost:8000
```

### Optional Features

Enable via environment variables:

```bash
# Disable SQLite persistence (enabled by default)
SQLITE_ENABLED=false

# Model routing (default: single model)
MODEL_ROUTING_ENABLED=true
GEMINI_MODEL_FAST=gemini-2.5-flash
GEMINI_MODEL_STANDARD=gemini-2.5-flash
GEMINI_MODEL_COMPLEX=gemini-2.5-pro
```

## Running Tests

```bash
# All tests (no API key needed, tests use mocks)
pytest tests/ -v

# Skip integration tests (require real API keys)
pytest tests/ -v -m "not integration"
```

## API Endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Process message through ReAct agent pipeline |
| POST | `/api/session/seed` | Pre-populate session with onboarding data |
| GET | `/api/session/{id}` | Session state (phase, profile, strategies) |
| GET | `/api/session/{id}/outcomes` | Outcome tracking data |
| GET | `/api/session/{id}/messages` | Full message history |
| GET | `/api/sessions` | List all sessions ordered by activity |
| GET | `/api/health` | Health check |
| GET | `/api/knowledge/topics` | Approved topic boundaries |
| GET | `/api/knowledge/documents` | All knowledge documents for resource library |

### Observability

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/observability/sessions` | All sessions with quality and flag stats |
| GET | `/api/observability/sessions/{id}` | Full session detail (messages, traces, analyses, events) |
| GET | `/api/observability/sessions/{id}/events` | Filtered event log by category |
| POST | `/api/observability/sessions/{id}/analyze` | On-demand re-analysis of all turns |

## License

MIT
