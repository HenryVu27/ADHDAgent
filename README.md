# ADHDAgent

Parent-facing ADHD coaching chatbot using a ReAct agent architecture with multi-layer safety guardrails.

NeMo input guardrails → Gemini ReAct agent with tool calling → NeMo output guardrails.

## Key Features

- **ReAct agent** with LangGraph tool calling and full observability
- **Hybrid RAG** — Qdrant dense + sparse vectors with RRF and tag boosting
- **Multi-layer guardrails** — NeMo Colang input rails + Gemini output classifiers
- **4-tier memory** — in-memory or SQLite persistence, rolling summaries, episodic memory, fact extraction
- **Model routing** — rule-based complexity classification selects Gemini model tier per turn
- **Outcome tracking** — goals, progress, strategy effectiveness measurement
- **Sectioned system prompt** — structured context assembly with family profile, goals, and session history

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini (2.5-flash-lite / flash / pro) via langchain-google-genai |
| Embeddings | Gemini gemini-embedding-001 via google-genai SDK |
| Vector Search | Qdrant (in-memory for dev, remote for prod) with dense + sparse + RRF |
| Agent | LangGraph create_react_agent (ReAct loop with tool calling) |
| Guardrails | NeMo Guardrails (Colang 1.0) + direct Gemini output classifiers |
| Persistence | SQLite (opt-in) or in-memory session store |
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
# SQLite persistence (default: in-memory)
SQLITE_ENABLED=true
SQLITE_DB_PATH=adhd_agent.db

# Model routing (default: single model)
MODEL_ROUTING_ENABLED=true
GEMINI_MODEL_FAST=gemini-2.5-flash-lite
GEMINI_MODEL_STANDARD=gemini-2.5-flash
GEMINI_MODEL_COMPLEX=gemini-2.5-pro
```

## Running Tests

```bash
# All tests (no API key needed — tests use mocks)
pytest tests/ -v

# Skip integration tests (require real API keys)
pytest tests/ -v -m "not integration"
```

## Project Structure

```
ADHDAgent/
├── app/
│   ├── main.py                        # FastAPI app + dependency wiring
│   ├── config.py                      # All settings (Gemini, Qdrant, agent, memory, routing)
│   ├── db.py                          # SQLite schema, migrations, connection management
│   ├── models/
│   │   └── schemas.py                 # Pydantic data contracts
│   ├── agent/
│   │   ├── graph.py                   # build_agent() — LangGraph ReAct agent
│   │   ├── orchestrator.py            # Session management + agent invocation
│   │   ├── hooks.py                   # pre_model_hook (guardrails + context) + post_model_hook
│   │   ├── tools.py                   # 5 tools: search, profile, outcomes, goals
│   │   ├── prompts.py                 # Sectioned system prompt template + context helpers
│   │   ├── state.py                   # CoachingState (extends MessagesState)
│   │   ├── memory.py                  # MemoryManager (summary, fact extraction, episodes)
│   │   ├── model_router.py            # Complexity classification + model selection
│   │   ├── store_protocol.py          # SessionStoreBase ABC
│   │   ├── session_store.py           # InMemorySessionStore
│   │   └── sqlite_store.py            # SQLiteSessionStore
│   ├── llm/
│   │   └── client.py                  # Gemini API wrapper (generate, extract_json, embed)
│   ├── rag/
│   │   ├── knowledge_store.py         # Qdrant document store (dense + sparse vectors)
│   │   ├── retriever.py               # Hybrid retriever (dense + sparse + RRF + tag boost)
│   │   └── query_rewriter.py          # LLM query rewriting with conversation context
│   ├── guardrails/
│   │   ├── validator.py               # Input rails (NeMo) + output rails (Gemini)
│   │   ├── gemini_provider.py         # Gemini LLM provider for NeMo
│   │   └── config/                    # NeMo config (config.yml + rails.co)
│   ├── knowledge/                     # JSON knowledge base documents
│   └── api/
│       └── routes.py                  # API endpoints
├── frontend-react/                    # React 19 + TypeScript frontend
├── tests/                             # 180 tests (unit + integration)
├── requirements.txt
└── CLAUDE.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/chat` | Process message through ReAct agent pipeline |
| POST | `/api/session/seed` | Pre-populate session with onboarding data |
| GET | `/api/session/{id}` | Session state (phase, profile, strategies) |
| GET | `/api/session/{id}/outcomes` | Outcome tracking data |
| GET | `/api/health` | Health check |
| GET | `/api/knowledge/topics` | Approved topic boundaries |
| GET | `/api/knowledge/documents` | All knowledge documents for resource library |

## License

MIT
