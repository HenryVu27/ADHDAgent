# ADHDAgent

## Git Conventions

- Never add `Co-Authored-By` lines to commits.
- No conventional commit prefixes (`fix:`, `feat:`, `test:`, etc.). Just write a plain commit message.

Parent-facing ADHD coaching chatbot using a ReAct agent architecture:
Input gate (crisis + jailbreak) -> Context assembly -> Gemini ReAct agent with tool calling -> Output gate (medication + diagnosis + scope).

## Python Environment

**Always use the `adhd312` venv** (Python 3.12) for running, testing, and installing packages.
System Python is 3.14 which has compatibility issues with some dependencies.

```bash
# Run tests
./adhd312/Scripts/python.exe -m pytest tests/ -v

# Run the app
./adhd312/Scripts/python.exe -m uvicorn app.main:app --reload

# Install packages
./adhd312/Scripts/pip.exe install <package>
```

## Architecture

```
Parent message
    |
    v
Input Gate (app/guardrails/validator.py — InputGate)
    Single structured Gemini call classifying crisis + jailbreak.
    Crisis -> 988/911 resources. Jailbreak -> rejection.
    Blocked messages skip the agent entirely.
    |
    v
Context Assembly (app/agent/hooks.py — prepare_context)
    Family profile, goals, outcomes, rolling summary,
    episodic memory injected into sectioned system prompt.
    Trims conversation history to last N turns.
    |
    v
Gemini ReAct Agent (app/agent/graph.py — inner create_react_agent)
    Uses Pro model with thinking mode (budget_tokens configurable).
    Agent reasons about the parent's message and decides to:
      - Respond directly (greetings, acknowledgments, clarifying questions)
      - Call search_knowledge_base() to find evidence-based strategies
      - Call update_family_profile() when learning about the family
      - Call track_outcome() when parent reports strategy results
      - Call manage_goals() to set/track/complete goals
    Loops until the agent produces a final text response.
    |
    v
Output Gate (app/guardrails/validator.py — OutputGate)
    Single structured Gemini call classifying medication, diagnosis, scope.
    Violations replaced with safe fallback response.
    |
    v
Background tasks (app/agent/memory.py + app/agent/analyzer.py)
    Rolling conversation summary (every N turns).
    Gated fact extraction from user messages.
    Episodic memory for outcome events.
    Turn quality analysis (ConversationAnalyzer).
    EventBus emit for observability.
    |
    v
Response to parent (with PipelineTrace for frontend)
```

## Tech Stack

- **LLM**: Google Gemini 2.5 — Pro (gemini-2.5-pro) for ReAct agent with thinking mode, Flash (gemini-2.5-flash) for utilities (guardrails, memory, analyzer). Tenacity retry on all API calls.
- **Embeddings**: Gemini gemini-embedding-001 via google-genai SDK
- **Vector Search**: Qdrant (in-memory for dev, remote for prod) with dense + sparse + RRF
- **Agent**: LangGraph create_react_agent (ReAct loop with tool calling)
- **Guardrails**: LangGraph-native gate nodes — structured Gemini classifiers for input (crisis + jailbreak) and output (medication + diagnosis + scope)
- **Reranker**: FastEmbed cross-encoder (BAAI/bge-reranker-base), local ONNX inference
- **Persistence**: aiosqlite (default) or in-memory session store
- **API**: FastAPI
- **Frontend**: React 19 + TypeScript + Vite + Tailwind CSS + shadcn/ui

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run
uvicorn app.main:app --reload

# Open browser
# http://localhost:8000
```

## Testing Policy

- **Never run integration tests or E2E tests** that require a real API key (e.g., `GEMINI_API_KEY`). These consume limited API quota.
- **Always ask the user before running any tests**, even mock-only unit tests.

## Running Tests

```bash
# All tests (no API key needed - tests use mocks)
pytest tests/ -v

# Specific test files
pytest tests/test_api.py -v
pytest tests/test_agent_tools.py -v
pytest tests/test_agent_hooks.py -v
pytest tests/test_agent_orchestrator.py -v
pytest tests/test_session_store.py -v
pytest tests/test_sqlite_store.py -v
pytest tests/test_guardrails.py -v
pytest tests/test_rag.py -v
pytest tests/test_prompts.py -v
pytest tests/test_memory.py -v
pytest tests/test_event_bus.py -v
pytest tests/test_analyzer.py -v
pytest tests/test_observability_api.py -v

# Skip integration tests (require real API keys)
pytest tests/ -v -m "not integration"

# E2E conversation tests (require GEMINI_API_KEY)
pytest tests/test_e2e_conversations.py -v -m integration -s
```

## Adding Knowledge Documents

1. Create a JSON file in `app/knowledge/`
2. Each document needs: `id`, `name`, `description`, `tags`, and either `steps` (strategies) or `key_points` (guidance/facts)
3. Add `source` for attribution
4. The knowledge base auto-loads all JSON files on startup
5. Qdrant indexes rebuild automatically on startup

## Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | App startup + dependency wiring |
| `app/config.py` | All settings (Gemini, Qdrant, agent, memory, thinking) |
| `app/db.py` | aiosqlite schema, migrations, connection management |
| `app/models/schemas.py` | All Pydantic data contracts |
| `app/api/routes.py` | Core API endpoints |
| `app/api/observability_routes.py` | Observability API endpoints |
| **Agent** | |
| `app/agent/graph.py` | `build_agent()` — custom StateGraph: input_gate → react_agent → output_gate |
| `app/agent/orchestrator.py` | `AgentOrchestrator` — session management + agent invocation |
| `app/agent/tools.py` | 5 tools: search_knowledge_base, get/update_family_profile, track_outcome, manage_goals |
| `app/agent/hooks.py` | `create_prepare_context()` — context assembly + conversation trimming (pre_model_hook) |
| `app/agent/prompts.py` | Sectioned system prompt template + context assembly helpers |
| `app/agent/state.py` | `CoachingState` extending `MessagesState` |
| `app/agent/memory.py` | `MemoryManager` — rolling summary, fact extraction, episodic memory |
| `app/agent/store_protocol.py` | `SessionStoreBase` ABC — interface for all session stores |
| `app/agent/session_store.py` | `InMemorySessionStore` implementation |
| `app/agent/sqlite_store.py` | `SQLiteSessionStore` implementation |
| `app/agent/event_bus.py` | `EventBus` - structured observability events |
| `app/agent/analyzer.py` | `ConversationAnalyzer` - per-turn quality analysis |
| **LLM** | |
| `app/llm/client.py` | Gemini API wrapper (generate, extract_json, embed) |
| **RAG** | |
| `app/rag/knowledge_store.py` | Qdrant-based document store (dense + sparse vectors) |
| `app/rag/retriever.py` | Hybrid retriever (dense + sparse + RRF + tag boost + optional reranking) |
| `app/rag/reranker.py` | FastEmbed cross-encoder reranker (local ONNX, enabled by default) |
| `app/rag/query_rewriter.py` | LLM query rewriting with conversation context |
| **Guardrails** | |
| `app/guardrails/validator.py` | `InputGate` (crisis + jailbreak) + `OutputGate` (medication + diagnosis + scope) |

## RAG Conventions

- `search_knowledge_base` returns structured output (metadata + full steps/key_points). Do not truncate document content.
- Age filtering auto-applies from the family profile's `child_age`. The `_build_filter` in `knowledge_store.py` must always include `"all"` alongside the derived age range.
- `RAG_RERANKER` controls the reranker: `"none"` | `"cross_encoder"` | `"colbert"`, default `"cross_encoder"`. Cross-encoder uses FastEmbed local ONNX (BAAI/bge-reranker-base); ColBERT uses precomputed multi-vectors in Qdrant. When adding retrieval pipeline steps, insert between hybrid search and facet computation in `retriever.py`.
- Technical design rationale lives in `docs/rag_design_decisions.md`.
