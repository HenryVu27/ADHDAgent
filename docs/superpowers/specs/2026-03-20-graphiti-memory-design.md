# Graphiti Memory Layer Design

Replace the current `MemoryManager` subsystem (rolling summaries, fact extraction, episodic memory, episode linking, emotion inference) with Graphiti backed by Neo4j Aura. The existing Qdrant RAG pipeline for knowledge base retrieval is untouched.

## Problem

Five weaknesses in the current memory system:

1. **No semantic retrieval over past conversations.** Rolling summaries cannot be searched. If a parent mentioned something 20 turns ago, it is lost unless the summary captured it.
2. **Rolling summary is lossy.** A 2-4 sentence summary at `SUMMARY_INTERVAL_TURNS=5` drops information rapidly. No importance-weighted retention or hierarchical summarization.
3. **Emotional episode spam.** `_run_emotional_shift_check` creates an episode for every non-neutral emotion. 5 turns of frustration = 5 near-identical episodes. No deduplication or debouncing.
4. **Episode linking is O(n) and rule-based.** Scans last 20 episodes per new episode via `asyncio.create_task` (fire-and-forget, can silently fail). Linking uses string overlap on strategy names or exact emotion match -- misses synonyms and semantic relationships.
5. **Naive context window management.** Character-based drop-oldest (`CONTEXT_MAX_CHARS: 120000`). No importance-based compression or selective retention.

## Solution

Adopt Graphiti (`graphiti-core[google-genai]`) with Neo4j Aura as the memory layer. Graphiti replaces the memory/episode/summary subsystem entirely. SQLite retains session metadata, messages, traces, goals, outcomes, and non-memory concerns.

## Architecture

```
Parent message
    |
    v
[Existing pipeline: input gate -> context assembly -> ReAct agent -> output gate]
    |
    v
post_turn_tasks() [background, non-blocking]
    |
    v
graphiti.add_episode()
    - Extracts entities (Child, Parent, Strategy, Challenge, EmotionalState, Goal)
    - Creates temporal edges (TRIED_STRATEGY, HAS_CHALLENGE, EXPERIENCED_EMOTION, etc.)
    - Deduplicates entities (5 "frustrated" messages -> 1 EmotionalState node)
    - Invalidates contradicted facts (temporal valid_at/invalid_at)
    - Links episode to all mentioned entities via MENTIONS edges
```

### Two Retrieval Systems (Complementary)

| Agent question | System | Example |
|---|---|---|
| "What ADHD strategies exist for homework refusal?" | **Qdrant** (knowledge base) via `search_knowledge_base` | Domain knowledge |
| "What has this family tried before?" | **Graphiti** (memory graph) via `search_memory` | Family-specific history |
| "Did we discuss bedtime routines before?" | **Graphiti** | Past conversation recall |
| "What evidence-based approaches work for 8-year-olds?" | **Qdrant** with age filter | Evidence-based knowledge |

## Graph Schema

### Entity Types

| Entity | Purpose | Key Attributes |
|--------|---------|----------------|
| `Child` | The child being discussed | `name`, `age`, `diagnosis_status`, `adhd_subtype` |
| `Parent` | The parent in the conversation | `name` |
| `Strategy` | An ADHD strategy mentioned or tried | `name`, `source` (knowledge base doc ID if applicable), `evidence_level` |
| `Challenge` | A specific difficulty area | `name`, `severity` |
| `EmotionalState` | A detected emotional state | `emotion` (frustrated/anxious/hopeful/etc.), `intensity` |
| `Goal` | A goal the parent set | `description`, `status` |

### Edge Types

| Edge | Connects | Temporal? | Key Attributes |
|------|----------|-----------|----------------|
| `TRIED_STRATEGY` | Parent -> Strategy | Yes | `outcome`, `notes` |
| `HAS_CHALLENGE` | Child -> Challenge | Yes | `context` |
| `EXPERIENCED_EMOTION` | Parent -> EmotionalState | Yes | `trigger` |
| `ADDRESSES_CHALLENGE` | Strategy -> Challenge | No | `effectiveness` |
| `SET_GOAL` | Parent -> Goal | Yes | `motivation` |

Temporal edges use Graphiti's `valid_at`/`invalid_at` system. When a parent says "we stopped using the timer system," the `TRIED_STRATEGY` edge gets `invalid_at` set automatically. No more `negated_strategies` extraction logic.

### Emotion Deduplication

Instead of 5 near-identical "emotional_shift" episodes, Graphiti creates one `EmotionalState(emotion="frustrated")` entity node. Each turn that expresses frustration creates a `MENTIONS` edge from the episode node to that entity. Querying "when was this parent frustrated?" is a graph traversal returning one entity with temporal metadata.

## Ingestion Pipeline

### Current (deleted)

```
post_turn_tasks()
  |-- _update_summary()              # LLM: compress into 2-4 sentences
  |-- _extract_facts()               # LLM: extract profile fields
  |-- _create_episode()              # Rule: episode if track_outcome called
  |-- _create_goal_episode()         # Rule: episode if manage_goals called
  +-- _run_emotional_shift_check()   # LLM: classify emotion, create episode
```

### New

```
post_turn_tasks()
  +-- graphiti.add_episode(
        name=f"session_{session_id}_turn_{turn}",
        episode_body=f"Parent: {user_message}\nCoach: {assistant_response}",
        source=EpisodeType.message,
        source_description=f"ADHD coaching session {session_id}",
        reference_time=datetime.now(),
        group_id=str(user_id),
      )
```

Graphiti internally runs 6-10 LLM calls (Gemini Flash) per episode for entity extraction, deduplication, relationship extraction, and temporal invalidation. Retrieval is LLM-free.

### Profile Sync

`FamilyProfile` in SQLite remains the canonical source for the system prompt and RAG age filtering. Graphiti is not authoritative for profile data -- it discovers facts, and we selectively sync them into SQLite.

**Direction:** One-way, Graphiti -> SQLite. The `update_family_profile` agent tool continues to write directly to SQLite (unchanged). Graphiti may independently extract the same facts from conversation text, but SQLite wins on conflict since it reflects explicit agent actions.

**Mechanism:** After `add_episode()` completes, run a Cypher query against Neo4j to read the latest `Child` entity attributes for the current `group_id`:

```python
async def _sync_profile(self, session_id: str, user_id: int) -> None:
    """Sync Graphiti-discovered facts into SQLite FamilyProfile."""
    query = """
    MATCH (c:Child) WHERE c.group_id = $group_id
    RETURN c.name AS child_name, c.age AS child_age,
           c.diagnosis_status AS diagnosis_status, c.adhd_subtype AS adhd_subtype
    ORDER BY c.created_at DESC LIMIT 1
    """
    result = await self._graphiti.driver.execute_query(query, {"group_id": str(user_id)})
    if not result.records:
        return
    record = result.records[0]
    # Only update SQLite fields that are currently empty
    current_profile = (await self._store.get(session_id)).family_profile
    updates = {}
    for field in ("child_name", "child_age", "diagnosis_status", "adhd_subtype"):
        graph_val = record.get(field)
        sqlite_val = getattr(current_profile, field, None)
        if graph_val and not sqlite_val:
            updates[field] = graph_val
    if updates:
        await self._store.update_profile(session_id, **updates)
```

**Negated strategies:** When Graphiti sets `invalid_at` on a `TRIED_STRATEGY` edge (parent says "we stopped using timers"), the profile sync does not need to handle this -- the `search_memory` tool and context assembly already surface only valid edges (`invalid_at is None`). The RAG outcome boost continues to read from SQLite `outcomes` table (unchanged).

### end_of_session_tasks

**Deleted.** The current `end_of_session_tasks` generates a `UserSummary` longitudinal summary for returning users. With Graphiti, the graph persists across sessions partitioned by `group_id=str(user_id)`. A `graphiti.search()` call with the user's `group_id` naturally returns cross-session facts without needing a separate longitudinal summary.

The associated store methods (`get_user_summary`, `save_user_summary`) are removed from `SessionStoreBase`. `get_user_episodes` and `get_user_outcomes` remain (used by observability routes).

## Retrieval

### Passive: Context Assembly (hooks.py)

Current code reads summaries and episodes from SQLite:

```python
latest_summary = await session_store.get_latest_summary(session_id)
recent_episodes = await session_store.get_recent_episodes(session_id, limit=5)
```

New code queries Graphiti for semantically relevant, temporally valid facts:

```python
relevant_edges = await graphiti.search(
    last_user_message,
    group_ids=[str(user_id)],    # tenant isolation -- only this family's data
    num_results=settings.GRAPHITI_CONTEXT_RESULTS,
)
memory_context = "\n".join([
    f"- {edge.fact} (since {edge.valid_at.strftime('%b %Y')})"
    for edge in relevant_edges
    if edge.invalid_at is None
])
```

`group_ids` provides tenant isolation (only this family's data). If a center-node search is needed later (e.g., reranking by proximity to the `Child` entity), the child node UUID can be looked up from the graph and passed as `center_node_uuid` alongside `group_ids`.

Cross-session context (`<prior-sessions>` block) simplifies -- the graph persists across sessions with the same `group_id`, so the same `graphiti.search()` call returns cross-session facts naturally.

### Active: search_memory Tool

New tool in `tools.py`:

```python
@tool
async def search_memory(query: str, config: RunnableConfig = None) -> str:
    """Search conversation memory for what this family has shared,
    tried, or experienced. Use when you need to recall past
    discussions, strategy outcomes, emotional patterns, or
    family context from previous turns or sessions."""
    user_id = _get_user_id(config)  # extract from RunnableConfig configurable
    edges = await graphiti.search(
        query,
        group_ids=[str(user_id)],   # tenant isolation
        num_results=settings.GRAPHITI_SEARCH_RESULTS,
    )
    ...
```

`user_id` is extracted from `RunnableConfig.configurable` (same pattern as `session_id`). The tool factory signature becomes `create_tools(retriever, session_store, graphiti_client=None)`.

System prompt updated to describe when to use `search_memory` vs `search_knowledge_base`.

### Context Window Improvement

The memory section of the system prompt becomes compact structured facts instead of raw summary + episode text. Drop-oldest trimming of LangChain message history stays as-is.

## Error Handling

**Ingestion failures:** `graphiti.add_episode()` makes 6-10 LLM calls. Any can fail (rate limits, Neo4j connection drops, Gemini timeout). Strategy:

- Wrap in `asyncio.wait_for(timeout=settings.GRAPHITI_INGESTION_TIMEOUT_S)`.
- On failure, log the error and drop the episode. No retry -- the next turn's episode will capture ongoing context. Lost episodes are acceptable because:
  - The conversation text is always persisted in SQLite `messages` table regardless.
  - Graphiti entity extraction is cumulative -- if "child is 8" appears in turn 3 and turn 7, losing turn 3's episode still gets the entity from turn 7.
- If Neo4j is unreachable at startup (`build_indices_and_constraints` fails), set `graphiti_client = None` and fall back to SQLite memory (same as `GRAPHITI_ENABLED=False`).
- Emit `EventBus` event on both success and failure for observability.

**Retrieval failures:** If `graphiti.search()` fails in context assembly, fall back to empty memory context (the agent still has the conversation message history). If it fails in the `search_memory` tool, return a user-friendly error string ("Memory search is temporarily unavailable").

## Observability

New events emitted via `EventBus`:

| Event | When | Payload |
|-------|------|---------|
| `graphiti_episode_ingested` | After successful `add_episode()` | `session_id`, `turn`, `duration_ms`, `entity_count` |
| `graphiti_ingestion_failed` | After failed `add_episode()` | `session_id`, `turn`, `error` |
| `graphiti_search_completed` | After `graphiti.search()` in context assembly or tool | `session_id`, `query`, `result_count`, `duration_ms` |
| `graphiti_profile_synced` | After profile sync updates SQLite | `session_id`, `fields_updated` |

These replace the current memory events (`summary_updated`, `facts_extracted`, `episode_created`, `emotional_shift`, `emotion_inferred`, `strategy_negated`, `longitudinal_summary_updated`).

## Configuration

New fields in `app/config.py`:

```python
# Neo4j
NEO4J_URI: str = "bolt://localhost:7687"
NEO4J_USER: str = "neo4j"
NEO4J_PASSWORD: str = ""
NEO4J_DATABASE: str = "neo4j"

# Graphiti
GRAPHITI_ENABLED: bool = True
GRAPHITI_LLM_MODEL: str = ""          # Defaults to GEMINI_UTILITY_MODEL
GRAPHITI_EMBEDDING_MODEL: str = ""    # Defaults to GEMINI_EMBEDDING_MODEL
GRAPHITI_GROUP_ID_SOURCE: str = "user_id"
GRAPHITI_CONTEXT_RESULTS: int = 10
GRAPHITI_SEARCH_RESULTS: int = 10
GRAPHITI_INGESTION_TIMEOUT_S: float = 30.0
```

`GRAPHITI_ENABLED` feature flag: when `False` or Neo4j credentials missing, falls back to current SQLite memory. Enables local dev without Neo4j and test isolation.

`GRAPHITI_LLM_MODEL` and `GRAPHITI_EMBEDDING_MODEL` default to `GEMINI_UTILITY_MODEL` and `GEMINI_EMBEDDING_MODEL` respectively via the existing `@model_validator(mode="after")` pattern in `_fill_model_defaults`.

## Initialization

In `app/main.py` `build_dependencies()`:

```python
if settings.GRAPHITI_ENABLED and settings.NEO4J_PASSWORD:
    graphiti_client = Graphiti(
        settings.NEO4J_URI,
        settings.NEO4J_USER,
        settings.NEO4J_PASSWORD,
        llm_client=GeminiClient(
            config=LLMConfig(api_key=settings.GEMINI_API_KEY, model=graphiti_llm_model)
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=settings.GEMINI_API_KEY,
                embedding_model=graphiti_embedding_model,
            )
        ),
    )
    await graphiti_client.build_indices_and_constraints()
else:
    graphiti_client = None
```

Shutdown: `await graphiti_client.close()` in FastAPI shutdown handler.

Graph partitioned by `group_id=str(user_id)` so families do not leak context.

## File Changes

### Modified

| File | Change |
|------|--------|
| `app/config.py` | Add `NEO4J_*` and `GRAPHITI_*` settings |
| `app/main.py` | Initialize Graphiti client, pass to MemoryManager and tool factory, add shutdown handler |
| `app/agent/memory.py` | Gut internals: `post_turn_tasks` calls `graphiti.add_episode()`. Delete `_update_summary`, `_extract_facts`, `_create_episode`, `_create_goal_episode`, `_run_emotional_shift_check`, `_infer_emotion`, `_link_episode`, `end_of_session_tasks`. Add `_sync_profile`. |
| `app/agent/hooks.py` | `prepare_context` queries `graphiti.search()` instead of SQLite summaries/episodes. Remove `<prior-sessions>` special-casing. |
| `app/agent/tools.py` | Add `search_memory` tool |
| `app/agent/prompts.py` | Update system prompt: replace summary/episodes section with graph memory context. Add `search_memory` tool description. |
| `app/agent/graph.py` | Add `search_memory` to tool list |
| `app/agent/store_protocol.py` | Remove: `get_latest_summary`, `save_summary`, `add_episode`, `get_recent_episodes`, `get_episodes_with_ids`, `add_episode_link`, `get_episode_links`, `get_user_summary`, `save_user_summary` |
| `app/agent/sqlite_store.py` | Remove implementations of deleted protocol methods |
| `app/agent/session_store.py` | Remove in-memory implementations of deleted protocol methods |
| `app/models/schemas.py` | `SessionSummary`, `EpisodicMemory`, `EpisodeLink` become unused. Add custom entity/edge Pydantic models for Graphiti. |

### New

| File | Purpose |
|------|---------|
| `app/agent/graphiti_client.py` | Factory function to create configured Graphiti instance. Custom entity and edge type definitions. |

### Unchanged

- `app/rag/*` -- entire Qdrant retrieval pipeline
- `app/guardrails/*`
- `app/api/*` -- observability routes (`observability_routes.py`) read traces/analyses from SQLite, not episodes or summaries directly. No changes needed. The `SessionDetailResponse` schema includes `events` from `EventBus` which will now contain Graphiti events instead of memory events.
- `app/agent/analyzer.py`
- `app/agent/event_bus.py`
- `app/db.py` -- no migration to drop tables (keep for data safety), just stop writing to `session_summaries`, `episodes`, `episode_links`

### Tests

| File | Change |
|------|--------|
| `tests/test_memory.py` | Rewrite: mock `graphiti.add_episode()` and `graphiti.search()` instead of SQLite |
| `tests/test_agent_hooks.py` | Update context assembly assertions for graph-sourced memory |
| `tests/test_agent_tools.py` | Add tests for `search_memory` tool |
| Other test files | Unchanged |

## How Each Weakness Is Addressed

| # | Weakness | How Graphiti Fixes It |
|---|----------|-----------------------|
| 1 | No semantic search over past turns | `graphiti.search()` provides hybrid vector+BM25+graph traversal over all ingested conversation turns. A mention from turn 5 is retrievable at turn 50. |
| 2 | Lossy rolling summary | No more summaries. Facts live as graph nodes/edges with full fidelity. Nothing is compressed away. |
| 3 | Emotional episode spam | Entity deduplication: 5 "frustrated" messages create 1 `EmotionalState` node with 5 temporal `MENTIONS` edges, not 5 separate episodes. |
| 4 | O(n) rule-based episode linking | Graphiti automatically links episodes to entities via `MENTIONS` edges. Relationships are structural (graph edges), not rule-based string matching. Semantic deduplication handles synonyms. |
| 5 | Naive context window management | Memory section becomes compact structured facts from `graphiti.search()` instead of raw summary + episode text. More information-dense, less context pressure. |

## Cost

- **Ingestion**: 6-10 Gemini Flash LLM calls per turn (~$0.001-0.003/turn). Non-blocking background task.
- **Retrieval**: Zero LLM calls. Vector + BM25 + graph traversal only. P95 ~300ms.
- **Neo4j Aura free tier**: 200K nodes, 400K relationships. Sufficient for early usage.

## Data Migration

**No backfill of existing data.** Existing sessions in SQLite are not replayed into Graphiti. A returning user's first post-migration session will start with an empty graph for their `group_id`. This is acceptable because:
- The app is pre-production; no real user data to preserve.
- If backfill is needed later, a script can replay SQLite messages as Graphiti episodes using `add_episode_bulk()`.

The old SQLite tables (`session_summaries`, `episodes`, `episode_links`, `user_summaries`) are not dropped. They remain in the schema but are no longer written to.

## Known Limitations

- Graphiti pronoun resolution is imperfect (issue #1171). Mitigated by formatting turns as `Parent: ... Coach: ...` with explicit speaker labels.
- `add_episode` latency is seconds (multiple LLM round-trips). Must be fire-and-forget, not blocking the response path.
- Neo4j Aura free tier has size limits. Monitor node/edge counts. Upgrade path: Aura Professional or self-hosted.
- Orphaned entities can accumulate on episode deletion (issue #1083). Periodic cleanup may be needed.
- `update_family_profile` tool writes to SQLite only, not to the graph. Graph discovers facts independently from conversation text. Minor divergence is possible but acceptable since SQLite is canonical for profile.
