# Web Search Tool Design

**Date**: 2026-03-20
**Status**: Draft

## Problem

The agent's knowledge base is curated and static. When parents ask about recent research, current resources, or topics not covered by the KB, the agent can only answer from its training data (which has a knowledge cutoff) or deflect. A web search tool fills this recency and coverage gap.

Examples:
- "What does the latest research say about screen time and ADHD?"
- "Are there any ADHD support groups near Austin, TX?"
- "What are the new IEP accommodation guidelines for 2026?"

## Design

### Tool: `search_web`

A new LangChain `@tool` in `app/agent/tools.py` that performs a grounded web search via the `google-genai` SDK's built-in `GoogleSearch` tool.

**Signature:**

```python
@tool
async def search_web(
    query: str,
    config: RunnableConfig = None,
) -> str:
    """Search the web for current information about ADHD, parenting, or child development.

    Use this ONLY when the knowledge base does not have what you need — for recent
    research, current events, local resources, or topics not covered by curated documents.
    Always try search_knowledge_base first.

    Use specific search terms, not full questions.
    Good: "IEP accommodation guidelines 2026", "ADHD support groups Austin TX"
    Bad: "What should I do about my child's school?"

    Args:
        query: Specific search query using topic keywords
    """
```

**Implementation:**

1. Guard: if `settings.WEB_SEARCH_ENABLED` is False or `gemini_client` is None, return a message saying web search is disabled.
2. Scope the query: prepend `"ADHD: "` to bias results toward the domain. Use a short prefix to avoid diluting geographic or specific queries (e.g., "ADHD support groups Austin TX" should not become "ADHD parenting child development: ADHD support groups Austin TX").
3. Call `GeminiClient.search_web(scoped_query, timeout=settings.WEB_SEARCH_TIMEOUT_S)`.
4. Format the result as synthesized answer + numbered source list.
5. On error: return a user-friendly message, never raise.

**Return format to agent:**

```
[Web Search Results]

<synthesized answer from Flash with GoogleSearch grounding>

Sources:
[1] Title — URL
[2] Title — URL
...
```

If the response has no grounding metadata (no sources), return the answer text without a Sources section.

### GeminiClient change

Add a `search_web` method to `app/llm/client.py`:

```python
async def search_web(
    self,
    query: str,
    timeout: float | None = None,
) -> tuple[str, list[dict]]:
    """Perform a grounded web search via Gemini Flash + GoogleSearch.

    Returns (answer_text, sources) where sources is a list of
    {"title": str, "url": str} dicts from grounding metadata.
    """
```

Key implementation details:
- Apply `_retry_policy` to the `generate_content` call, consistent with all other SDK calls in the client.
- Use `asyncio.wait_for(coro, timeout=timeout)` for timeout enforcement, same pattern as `generate()`.
- Extract grounding metadata from `response.candidates[0].grounding_metadata.grounding_chunks`. Each chunk has a `.web` attribute with `.uri` and `.title` fields. If `grounding_metadata` is None or `grounding_chunks` is empty, return an empty sources list.
- Use `settings.GEMINI_UTILITY_MODEL` (Flash) for the search call.

This keeps the google-genai SDK interaction centralized in the client, consistent with `generate`, `extract_json`, and `embed`.

### Tool registration

In `create_tools()`, the new tool needs access to a `GeminiClient` instance.

**Approach**: Add `gemini_client` as an **optional** parameter (default `None`) to `create_tools()`. When `None`, the `search_web` tool returns a "web search is not available" message. This preserves backward compatibility with existing call sites (tests, eval runners) that don't pass a client.

`app/main.py` already creates a `GeminiClient` and passes it through. The docstring ("Create the 7 agent tools") updates to 8.

### System prompt update

Add `search_web` to the `<tools>` section in `app/agent/prompts.py`:

```
- **search_web**: Search the web for current information not in the knowledge base. Use for: recent research, current events, local resources, school policy questions, anything time-sensitive. Always try search_knowledge_base first — only use search_web when the KB doesn't cover the topic or the parent needs up-to-date information.
```

Also add a step to the search results workflow (after step 4):

```
5. If the knowledge base has no relevant results and the question is about recent information or local resources, use search_web.
```

### Streaming status

Already handled — `app/agent/orchestrator.py` lines 632-635 already have a status handler for `search_web` that shows `"Searching the web for '{query}'..."`.

### Config

Uses existing settings, no new config needed:
- `WEB_SEARCH_ENABLED: bool = True` (already exists)
- `WEB_SEARCH_TIMEOUT_S: float = 15.0` (already exists)

New optional setting for per-session throttling:
- `WEB_SEARCH_MAX_PER_SESSION: int = 3` — Maximum web searches allowed per conversation session. Prevents cost runaway. The tool returns a "search limit reached" message after this count. Tracked via a simple counter in the tool closure (keyed by session_id).

### Guardrails

The query scoping (prepending `"ADHD: "`) is a soft guardrail. The existing output gate (medication, diagnosis, scope checks) catches any problematic content in the final response. No additional guardrails needed — the agent's own judgment about when to search, combined with these two layers, is sufficient.

## Files to modify

| File | Change |
|------|--------|
| `app/llm/client.py` | Add `search_web()` method with retry policy |
| `app/agent/tools.py` | Add `search_web` tool, add optional `gemini_client` param to `create_tools()` |
| `app/main.py` | Pass `gemini_client` to `create_tools()` |
| `app/agent/prompts.py` | Add `search_web` to `<tools>` section + search workflow step 5 |
| `app/config.py` | Add `WEB_SEARCH_MAX_PER_SESSION` setting |
| `tests/test_agent_tools.py` | Add tests for `search_web` (mocked) |

Backward-compatible call sites (no changes needed due to `gemini_client=None` default):
- `eval/runners/response_runner.py`
- `tests/test_integration_pipeline.py`

## Testing

- Mock `GeminiClient.search_web()` to return canned responses.
- Test: tool returns formatted results with sources when search succeeds.
- Test: tool returns answer without Sources section when grounding metadata is empty.
- Test: tool returns fallback message when `WEB_SEARCH_ENABLED=False`.
- Test: tool returns fallback message when `gemini_client` is None.
- Test: tool returns error message on timeout/exception (never raises).
- Test: query scoping prepends "ADHD: " to the query.
- Test: per-session limit is enforced after `WEB_SEARCH_MAX_PER_SESSION` calls.

## Out of scope

- Result caching (can add later if search costs become a concern).
- Search result ranking or filtering beyond what Gemini grounding provides.
- Displaying source links as clickable cards in the frontend (the agent cites them inline in its text response).
