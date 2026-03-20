# Web Search Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **IMPORTANT:** Execute this plan in a **separate git worktree** using the superpowers:using-git-worktrees skill. Branch name: `feature/web-search-tool`.

**Goal:** Add a `search_web` tool that lets the ReAct agent perform grounded web searches via Gemini Flash + GoogleSearch when the curated knowledge base doesn't cover the parent's question.

**Architecture:** New `search_web()` method on `GeminiClient` wraps the google-genai SDK's `GoogleSearch` grounding. A new LangChain `@tool` in the existing `create_tools()` factory exposes it to the agent. Optional `gemini_client` parameter keeps backward compatibility.

**Tech Stack:** google-genai SDK (`types.GoogleSearch`), LangChain `@tool`, asyncio, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-web-search-tool-design.md`

---

### Task 1: Add `WEB_SEARCH_MAX_PER_SESSION` config

**Files:**
- Modify: `app/config.py:61-63`

- [ ] **Step 1: Add the setting**

In `app/config.py`, after `WEB_SEARCH_TIMEOUT_S` (line 63), add:

```python
    WEB_SEARCH_MAX_PER_SESSION: int = 3
```

- [ ] **Step 2: Verify app still starts**

Run: `python -c "from app.config import settings; print(settings.WEB_SEARCH_MAX_PER_SESSION)"`
Expected: `3`

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "Add WEB_SEARCH_MAX_PER_SESSION config setting"
```

---

### Task 2: Add `GeminiClient.search_web()` method with tests

**Files:**
- Modify: `app/llm/client.py:54-178`
- Create: `tests/test_llm_search_web.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm_search_web.py`:

```python
"""Tests for GeminiClient.search_web() method."""

import asyncio
import pytest
from unittest.mock import MagicMock, patch


class TestGeminiClientSearchWeb:

    @pytest.fixture
    def mock_response(self):
        """Build a mock google-genai response with grounding metadata."""
        chunk = MagicMock()
        chunk.web.uri = "https://example.com/adhd-research"
        chunk.web.title = "ADHD Research 2026"

        metadata = MagicMock()
        metadata.grounding_chunks = [chunk]

        candidate = MagicMock()
        candidate.grounding_metadata = metadata

        response = MagicMock()
        response.text = "Recent studies show screen time impacts vary by age."
        response.candidates = [candidate]
        return response

    @pytest.fixture
    def mock_response_no_grounding(self):
        """Response with text but no grounding metadata."""
        candidate = MagicMock()
        candidate.grounding_metadata = None

        response = MagicMock()
        response.text = "General answer without sources."
        response.candidates = [candidate]
        return response

    @patch("app.llm.client.genai")
    async def test_returns_answer_and_sources(self, mock_genai, mock_response):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        from app.llm.client import GeminiClient
        with patch.object(GeminiClient, "__init__", lambda self: None):
            client = GeminiClient()
            client._client = mock_client
            client._model = "gemini-2.5-flash"

        answer, sources = await client.search_web("ADHD screen time")
        assert "screen time" in answer.lower()
        assert len(sources) == 1
        assert sources[0]["title"] == "ADHD Research 2026"
        assert sources[0]["url"] == "https://example.com/adhd-research"

    @patch("app.llm.client.genai")
    async def test_returns_empty_sources_when_no_grounding(self, mock_genai, mock_response_no_grounding):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response_no_grounding
        mock_genai.Client.return_value = mock_client

        from app.llm.client import GeminiClient
        with patch.object(GeminiClient, "__init__", lambda self: None):
            client = GeminiClient()
            client._client = mock_client
            client._model = "gemini-2.5-flash"

        answer, sources = await client.search_web("ADHD general")
        assert "General answer" in answer
        assert sources == []

    @patch("app.llm.client.genai")
    async def test_returns_empty_on_no_text(self, mock_genai):
        response = MagicMock()
        response.text = None
        response.candidates = []

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = response
        mock_genai.Client.return_value = mock_client

        from app.llm.client import GeminiClient
        with patch.object(GeminiClient, "__init__", lambda self: None):
            client = GeminiClient()
            client._client = mock_client
            client._model = "gemini-2.5-flash"

        answer, sources = await client.search_web("ADHD test")
        assert answer == ""
        assert sources == []

    @patch("app.llm.client.genai")
    async def test_raises_on_api_error(self, mock_genai):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ConnectionError("network down")
        mock_genai.Client.return_value = mock_client

        from app.llm.client import GeminiClient
        with patch.object(GeminiClient, "__init__", lambda self: None):
            client = GeminiClient()
            client._client = mock_client
            client._model = "gemini-2.5-flash"

        with pytest.raises(ConnectionError):
            await client.search_web("test query", timeout=2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_search_web.py -v`
Expected: FAIL — `GeminiClient` has no `search_web` method

- [ ] **Step 3: Implement `GeminiClient.search_web()`**

Add to `app/llm/client.py` after the `extract_json` method (after line 139):

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
        try:
            from google.genai import types

            config = GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
                max_output_tokens=1024,
            )
            coro = asyncio.to_thread(
                _retry_policy(self._client.models.generate_content),
                model=self._model,
                contents=query,
                config=config,
            )
            if timeout is not None:
                response = await asyncio.wait_for(coro, timeout=timeout)
            else:
                response = await coro

            # Extract answer text
            answer = response.text or ""

            # Extract grounding sources
            sources: list[dict] = []
            if response.candidates:
                metadata = response.candidates[0].grounding_metadata
                if metadata and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            sources.append({
                                "title": getattr(chunk.web, "title", "") or "",
                                "url": getattr(chunk.web, "uri", "") or "",
                            })

            return answer, sources
        except Exception as e:
            logger.error(f"Gemini search_web failed: {e}")
            raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_search_web.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/llm/client.py tests/test_llm_search_web.py
git commit -m "Add GeminiClient.search_web() with GoogleSearch grounding"
```

---

### Task 3: Add `search_web` tool to `create_tools()` with tests

**Files:**
- Modify: `app/agent/tools.py:123-513`
- Modify: `tests/test_agent_tools.py`

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_agent_tools.py`:

```python
from unittest.mock import patch as sync_patch


class TestSearchWeb:

    @pytest.fixture
    async def mock_gemini(self):
        client = AsyncMock()
        client.search_web = AsyncMock(return_value=(
            "Recent research shows ADHD screen time effects vary.",
            [{"title": "ADHD Study 2026", "url": "https://example.com/study"}],
        ))
        return client

    @pytest.fixture
    async def web_tool_set(self, mock_retriever, session_store, mock_gemini):
        tools_list = create_tools(
            retriever=mock_retriever,
            session_store=session_store,
            gemini_client=mock_gemini,
        )
        return {t.name: t for t in tools_list}

    async def test_returns_formatted_results_with_sources(self, web_tool_set, mock_gemini):
        result = await web_tool_set["search_web"].ainvoke(
            {"query": "ADHD screen time research"},
            config=_config(),
        )
        assert "[Web Search Results]" in result
        assert "screen time" in result.lower()
        assert "Sources:" in result
        assert "ADHD Study 2026" in result
        assert "https://example.com/study" in result
        # Verify query was scoped with "ADHD: " prefix
        call_query = mock_gemini.search_web.call_args[1].get("query") or mock_gemini.search_web.call_args[0][0]
        assert call_query.startswith("ADHD: ")

    async def test_returns_answer_without_sources_section(self, mock_retriever, session_store):
        no_sources_gemini = AsyncMock()
        no_sources_gemini.search_web = AsyncMock(return_value=(
            "General ADHD information.",
            [],
        ))
        tools_list = create_tools(
            retriever=mock_retriever,
            session_store=session_store,
            gemini_client=no_sources_gemini,
        )
        ts = {t.name: t for t in tools_list}
        result = await ts["search_web"].ainvoke(
            {"query": "ADHD general"},
            config=_config(),
        )
        assert "[Web Search Results]" in result
        assert "General ADHD information" in result
        assert "Sources:" not in result

    async def test_disabled_when_setting_false(self, mock_retriever, session_store, mock_gemini):
        tools_list = create_tools(
            retriever=mock_retriever,
            session_store=session_store,
            gemini_client=mock_gemini,
        )
        ts = {t.name: t for t in tools_list}
        with sync_patch("app.agent.tools.settings") as mock_settings:
            mock_settings.WEB_SEARCH_ENABLED = False
            mock_settings.WEB_SEARCH_TIMEOUT_S = 15.0
            mock_settings.WEB_SEARCH_MAX_PER_SESSION = 3
            result = await ts["search_web"].ainvoke(
                {"query": "anything"},
                config=_config(),
            )
        assert "not available" in result.lower() or "disabled" in result.lower()

    async def test_disabled_when_no_gemini_client(self, mock_retriever, session_store):
        tools_list = create_tools(
            retriever=mock_retriever,
            session_store=session_store,
            gemini_client=None,
        )
        ts = {t.name: t for t in tools_list}
        result = await ts["search_web"].ainvoke(
            {"query": "anything"},
            config=_config(),
        )
        assert "not available" in result.lower() or "disabled" in result.lower()

    async def test_returns_error_on_exception(self, mock_retriever, session_store):
        failing_gemini = AsyncMock()
        failing_gemini.search_web = AsyncMock(side_effect=TimeoutError("timed out"))
        tools_list = create_tools(
            retriever=mock_retriever,
            session_store=session_store,
            gemini_client=failing_gemini,
        )
        ts = {t.name: t for t in tools_list}
        result = await ts["search_web"].ainvoke(
            {"query": "test"},
            config=_config(),
        )
        assert "Error:" in result or "error" in result.lower()
        assert "Traceback" not in result

    async def test_per_session_limit(self, mock_retriever, session_store, mock_gemini):
        tools_list = create_tools(
            retriever=mock_retriever,
            session_store=session_store,
            gemini_client=mock_gemini,
        )
        ts = {t.name: t for t in tools_list}
        with sync_patch("app.agent.tools.settings") as mock_settings:
            mock_settings.WEB_SEARCH_ENABLED = True
            mock_settings.WEB_SEARCH_TIMEOUT_S = 15.0
            mock_settings.WEB_SEARCH_MAX_PER_SESSION = 2
            # First two calls succeed
            r1 = await ts["search_web"].ainvoke({"query": "q1"}, config=_config("limit_test"))
            r2 = await ts["search_web"].ainvoke({"query": "q2"}, config=_config("limit_test"))
            assert "[Web Search Results]" in r1
            assert "[Web Search Results]" in r2
            # Third call hits the limit
            r3 = await ts["search_web"].ainvoke({"query": "q3"}, config=_config("limit_test"))
            assert "limit" in r3.lower()
            # Different session is unaffected
            r4 = await ts["search_web"].ainvoke({"query": "q4"}, config=_config("other_session"))
            assert "[Web Search Results]" in r4
```

Also update the existing test at line 507-508:

```python
class TestCreateTools:

    def test_returns_8_tools(self, tools):
        assert len(tools) == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent_tools.py::TestSearchWeb -v`
Expected: FAIL — `search_web` tool does not exist

- [ ] **Step 3: Implement the `search_web` tool**

In `app/agent/tools.py`:

1. Add a module-level settings import at the top of `app/agent/tools.py` (after the existing imports, around line 11):

```python
from app.config import settings
```

2. Update `create_tools` signature (line 123-126) to accept optional `gemini_client`:

```python
def create_tools(
    retriever: HybridRetriever,
    session_store: SessionStoreBase,
    gemini_client=None,
) -> list:
    """Create the 8 agent tools, bound to the given retriever and session store.

    Each tool closes over the provided dependencies — no module-level globals.
    Multiple calls with different dependencies produce independent tool sets.
    """
    # Per-session web search counter (keyed by session_id)
    _web_search_counts: dict[str, int] = {}
```

2. Add the `search_web` tool before the `return` statement (before line 505):

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
        if not settings.WEB_SEARCH_ENABLED or gemini_client is None:
            return "Web search is not available. Answer from your own knowledge instead."

        session_id = _get_session_id(config)
        count = _web_search_counts.get(session_id, 0)
        if count >= settings.WEB_SEARCH_MAX_PER_SESSION:
            return (
                f"Web search limit reached ({settings.WEB_SEARCH_MAX_PER_SESSION} per session). "
                "Answer from your own knowledge instead."
            )

        try:
            scoped_query = f"ADHD: {query}"
            answer, sources = await gemini_client.search_web(
                query=scoped_query,
                timeout=settings.WEB_SEARCH_TIMEOUT_S,
            )
            _web_search_counts[session_id] = count + 1

            if not answer:
                return "Web search returned no results. Try a different query or answer from your own knowledge."

            parts = ["[Web Search Results]", "", answer]

            if sources:
                parts.append("")
                parts.append("Sources:")
                for i, src in enumerate(sources, 1):
                    title = src.get("title", "Untitled")
                    url = src.get("url", "")
                    parts.append(f"[{i}] {title} — {url}")

            return "\n".join(parts)
        except Exception as e:
            logger.exception("search_web failed")
            return f"Error: web search failed ({type(e).__name__}). Answer from your own knowledge instead."
```

3. Update the return list (line 505-513) to include `search_web`:

```python
    return [
        search_knowledge_base,
        get_document_details,
        get_related_documents,
        get_family_profile,
        update_family_profile,
        track_outcome,
        manage_goals,
        search_web,
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent_tools.py -v`
Expected: All tests PASS (including updated `test_returns_8_tools`)

- [ ] **Step 5: Commit**

```bash
git add app/agent/tools.py tests/test_agent_tools.py
git commit -m "Add search_web tool with per-session rate limiting"
```

---

### Task 4: Wire `gemini_client` into `create_tools()` in `app/main.py`

**Files:**
- Modify: `app/main.py:161`

- [ ] **Step 1: Update the `create_tools` call**

Change line 161 in `app/main.py` from:

```python
    tools = create_tools(retriever=retriever, session_store=session_store)
```

to:

```python
    tools = create_tools(retriever=retriever, session_store=session_store, gemini_client=gemini)
```

- [ ] **Step 2: Verify no import errors**

Run: `python -c "from app.main import build_dependencies; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "Pass gemini_client to create_tools for web search"
```

---

### Task 5: Update system prompt with `search_web` guidance

**Files:**
- Modify: `app/agent/prompts.py:62-89`

- [ ] **Step 1: Add `search_web` to the `<tools>` section**

In `app/agent/prompts.py`, after the `manage_goals` line (line 72), add:

```
- **search_web**: Search the web for current information not in the knowledge base. Use for: recent research, current events, local resources, school policy questions, anything time-sensitive. Always try search_knowledge_base first — only use search_web when the KB doesn't cover the topic or the parent needs up-to-date information.
```

- [ ] **Step 2: Add step 6 to the search workflow**

After line 82 (`5. When you have good results...`), add a new step:

```
6. If the knowledge base has no relevant results and the question is about recent information, local resources, or current events, use search_web.
```

Renumber: the existing step 5 stays, new step 6 is added. The bullet points below (Evidence framing, Be selective, etc.) remain unnumbered.

- [ ] **Step 3: Run prompt tests to verify no breakage**

Run: `pytest tests/test_prompts.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add app/agent/prompts.py
git commit -m "Add search_web guidance to system prompt and search workflow"
```

---

### Task 6: Final integration check

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v -m "not integration" --ignore=tests/test_e2e_conversations.py`
Expected: All PASS

- [ ] **Step 2: Verify tool count in all test files**

Grep for any hardcoded tool count assertions:

```bash
grep -rn "len(tools)" tests/
```

Fix any that still assert 7.

- [ ] **Step 3: Commit any remaining fixes**

```bash
git add -A
git commit -m "Fix remaining tool count assertions"
```
