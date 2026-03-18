# Retrieval Personalization, Latency Tracking, and Summary Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add personalized retrieval ranking from outcome history, latency instrumentation across the pipeline, and importance-weighted rolling summaries.

**Architecture:** Three independent changes: (1) a post-rerank outcome boost step in the retriever using Jaccard token-overlap matching, (2) `duration_ms` instrumentation on EventBus emit calls in retriever/reranker/memory, (3) episode injection into the summarization prompt. All changes are additive — no existing behavior is modified.

**Tech Stack:** Python, Pydantic, pytest, asyncio, EventBus (existing)

**Spec:** `docs/superpowers/specs/2026-03-18-retrieval-personalization-design.md`

**Testing policy:** Never run integration tests or E2E tests requiring a real API key. Always ask the user before running any tests.

**Python environment:** Always use the `adhd312` venv. Run tests with `./adhd312/Scripts/python.exe -m pytest`. Install packages with `./adhd312/Scripts/pip.exe install`.

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `app/config.py` | Modify | Add 4 outcome boost settings |
| `app/rag/retriever.py` | Modify | Add `event_bus` param, `_apply_outcome_boost()`, latency emits, outcome boost step in `retrieve()` |
| `app/rag/reranker.py` | Modify | Add `event_bus` param, latency emit in `rerank()` |
| `app/agent/memory.py` | Modify | Add `duration_ms` to existing emits, new emotion emit, episode injection in `_update_summary` |
| `app/main.py` | Modify | Wire `event_bus` into retriever and reranker constructors |
| `tests/test_rag.py` | Modify | Add outcome boost tests, latency emit tests |
| `tests/test_memory.py` | Modify | Add importance-weighted summary test, latency emit tests |

---

## Task 1: Add outcome boost config settings

**Files:**
- Modify: `app/config.py:17-26` (RAG settings block)

- [ ] **Step 1: Add settings to config.py**

In `app/config.py`, add these 4 settings after `RAG_QUERY_CACHE_SIMILARITY` (line 25):

```python
RAG_OUTCOME_BOOST_POSITIVE: float = 0.10
RAG_OUTCOME_BOOST_NEGATIVE: float = -0.15
RAG_OUTCOME_BOOST_CAP: float = 0.30
RAG_OUTCOME_JACCARD_THRESHOLD: float = 0.3
```

- [ ] **Step 2: Verify import works**

Run: `./adhd312/Scripts/python.exe -c "from app.config import settings; print(settings.RAG_OUTCOME_BOOST_POSITIVE, settings.RAG_OUTCOME_BOOST_NEGATIVE, settings.RAG_OUTCOME_BOOST_CAP, settings.RAG_OUTCOME_JACCARD_THRESHOLD)"`
Expected: `0.1 -0.15 0.3 0.3`

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "Add outcome boost config settings for personalized retrieval"
```

---

## Task 2: Implement `_apply_outcome_boost` with tests (TDD)

**Files:**
- Modify: `app/rag/retriever.py`
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Write failing tests for outcome boost**

Add to `tests/test_rag.py`:

```python
from app.models.schemas import Outcome


def _make_result(name: str, score: float, tags: list[str] | None = None) -> RetrievalResult:
    """Helper to create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        document_id=name.lower().replace(" ", "_"),
        document_name=name,
        content=f"Content for {name}",
        score=score,
        tags=tags or [],
    )


class TestOutcomeBoost:

    @pytest.fixture
    def retriever_for_boost(self, knowledge_store):
        return HybridRetriever(knowledge_store=knowledge_store, gemini_client=None)

    def test_positive_outcome_boosts_matching_doc(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer", "homework"]),
            _make_result("Reward Chart System", 0.85, ["rewards", "motivation"]),
        ]
        outcomes = [Outcome(strategy_name="visual timer", signal="positive", turn=1)]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        timer_doc = next(r for r in boosted if "Timer" in r.document_name)
        reward_doc = next(r for r in boosted if "Reward" in r.document_name)
        assert timer_doc.score == pytest.approx(0.90, abs=0.01)  # 0.80 + 0.10
        assert reward_doc.score == pytest.approx(0.85, abs=0.01)  # unchanged

    def test_negative_outcome_penalizes_matching_doc(self, retriever_for_boost):
        candidates = [
            _make_result("Reward Chart System", 0.85, ["rewards", "motivation"]),
        ]
        outcomes = [Outcome(strategy_name="reward chart", signal="negative", turn=1)]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.70, abs=0.01)  # 0.85 - 0.15

    def test_mixed_outcome_no_change(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [Outcome(strategy_name="visual timer", signal="mixed", turn=1)]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.80, abs=0.01)

    def test_boost_capped_at_max(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        # 5 positive outcomes would be 5 * 0.10 = 0.50 uncapped, but cap is 0.30
        outcomes = [
            Outcome(strategy_name="visual timer", signal="positive", turn=i)
            for i in range(5)
        ]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(1.10, abs=0.01)  # 0.80 + 0.30 (capped)

    def test_penalty_capped_at_negative_max(self, retriever_for_boost):
        candidates = [
            _make_result("Reward Chart System", 0.85, ["rewards"]),
        ]
        outcomes = [
            Outcome(strategy_name="reward chart", signal="negative", turn=i)
            for i in range(5)
        ]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.55, abs=0.01)  # 0.85 - 0.30 (capped)

    def test_empty_outcomes_is_noop(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
            _make_result("Reward Chart System", 0.85, ["rewards"]),
        ]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, [])

        assert boosted[0].score == pytest.approx(0.85, abs=0.01)
        assert boosted[1].score == pytest.approx(0.80, abs=0.01)

    def test_no_matching_outcome_is_noop(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [Outcome(strategy_name="completely unrelated strategy", signal="positive", turn=1)]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert boosted[0].score == pytest.approx(0.80, abs=0.01)

    def test_results_resorted_after_boost(self, retriever_for_boost):
        candidates = [
            _make_result("Reward Chart System", 0.90, ["rewards"]),
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [
            Outcome(strategy_name="reward chart", signal="negative", turn=1),
            Outcome(strategy_name="visual timer", signal="positive", turn=2),
        ]

        boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        # Timer: 0.80 + 0.10 = 0.90, Reward: 0.90 - 0.15 = 0.75
        # Timer should now be first
        assert "Timer" in boosted[0].document_name
        assert "Reward" in boosted[1].document_name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestOutcomeBoost -v`
Expected: FAIL with `AttributeError: 'HybridRetriever' object has no attribute '_apply_outcome_boost'`

- [ ] **Step 3: Implement `_apply_outcome_boost` in retriever.py**

Add this import at the top of `app/rag/retriever.py` (after line 8):

```python
import re
```

Add this import to the existing imports block (after line 23):

```python
from app.models.schemas import Outcome
```

Add this method to `HybridRetriever` class (after `_build_results`, around line 207):

```python
def _apply_outcome_boost(
    self,
    candidates: list[RetrievalResult],
    outcomes: list[Outcome],
) -> list[RetrievalResult]:
    """Apply score boost/penalty based on family outcome history.

    Uses Jaccard token-overlap between outcome strategy names and
    document name + tags. Returns re-sorted candidates with boost metadata.
    """
    if not outcomes or not candidates:
        return candidates

    threshold = settings.RAG_OUTCOME_JACCARD_THRESHOLD
    boost_pos = settings.RAG_OUTCOME_BOOST_POSITIVE
    boost_neg = settings.RAG_OUTCOME_BOOST_NEGATIVE
    cap = settings.RAG_OUTCOME_BOOST_CAP

    # Pre-tokenize all outcome strategy names
    outcome_tokens = []
    for o in outcomes:
        tokens = set(re.sub(r'[^\w\s]', '', o.strategy_name.lower()).split())
        outcome_tokens.append((o, tokens))

    for result in candidates:
        # Tokenize document name + tags
        doc_text = result.document_name.lower()
        for tag in result.tags:
            doc_text += " " + tag.lower().replace("_", " ")
        doc_tokens = set(re.sub(r'[^\w\s]', '', doc_text).split())

        if not doc_tokens:
            continue

        total_boost = 0.0
        for outcome, o_tokens in outcome_tokens:
            if not o_tokens:
                continue
            intersection = o_tokens & doc_tokens
            union = o_tokens | doc_tokens
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard >= threshold:
                if outcome.signal == "positive":
                    total_boost += boost_pos
                elif outcome.signal == "negative":
                    total_boost += boost_neg
                # "mixed" -> no change

        # Cap the total boost
        total_boost = max(-cap, min(cap, total_boost))
        if total_boost != 0.0:
            result.score += total_boost

    # Re-sort by adjusted score
    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestOutcomeBoost -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/retriever.py tests/test_rag.py
git commit -m "Add outcome boost to retrieval pipeline with Jaccard token-overlap matching"
```

---

## Task 3: Wire outcome boost into `retrieve()` pipeline

**Files:**
- Modify: `app/rag/retriever.py:64-129` (the `retrieve` method)
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Write integration test first**

Add to `tests/test_rag.py`:

```python
from app.models.schemas import SessionState, FamilyProfile


class TestOutcomeBoostInPipeline:

    @pytest.mark.asyncio
    async def test_retrieve_applies_outcome_boost(self, knowledge_store):
        """Outcome boost integrates into the full retrieve() pipeline."""
        retriever = HybridRetriever(knowledge_store=knowledge_store, gemini_client=None)

        # First retrieve without outcomes to get a baseline
        baseline = await retriever.retrieve("homework timer strategies")
        assert len(baseline.results) > 0, "Need keyword results for this test"

        # Now retrieve with a negative outcome for the top result
        top_doc_name = baseline.results[0].document_name
        top_original_score = baseline.results[0].score
        state = SessionState(
            session_id="test",
            outcomes=[Outcome(strategy_name=top_doc_name, signal="negative", turn=1)],
        )
        boosted = await retriever.retrieve("homework timer strategies", state=state)

        # The penalized doc should have a lower score
        penalized = next((r for r in boosted.results if r.document_name == top_doc_name), None)
        assert penalized is not None
        assert penalized.score < top_original_score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestOutcomeBoostInPipeline -v`
Expected: FAIL (outcome boost not wired into retrieve() yet)

- [ ] **Step 3: Add outcome boost step to `retrieve()`**

In `app/rag/retriever.py`, in the `retrieve()` method, add step 3c after the relevance threshold filter (after line 111, before `# Step 4: Compute facets`):

```python
        # Step 3c: Outcome boost/penalty (personalization from family history)
        if state and state.outcomes:
            candidates = self._apply_outcome_boost(candidates, state.outcomes)
```

- [ ] **Step 4: Run the test**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestOutcomeBoostInPipeline -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/rag/retriever.py tests/test_rag.py
git commit -m "Wire outcome boost into retrieve() pipeline after reranking"
```

---

## Task 4: Add latency tracking to retriever and reranker

**Files:**
- Modify: `app/rag/retriever.py` (add `event_bus` param, latency emits)
- Modify: `app/rag/reranker.py` (add `event_bus` param, latency emit)
- Modify: `app/main.py` (wire `event_bus`)
- Modify: `tests/test_rag.py` (latency emit tests)

- [ ] **Step 1: Write failing test for retriever latency emit**

Add to `tests/test_rag.py`:

```python
from unittest.mock import AsyncMock


class TestLatencyTracking:

    @pytest.mark.asyncio
    async def test_retrieve_emits_latency(self, knowledge_store):
        event_bus = AsyncMock()
        retriever = HybridRetriever(
            knowledge_store=knowledge_store,
            gemini_client=None,
            event_bus=event_bus,
        )

        await retriever.retrieve("homework strategies")

        # EventBus.emit signature: emit(category, event_type, ..., duration_ms=0.0, ...)
        # Our call: emit("rag", "retrieve", duration_ms=..., detail=...)
        calls = [c for c in event_bus.emit.call_args_list if len(c.args) > 1 and c.args[1] == "retrieve"]
        assert len(calls) >= 1
        assert calls[0].kwargs.get("duration_ms", 0) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestLatencyTracking::test_retrieve_emits_latency -v`
Expected: FAIL (HybridRetriever doesn't accept `event_bus` param yet)

- [ ] **Step 3: Add `event_bus` to HybridRetriever.__init__**

In `app/rag/retriever.py`, modify `__init__` to accept `event_bus`:

```python
def __init__(
    self,
    knowledge_store: KnowledgeStore,
    gemini_client=None,
    query_rewriter: QueryRewriter | None = None,
    reranker: FastEmbedReranker | None = None,
    colbert_index: ColBERTIndex | None = None,
    event_bus=None,
):
    self._store = knowledge_store
    self._gemini = gemini_client
    self._rewriter = query_rewriter
    self._reranker = reranker
    self._colbert = colbert_index
    self._event_bus = event_bus
    self._query_cache: list[tuple[list[float], list[RetrievalResult], float]] = []
```

- [ ] **Step 4: Add latency emit to `retrieve()`**

At the start of `retrieve()`, add `t0 = time.monotonic()`. At the end, before the return, add:

```python
        duration_ms = (time.monotonic() - t0) * 1000
        if self._event_bus:
            await self._event_bus.emit(
                "rag", "retrieve", duration_ms=duration_ms,
                detail={"result_count": len(results), "rewritten": rewritten_query is not None},
            )
```

Also add latency emit inside `_hybrid_search()`. Add `t0 = time.monotonic()` at the start of the method. Emit before the two search-path returns only (NOT the cache-hit return at line 162, which should not be counted as a search):

Before `return results` (hybrid search success, line 180):
```python
                if self._event_bus:
                    await self._event_bus.emit(
                        "rag", "hybrid_search", duration_ms=(time.monotonic() - t0) * 1000,
                        detail={"method": "hybrid", "result_count": len(results)},
                    )
```

Before `return self._keyword_fallback(...)` (keyword fallback, line 183):
```python
        fallback_results = self._keyword_fallback(query, top_k)
        if self._event_bus:
            await self._event_bus.emit(
                "rag", "hybrid_search", duration_ms=(time.monotonic() - t0) * 1000,
                detail={"method": "keyword", "result_count": len(fallback_results)},
            )
        return fallback_results
```

- [ ] **Step 5: Add `event_bus` to FastEmbedReranker**

In `app/rag/reranker.py`, add `import time` at the top (after `import math`). Modify `__init__`:

```python
def __init__(self, model_name: str = "BAAI/bge-reranker-base", event_bus=None):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    self._model = TextCrossEncoder(model_name=model_name)
    self._event_bus = event_bus
    logger.info(f"FastEmbed reranker loaded (model={model_name})")
```

In `rerank()`, add `t0 = time.monotonic()` after the early return for empty results. Add latency emit at end of `rerank()`, before the return:

```python
        if self._event_bus:
            await self._event_bus.emit(
                "rag", "rerank", duration_ms=(time.monotonic() - t0) * 1000,
                detail={"input_count": len(results), "output_count": len(reranked)},
            )
```

- [ ] **Step 6: Wire event_bus in main.py**

In `app/main.py`, the `event_bus` is created at line 133, but the retriever is created at line 121 (before the event_bus). Move the event_bus creation to before the retriever, or pass it after construction.

Simplest: move the EventBus creation (lines 132-134) to after `db_conn` initialization (line 106) but before the reranker/retriever construction (line 115). The EventBus needs `conn=db_conn`, so it must stay after the `db_conn` block. Then pass `event_bus=event_bus` to both the `HybridRetriever` and `FastEmbedReranker` constructors:

```python
# Reranker (line ~118):
reranker = FastEmbedReranker(model_name=settings.RAG_RERANK_MODEL, event_bus=event_bus)

# Retriever (line ~121):
retriever = HybridRetriever(
    knowledge_store=store,
    gemini_client=gemini,
    query_rewriter=query_rewriter,
    reranker=reranker,
    colbert_index=colbert,
    event_bus=event_bus,
)
```

- [ ] **Step 7: Run tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestLatencyTracking -v`
Expected: PASS

- [ ] **Step 8: Run all existing RAG tests to verify no regressions**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py -v`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add app/rag/retriever.py app/rag/reranker.py app/main.py tests/test_rag.py
git commit -m "Add latency tracking to retriever and reranker via EventBus"
```

---

## Task 5: Add latency tracking to memory operations

**Files:**
- Modify: `app/agent/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write failing test for summary latency emit**

Add to `tests/test_memory.py`:

```python
class TestLatencyTracking:

    @pytest.mark.asyncio
    async def test_summary_emits_duration_ms(self):
        store = await _make_store_with_messages(turn_count=5)
        gemini = _make_gemini_mock(generate_return="Summary of conversation.")
        event_bus = AsyncMock()
        mm = MemoryManager(session_store=store, gemini_client=gemini, event_bus=event_bus)

        await mm.post_turn_tasks(
            session_id="s1", turn=5,
            user_message="This is turn five with enough content for processing",
            assistant_response="Got it.",
        )

        # Find the summary_updated emit call
        summary_calls = [
            c for c in event_bus.emit.call_args_list
            if len(c.args) >= 2 and c.args[1] == "summary_updated"
        ]
        assert len(summary_calls) >= 1
        call = summary_calls[0]
        # duration_ms should be passed and positive
        duration = call.kwargs.get("duration_ms", 0)
        assert duration > 0

    @pytest.mark.asyncio
    async def test_emotion_inference_emits_duration_ms(self):
        store = await _make_store_with_messages(turn_count=1)
        gemini = _make_gemini_mock(generate_return="frustrated")
        event_bus = AsyncMock()
        mm = MemoryManager(session_store=store, gemini_client=gemini, event_bus=event_bus)

        await mm.post_turn_tasks(
            session_id="s1", turn=1,
            user_message="I am so frustrated with my child's homework avoidance and nothing is working",
            assistant_response="I understand.",
        )

        # Find the emotion_inferred emit call
        emotion_calls = [
            c for c in event_bus.emit.call_args_list
            if len(c.args) >= 2 and c.args[1] == "emotion_inferred"
        ]
        assert len(emotion_calls) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py::TestLatencyTracking -v`
Expected: FAIL (no `duration_ms` in emit calls, no `emotion_inferred` event)

- [ ] **Step 3: Add `duration_ms` to `_update_summary`**

In `app/agent/memory.py`, add `import time` at the top. In `_update_summary`, add `t0 = time.monotonic()` at the start (after the `if not self._gemini` guard). Update the existing emit call (line 145) to include `duration_ms`:

```python
            if self._event_bus:
                await self._event_bus.emit("memory", "summary_updated", session_id, current_turn,
                                           duration_ms=(time.monotonic() - t0) * 1000)
```

- [ ] **Step 4: Add `duration_ms` to `_extract_facts`**

Add `t0 = time.monotonic()` at the start of `_extract_facts` (after the `if not self._gemini` guard). Update the existing emit call (line 247) to include `duration_ms`:

```python
                    if self._event_bus:
                        await self._event_bus.emit("memory", "facts_extracted", session_id, turn,
                                                   duration_ms=(time.monotonic() - t0) * 1000,
                                                   detail={"fields": list(filtered.keys())})
```

- [ ] **Step 5: Add latency emit to `_infer_emotion`**

Add `t0 = time.monotonic()` at the start of `_infer_emotion` (after the `if not self._gemini` guard). Add a single emit after the LLM call completes (after line 480, `emotion = result.strip().lower()`), before any return branching:

```python
            emotion = result.strip().lower()
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", "emotion_inferred", session_id, 0,
                    duration_ms=(time.monotonic() - t0) * 1000,
                    detail={"emotion": emotion},
                )
            if emotion not in valid_emotions or emotion == "neutral":
                return ""
            return emotion
```

This replaces the existing lines 481-483. The emit fires once for every invocation (including neutral results and calls from `_create_episode` and `_create_goal_episode`), giving consistent latency data.

- [ ] **Step 6: Run tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py::TestLatencyTracking -v`
Expected: PASS

- [ ] **Step 7: Run all memory tests for regressions**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add app/agent/memory.py tests/test_memory.py
git commit -m "Add latency tracking to memory operations via EventBus duration_ms"
```

---

## Task 6: Importance-weighted rolling summary

**Files:**
- Modify: `app/agent/memory.py:81-148` (`_update_summary`)
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write failing test for episode injection in summary**

Add to `tests/test_memory.py`:

```python
class TestImportanceWeightedSummary:

    @pytest.mark.asyncio
    async def test_summary_prompt_includes_episodes(self):
        """Episodes from the current window are injected into the summary prompt."""
        store = await _make_store_with_messages(turn_count=5)

        # Add an episode in the turn window
        episode = EpisodicMemory(
            event_type="outcome_reported",
            summary="Parent reported positive outcome for 'visual timer'",
            outcome="positive",
            strategies_involved=["visual timer"],
            emotional_context="hopeful",
            turn_range_start=3,
            turn_range_end=3,
        )
        await store.add_episode("s1", episode)

        gemini = _make_gemini_mock(generate_return="Summary with key events.")
        mm = MemoryManager(session_store=store, gemini_client=gemini)

        await mm._update_summary("s1", 5)

        # The generate call should include the episode in the prompt
        call_args = gemini.generate.call_args
        prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "Key events this window" in prompt
        assert "outcome_reported" in prompt
        assert "visual timer" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py::TestImportanceWeightedSummary -v`
Expected: FAIL (prompt doesn't contain "Key events this window")

- [ ] **Step 3: Implement episode injection in `_update_summary`**

In `app/agent/memory.py`, in `_update_summary`, after determining `start_turn` (line 88) and before building `conversation_text` (line 90), add:

```python
        # Load episodes from this turn window for importance weighting
        all_episodes = await self._store.get_episodes_with_ids(session_id, limit=50)
        window_episodes = [
            ep for _id, ep in all_episodes
            if ep.turn_range_start >= start_turn and ep.turn_range_start <= current_turn
        ]
        key_events_block = ""
        if window_episodes:
            event_lines = []
            for ep in window_episodes:
                line = f"- [{ep.event_type}] {ep.summary}"
                if ep.emotional_context:
                    line += f" (mood: {ep.emotional_context})"
                event_lines.append(line)
            key_events_block = "\nKey events this window:\n" + "\n".join(event_lines)
```

Then inject `key_events_block` into the prompt. Add it before the conversation transcript. Modify the prompt string to include it before the `New conversation to incorporate:` section:

```python
        prompt = f"""Summarize the emotional and narrative arc of this ADHD coaching conversation.

Focus ONLY on what is NOT already captured in the structured family profile:
- How the parent is feeling and what's weighing on them
- Specific concerns, quotes, or worries they've expressed
- How the conversation has evolved (what was tried, how they reacted)
- Relational context (frustration level, trust built, resistance encountered)

{profile_note}

Omit: demographic facts, strategy names, diagnosis details, and anything already in the previous summary.

{f"Previous summary: {prior_summary}" if prior_summary else ""}
{key_events_block}
{f"Pay special attention to the key events above -- they represent important moments that should be preserved in the summary." if key_events_block else ""}
New conversation to incorporate:
{conversation_text}

Write a concise summary (2-4 sentences) focused on narrative and emotional context only."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py::TestImportanceWeightedSummary -v`
Expected: PASS

- [ ] **Step 5: Run all memory tests for regressions**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_memory.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/agent/memory.py tests/test_memory.py
git commit -m "Add importance-weighted rolling summary with episode injection"
```

---

## Task 7: Outcome boost observability emit

**Files:**
- Modify: `app/rag/retriever.py` (emit boost metadata from `retrieve()`)
- Modify: `tests/test_rag.py`

- [ ] **Step 1: Update tests FIRST for new return type (TDD)**

In `tests/test_rag.py`, update all `TestOutcomeBoost` tests to expect a tuple return. Change lines like:

```python
boosted = retriever_for_boost._apply_outcome_boost(candidates, outcomes)
```

to:

```python
boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)
```

Add one test verifying metadata:

```python
    def test_boost_returns_metadata(self, retriever_for_boost):
        candidates = [
            _make_result("Visual Timer Strategy", 0.80, ["timer"]),
        ]
        outcomes = [Outcome(strategy_name="visual timer", signal="positive", turn=1)]

        boosted, metadata = retriever_for_boost._apply_outcome_boost(candidates, outcomes)

        assert len(metadata) == 1
        assert metadata[0]["strategy"] == "visual timer"
        assert metadata[0]["document"] == "Visual Timer Strategy"
        assert metadata[0]["boost"] == pytest.approx(0.10, abs=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py::TestOutcomeBoost -v`
Expected: FAIL (cannot unpack non-tuple return)

- [ ] **Step 3: Update `_apply_outcome_boost` to return tuple with metadata**

Change the method signature to return a tuple:

```python
def _apply_outcome_boost(
    self,
    candidates: list[RetrievalResult],
    outcomes: list[Outcome],
) -> tuple[list[RetrievalResult], list[dict]]:
```

Add a `boost_metadata` list at the start of the method. When a non-zero `total_boost` is applied to a result, append `{"strategy": outcome.strategy_name, "document": result.document_name, "boost": total_boost}` for each contributing outcome. Return `(candidates, boost_metadata)`. For the early return on empty inputs, return `(candidates, [])`.

- [ ] **Step 4: Update `retrieve()` to emit the boost event**

In `retrieve()`, update the outcome boost step:

```python
        # Step 3c: Outcome boost/penalty (personalization from family history)
        if state and state.outcomes:
            candidates, boost_metadata = self._apply_outcome_boost(candidates, state.outcomes)
            if boost_metadata and self._event_bus:
                await self._event_bus.emit(
                    "rag", "outcome_boost",
                    detail={"boosts": boost_metadata},
                )
```

- [ ] **Step 4: Run all RAG tests**

Run: `./adhd312/Scripts/python.exe -m pytest tests/test_rag.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `./adhd312/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add app/rag/retriever.py tests/test_rag.py
git commit -m "Add outcome boost observability emit with boost metadata"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run full test suite**

Run: `./adhd312/Scripts/python.exe -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify app imports cleanly**

Run: `./adhd312/Scripts/python.exe -c "from app.main import app; print('OK')"`
Expected: `OK` (no import errors)

- [ ] **Step 3: Commit any remaining changes**

Only if there are unstaged changes from fixes during verification.
