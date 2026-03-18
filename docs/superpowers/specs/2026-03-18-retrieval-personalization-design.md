# Retrieval Personalization, Latency Tracking, and Summary Improvements

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Three targeted improvements to the retrieval, memory, and observability subsystems.

## Overview

Three changes shipping in this batch, one deferred:

1. **Latency tracking** -- instrument key pipeline operations with `duration_ms` on existing EventBus emit calls
2. **Personalized retrieval** -- boost/penalize retrieval results based on family outcome history using token-overlap matching
3. **Importance-weighted rolling summary** -- inject episode events into the summarization prompt so the LLM preserves key moments
4. **Episode linking by embedding** -- deferred (episode links aren't consumed anywhere yet)

## 1. Latency Tracking

### Problem

`EventBus.emit()` accepts `duration_ms` but almost no callers pass it. There is no way to identify latency regressions, set SLOs, or debug slow sessions.

### Design

Instrument these operations with `time.monotonic()` before/after and pass the delta as `duration_ms`:

| Operation | File | Current emit? |
|-----------|------|---------------|
| `HybridRetriever.retrieve()` (total) | `retriever.py` | No emit -- add one |
| `HybridRetriever._hybrid_search()` (Qdrant) | `retriever.py` | No emit -- add one |
| `FastEmbedReranker.rerank()` | `reranker.py` | No emit -- add one |
| `MemoryManager._update_summary()` | `memory.py` | Has emit, missing `duration_ms` |
| `MemoryManager._extract_facts()` | `memory.py` | Has emit, missing `duration_ms` |
| `MemoryManager._infer_emotion()` | `memory.py` | No direct emit -- add latency-only emit |

### Changes

- **`HybridRetriever.__init__`**: Add optional `event_bus` parameter.
- **`main.py`**: Wire `event_bus` into the retriever constructor.
- **`retriever.py`**: Add `t0 = time.monotonic()` / emit with `duration_ms` in `retrieve()` and `_hybrid_search()`.
- **`reranker.py`**: Add optional `event_bus` parameter, emit with `duration_ms` in `rerank()`.
- **`memory.py`**: Add `duration_ms` to existing emit calls in `_update_summary` and `_extract_facts`. Add new emit in `_infer_emotion`.

Event categories and types:

| Category | Event Type | Emitted From |
|----------|-----------|--------------|
| `rag` | `retrieve` | `retriever.py:retrieve()` |
| `rag` | `hybrid_search` | `retriever.py:_hybrid_search()` |
| `rag` | `rerank` | `reranker.py:rerank()` |
| `memory` | `summary_updated` | `memory.py:_update_summary()` (existing, add duration) |
| `memory` | `facts_extracted` | `memory.py:_extract_facts()` (existing, add duration) |
| `memory` | `emotion_inferred` | `memory.py:_infer_emotion()` (new, latency-only -- fires on every invocation including calls from `_create_episode` and `_create_goal_episode`; distinct from the existing `emotional_shift` event which only fires when an emotional shift episode is created) |

## 2. Personalized Retrieval (Outcome Boost)

### Problem

Retrieval ranks documents purely by semantic similarity + tag boost + cross-encoder score. A strategy that failed 3 times for this family ranks identically to one that worked. The outcome data exists in `SessionState.outcomes` but never flows back into retrieval ranking.

### Design

A new post-rerank step in the retrieval pipeline that applies a score boost/penalty based on the family's outcome history, using token-overlap (Jaccard similarity) matching.

#### Pipeline position

Inserted between step 3b (relevance threshold) and step 4 (facet computation) in `retriever.py`:

```
1. Query rewriting
2. Hybrid search (RRF + tag boost)
3. Rerank (cross-encoder)
3b. Relevance threshold filter
3c. Outcome boost/penalty  <-- NEW
4. Compute facets
5. Trim to top_k
```

#### Matching algorithm

Token-overlap via Jaccard similarity:

1. Tokenize each `Outcome.strategy_name` into a lowercase word set (split on whitespace, strip punctuation)
2. For each candidate `RetrievalResult`, tokenize `document_name` + all `tags` into a word set
3. Compute Jaccard similarity: `|intersection| / |union|`
4. If Jaccard >= threshold (default 0.3), read `outcome.signal` to determine boost:
   - `signal == "positive"`: apply `+0.10` boost
   - `signal == "negative"`: apply `-0.15` penalty
   - `signal == "mixed"`: no change
5. Multiple outcomes stack additively, capped at configurable max
6. Re-sort candidates by adjusted score after all boosts are applied (needed because `candidates[:top_k]` in step 5 assumes sorted order)

**Note:** The `Outcome` model field is `signal` (not `outcome`). The inline comment says `"positive" or "negative"` but `"mixed"` is also valid -- the `track_outcome` tool validates for all three values. When outcomes list is empty (new sessions), this step is a no-op.

#### Configuration

New settings in `config.py`:

```python
RAG_OUTCOME_BOOST_POSITIVE: float = 0.10
RAG_OUTCOME_BOOST_NEGATIVE: float = -0.15
RAG_OUTCOME_BOOST_CAP: float = 0.30
RAG_OUTCOME_JACCARD_THRESHOLD: float = 0.3
```

#### Interface

New method on `HybridRetriever`:

```python
def _apply_outcome_boost(
    self,
    candidates: list[RetrievalResult],
    outcomes: list[Outcome],
) -> list[RetrievalResult]:
```

Called from `retrieve()` after the relevance threshold filter, before facet computation. `state.outcomes` is passed through from the existing `state` parameter.

#### Observability

`_apply_outcome_boost` is a sync method (no async needed for token overlap). It returns a list of boost metadata dicts alongside the adjusted candidates. The caller (`retrieve()`, which is async) emits a single summary event after the method returns:

```python
# In retrieve(), after calling _apply_outcome_boost:
if boost_metadata and self._event_bus:
    await self._event_bus.emit(
        "rag", "outcome_boost", session_id="", turn=0,
        detail={"boosts": boost_metadata},  # [{strategy, document, boost}, ...]
    )
```

### Why token overlap, not embedding similarity

- Zero latency cost (no API calls at retrieval time)
- No schema changes to `Outcome`
- Covers the 80% case: partial name matches like "visual timer" vs "visual countdown timer"
- Misses synonyms ("reward chart" vs "token economy") but this is acceptable for v1
- Can upgrade to embedding similarity later by storing vectors at `track_outcome` time

## 3. Importance-Weighted Rolling Summary

### Problem

`_update_summary` in `memory.py` summarizes every N turns equally. Turns where an outcome was tracked or a goal was set should be weighted higher. The episode data exists but the summarization prompt doesn't see it.

### Design

Before building `conversation_text` in `_update_summary`, query episodes that fall within `[start_turn, current_turn]` and inject them into the summarization prompt.

#### Changes to `_update_summary`

1. After determining `start_turn` and before building `conversation_text`, call `self._store.get_episodes_with_ids(session_id)` and filter to episodes where `turn_range_start` falls within `[start_turn, current_turn]`.
2. Format matching episodes as a "Key events" block:

```
Key events this window:
- [outcome_reported] Parent reported positive outcome for 'visual timer': worked great for homework
- [goal_set] Parent set new goal: 'consistent bedtime routine'
- [emotional_shift] Parent expressed overwhelmed emotional state
```

3. Inject this block into the summarization prompt before the conversation transcript, with the instruction: "Pay special attention to these key events -- they represent important moments that should be preserved in the summary."

#### What doesn't change

- No schema changes
- No new config values
- No changes to how episodes are created or stored
- The LLM does the importance weighting -- we just ensure it sees structured events alongside raw conversation

## 4. Episode Linking by Embedding (Deferred)

Episode linking at `memory.py:_link_episode` uses exact string match on `strategies_involved`. This misses partial matches ("visual timer" vs "visual countdown timer").

**Deferred because:**
- Episode links are write-only -- nothing in the retrieval or prompt pipeline consumes them
- Adding embedding similarity requires embedding episodes at creation time (schema change + migration)
- Low user-facing impact

**Revisit when:** Episode links are surfaced in the UI, used in retrieval ranking, or included in the system prompt.

## Files Changed

| File | Changes |
|------|---------|
| `app/config.py` | Add 4 outcome boost settings |
| `app/rag/retriever.py` | Add `event_bus` param, `_apply_outcome_boost()`, latency instrumentation, outcome boost step in `retrieve()` |
| `app/rag/reranker.py` | Add `event_bus` param, latency instrumentation |
| `app/agent/memory.py` | Add `duration_ms` to existing emits, new emotion emit, episode injection in `_update_summary` |
| `app/main.py` | Wire `event_bus` into retriever and reranker constructors |

## Testing

- Unit test for `_apply_outcome_boost`: verify boost/penalty applied correctly, cap respected, Jaccard threshold filtering works, re-sort after boost
- Unit test for token overlap edge cases: empty outcomes, no matching documents, mixed signal (no-op), new session (empty outcomes list)
- Unit test for importance-weighted summary: verify episodes are injected into the prompt
- Unit test for latency instrumentation: mock EventBus, call `retrieve()` / `rerank()` / `_update_summary()`, assert `emit` was called with `duration_ms > 0`
- Existing tests should continue passing unchanged (outcome boost is additive, latency tracking is observability-only)

Test files to create or modify: `tests/test_rag.py` (outcome boost + latency), `tests/test_memory.py` (summary episodes + latency)
