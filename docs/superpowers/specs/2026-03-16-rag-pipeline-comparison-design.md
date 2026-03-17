# RAG Pipeline Comparison — Design Spec

**Date:** 2026-03-16
**Scope:** Add configurable pipeline variants and an ablation eval runner to compare retrieval approaches.

---

## Goal

Compare 6 retrieval pipeline configurations against the existing golden dataset to identify which components (BM25 sparse encoding, cross-encoder reranker, ColBERT late interaction, query rewriting) actually improve retrieval metrics. One run produces a side-by-side comparison table.

---

## Prerequisites

- Qdrant server >= 1.10 and `qdrant-client` SDK >= 1.10 (required for multi-prefetch RRF fusion)
- `RAG_USE_QUERY_REWRITE=True` in environment (or the comparison runner overrides it per-config)
- FastEmbed `colbert-ir/colbertv2.0` ONNX model (~500 MB, downloaded on first use by FastEmbed)

---

## Components

### 1. `eval/pipeline_config.py` — new file

A dataclass defining the axes of variation and the 6 ablation configs.

```python
@dataclass
class PipelineConfig:
    name: str
    sparse_mode: Literal["tfidf", "bm25"]
    use_reranker: bool
    use_colbert: bool
    use_query_rewriter: bool
```

Ablation configs (each varies exactly one axis from `baseline`; `full` combines all):

| name | sparse | reranker | colbert | rewriter |
|---|---|---|---|---|
| `baseline` | tfidf | off | off | off |
| `+rewriter` | tfidf | off | off | on |
| `+reranker` | tfidf | on | off | off |
| `+bm25` | bm25 | off | off | off |
| `+colbert` | tfidf | off | on | off |
| `full` | bm25 | on | on | on |

Note: `full` combines all axes simultaneously and is not one axis away from any single other config. It serves as a "best possible" reference point, not a clean ablation step.

---

### 2. BM25 sparse encoding — modify `app/rag/knowledge_store.py`

**Constructor changes:**
- Add `sparse_mode: Literal["tfidf", "bm25"] = "tfidf"` parameter (default preserves current behavior)
- Add `collection_name: str | None = None` parameter — overrides `settings.QDRANT_COLLECTION` when provided, so multiple instances can coexist in the same Qdrant process without clobbering each other's collections
- Add `self._avgdl: float = 0.0` initialization before `_load_documents()` is called

**`_build_vocabulary()` additions:** Also computes `self._avgdl` (average document length in tokens across all chunks). If there are no chunks, `_avgdl` stays `0.0` and `_text_to_sparse_doc` must guard against division by zero.

**Sparse encoding split:** The existing `_text_to_sparse` method is replaced by two methods:

- `_text_to_sparse_doc(text: str) -> SparseVector` — used at **index time** in `build_index`. For BM25 mode, computes saturated TF with `k1=1.5, b=0.75`:
  ```
  tf_bm25 = count * (k1 + 1) / (count + k1 * (1 - b + b * doc_len / avgdl))
  ```
  For TF-IDF mode, returns raw `float(count)` (current behavior).

- `_text_to_sparse_query(text: str) -> SparseVector` — used at **query time** in `search_hybrid`. Returns raw count (1 per unique term) in both modes. BM25 saturation on a query string applies `avgdl` against the wrong length and has near-identity effect when each term appears once; using raw counts is more correct.

Qdrant applies IDF server-side via `Modifier.IDF` in both modes. No collection schema change — the sparse vector indices are identical, only values differ.

Since sparse vector values are baked into the Qdrant collection at index time, changing `sparse_mode` on an already-indexed instance has no effect on stored vectors. **A separate `KnowledgeStore` instance with a distinct `collection_name` is required for each distinct `sparse_mode`.**

**`build_index` change:** Accepts an optional `colbert_index: ColBERTIndex | None = None` parameter. When provided, `build_index` creates the collection with all three named vectors in a single `recreate_collection` call, and uploads Gemini dense embeddings, BM25/TF-IDF sparse vectors, and ColBERT token embeddings together in one `upsert` pass. This ensures the collection is always fully populated after a single `build_index` call. `ColBERTIndex` is never responsible for calling `recreate_collection` on its own.

**`_collection_is_current` change:** The idempotency check must also verify whether the `"colbert"` multi-vector config is present when ColBERT is enabled. A collection built without ColBERT that otherwise passes (same point count, dense and sparse present) must be treated as stale and rebuilt when `colbert_index` is provided. Concretely: if `colbert_index is not None` and `info.config.params.multi_vectors_config` is absent or does not contain `"colbert"`, return `False` to force a rebuild. (ColBERT lives in `multi_vectors_config`, not `sparse_vectors_config`.)

**`search_hybrid` change:** Accepts an optional `colbert_prefetch: Prefetch | None = None` parameter. When present, it is added alongside the existing dense and sparse `Prefetch` objects before `FusionQuery(fusion=Fusion.RRF)`.

---

### 3. `app/rag/colbert_index.py` — new file

Provides ColBERT token embeddings. Does **not** manage the Qdrant collection — collection lifecycle is handled entirely by `KnowledgeStore.build_index`.

**Model:** `colbert-ir/colbertv2.0` via FastEmbed `LateInteractionTextEmbedding`. Same `fastembed` package as the reranker, no new top-level dependency. First use triggers a ~500 MB model download.

**`ColBERTIndex.embed_chunks(chunks: list[dict]) -> list[list[list[float]]]`** — generates per-token embeddings for all chunks. Returns a list of token embedding matrices (one matrix per chunk, shape `[n_tokens, 128]`). `LateInteractionTextEmbedding` is synchronous ONNX inference; `build_index` wraps this call in `asyncio.to_thread(...)` to avoid blocking the event loop (same pattern as `FastEmbedReranker.rerank`). The returned matrices are stored directly in `PointStruct` as `vector={"colbert": token_matrix}` where `token_matrix: list[list[float]]` — a list of 128-dim vectors, one per token. Do not flatten.

**`ColBERTIndex.make_prefetch(query_text: str, limit: int) -> Prefetch`** — generates per-token query embeddings and returns a `Prefetch` for the `"colbert"` named vector. Called by `HybridRetriever._hybrid_search` when ColBERT is active. The `limit` argument is the same `prefetch_limit` already computed in `_hybrid_search` (`min(top_k * 3, len(self._store.chunks))`), keeping the ColBERT candidate pool aligned with dense and sparse prefetches.

**Qdrant collection schema (when ColBERT enabled):** `build_index` adds:
```python
multi_vectors_config={"colbert": MultiVectorParams(
    size=128,
    distance=Distance.COSINE,
    multivec_comparator=MultiVectorComparator.MAX_SIM,
)}
```
alongside the existing dense and sparse vector configs. Qdrant computes MaxSim natively at query time.

---

### 4. Modify `app/rag/retriever.py`

`HybridRetriever.__init__` gains an optional `colbert_index: ColBERTIndex | None = None` parameter stored as `self._colbert`.

`_hybrid_search` is updated: when `self._colbert` is not None, it calls `self._colbert.make_prefetch(search_query, prefetch_limit)` and passes the result as `colbert_prefetch` to `self._store.search_hybrid(...)`.

No other changes to `retriever.py`.

---

### 5. `eval/runners/comparison_runner.py` — new file

Runs all 6 ablation configs in sequence and produces a side-by-side comparison.

**Index strategy:**

Because sparse vector values and collection schema are baked in at index time, configs with different `sparse_mode` or `use_colbert` values require distinct collections. The runner builds up to 4 `KnowledgeStore` instances once upfront, each with a distinct `collection_name`:

| collection_name | sparse_mode | colbert | configs that use it |
|---|---|---|---|
| `rag_tfidf` | tfidf | off | baseline, +rewriter, +reranker |
| `rag_bm25` | bm25 | off | +bm25 |
| `rag_tfidf_colbert` | tfidf | on | +colbert |
| `rag_bm25_colbert` | bm25 | on | full |

This avoids re-embedding chunks per config.

**Query rewriter activation:**

`HybridRetriever.retrieve` checks both the `_rewriter` instance and `settings.RAG_USE_QUERY_REWRITE`. The comparison runner sets `settings.RAG_USE_QUERY_REWRITE = True` at startup. For configs where `use_query_rewriter=False`, the runner passes `query_rewriter=None` to `HybridRetriever`. For rewriter-enabled configs, it passes a `QueryRewriter` instance and passes a minimal `SessionState` stub with a non-empty `conversation_history` to `retriever.retrieve(question, state=stub_state)` so all four conditions in the rewrite guard are satisfied.

**CLI:**
```
python -m eval.runners.comparison_runner [--k 1 3 5] [--types fact_single reasoning]
```

**Output format:**
```
Pipeline       | MRR   | R@1   | R@3   | R@5   | NDCG@5
---------------+-------+-------+-------+-------+-------
baseline       | 0.xxx | 0.xxx | 0.xxx | 0.xxx | 0.xxx
+rewriter      | 0.xxx | ...
+reranker      | 0.xxx | ...
+bm25          | 0.xxx | ...
+colbert       | 0.xxx | ...
full           | 0.xxx | ...
```

Results JSON includes per-question-type breakdown for each config. Saved to `eval/data/results/comparison_<timestamp>.json`.

---

## Files Changed

| File | Change |
|---|---|
| `eval/pipeline_config.py` | New — PipelineConfig dataclass + 6 ablation configs |
| `app/rag/colbert_index.py` | New — ColBERT token embedding generation + prefetch factory |
| `app/rag/knowledge_store.py` | Add `sparse_mode`, `collection_name` params; `_avgdl` init; split `_text_to_sparse`; `build_index` accepts optional `ColBERTIndex`; `search_hybrid` accepts optional `colbert_prefetch` |
| `app/rag/retriever.py` | Add optional `colbert_index` param; pass ColBERT prefetch in `_hybrid_search` |
| `eval/runners/comparison_runner.py` | New — ablation comparison runner |

`retrieval_runner.py`, `reranker.py`, and all metrics files are unchanged.

---

## Out of Scope

- Tag boosting weight variation (separate experiment)
- Candidate pool size variation (separate experiment)
- Changes to production `Settings` / `config.py` — eval configs are eval-only
- SPLADE sparse vectors (would require a different Qdrant modifier)
