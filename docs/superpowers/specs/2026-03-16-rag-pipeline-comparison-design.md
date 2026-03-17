# RAG Pipeline Comparison — Design Spec

**Date:** 2026-03-16
**Scope:** Add configurable pipeline variants and an ablation eval runner to compare retrieval approaches.

---

## Goal

Compare 6 retrieval pipeline configurations against the existing golden dataset to identify which components (BM25 sparse encoding, cross-encoder reranker, ColBERT late interaction, query rewriting) actually improve retrieval metrics. One run produces a side-by-side comparison table.

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

Ablation configs (each varies exactly one axis from the previous):

| name | sparse | reranker | colbert | rewriter |
|---|---|---|---|---|
| `baseline` | tfidf | off | off | off |
| `+rewriter` | tfidf | off | off | on |
| `+reranker` | tfidf | on | off | off |
| `+bm25` | bm25 | off | off | off |
| `+colbert` | tfidf | off | on | off |
| `full` | bm25 | on | on | on |

---

### 2. BM25 sparse encoding — modify `app/rag/knowledge_store.py`

`KnowledgeStore.__init__` gets a `sparse_mode: Literal["tfidf", "bm25"] = "tfidf"` parameter (default preserves current behavior).

`_build_vocabulary()` also computes `self._avgdl` (average document length in tokens across all chunks).

`_text_to_sparse()` switches on `sparse_mode`:

- **tfidf** (current): value = `float(count)` per token
- **bm25**: value = BM25 saturated TF with `k1=1.5, b=0.75`:
  ```
  tf_bm25 = count * (k1 + 1) / (count + k1 * (1 - b + b * doc_len / avgdl))
  ```

Qdrant applies IDF server-side via `Modifier.IDF` in both cases. No collection schema change — the sparse vector indices are identical, only values differ.

---

### 3. `app/rag/colbert_index.py` — new file

Adds ColBERT late-interaction scoring as an additional retrieval signal.

**Model:** `colbert-ir/colbertv2.0` via FastEmbed `LateInteractionTextEmbedding` (same library as the existing reranker — no new dependency).

**Index time:** `ColBERTIndex.build(chunks, qdrant_client, collection_name)` adds a third named vector `"colbert"` to the Qdrant collection using:
```python
MultiVectorParams(size=128, distance=Distance.COSINE, multivec_comparator=MultiVectorComparator.MAX_SIM)
```
Each chunk gets a list of 128-dim token embeddings. Qdrant computes MaxSim natively.

**Query time:** `ColBERTIndex.make_prefetch(query_text, limit) -> Prefetch` generates per-token query embeddings and returns a `Prefetch` object for the `"colbert"` vector.

**Collection schema:** ColBERT requires a different collection schema (multi-vector config). The eval runner builds two separate KnowledgeStore instances — one without ColBERT vectors, one with — and reuses them across all configs to avoid redundant embedding API calls.

**Integration with `KnowledgeStore.search_hybrid`:** Accepts an optional `colbert_prefetch: Prefetch | None` parameter. When present, it is added alongside the existing dense and sparse prefetches before `FusionQuery(fusion=Fusion.RRF)`. Qdrant fuses all three signals via RRF.

---

### 4. `eval/runners/comparison_runner.py` — new file

Runs all 6 ablation configs in sequence and produces a side-by-side comparison.

**Execution flow:**
1. Build two KnowledgeStore instances once (non-ColBERT, ColBERT-enabled) — avoids re-embedding 60 chunks per config
2. For each config in order:
   - Select the appropriate store
   - Set `sparse_mode` on the store instance
   - Wire reranker and query rewriter based on config flags
   - Run the eval loop (same logic as `retrieval_runner.py`)
   - Collect results under the config name
3. Print comparison table
4. Save to `eval/data/results/comparison_<timestamp>.json`

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

Results JSON also includes per-question-type breakdown for each config.

---

## Files Changed

| File | Change |
|---|---|
| `eval/pipeline_config.py` | New — PipelineConfig dataclass + 6 ablation configs |
| `app/rag/colbert_index.py` | New — ColBERT index build + prefetch |
| `app/rag/knowledge_store.py` | Add `sparse_mode` param, BM25 TF weights, `avgdl`, optional ColBERT prefetch in `search_hybrid` |
| `eval/runners/comparison_runner.py` | New — ablation comparison runner |

`retrieval_runner.py`, `retriever.py`, `reranker.py`, and all metrics files are unchanged.

---

## Out of Scope

- Tag boosting weight variation (separate experiment)
- Candidate pool size variation (separate experiment)
- Changes to production `Settings` / `config.py` — eval configs are eval-only
- SPLADE sparse vectors (would require a different Qdrant modifier)
