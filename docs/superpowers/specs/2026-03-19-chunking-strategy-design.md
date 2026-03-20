# Chunking Strategy Design

**Date:** 2026-03-19
**Status:** Draft
**Issue:** No chunking strategy -- single chunk per document

## Problem

Every document in the knowledge base is embedded as a single chunk. This works for the current
small curated corpus (~100 docs, 490-2671 chars each) but cannot scale to larger corpora or
longer-form content (research papers, clinical guidelines, web-scraped references). Retrieval
precision degrades when a single embedding must represent an entire multi-step strategy or
multi-topic document.

## Goals

- Support both structured JSON documents (current format) and longer-form preprocessed text
- Improve retrieval precision by chunking at semantic boundaries
- Implement two production-grade chunking strategies for comparison
- Make the strategy configurable with zero-downtime switching
- Preserve the current `full_doc` contract so the agent layer is unaffected

## Non-Goals

- Document parsing/ingestion pipeline (user will preprocess upstream)
- Changing the agent tools or prompt layer
- Replacing the existing hybrid search (dense + sparse + RRF) pipeline

## Design

### Chunking Strategies

Two strategies, plus a `"none"` baseline that preserves current behavior:

#### Strategy 1: Recursive/Structural + Contextual Headers

Combines structural awareness with the Anthropic contextual retrieval pattern.

**Document detection tiers:**
1. **Structured** (has `steps` or `key_points` fields): each step/key_point becomes a chunk,
   description becomes a chunk.
2. **Sectioned** (has headings like `## Section` or `\n\nTitle\n`): split on heading boundaries,
   recurse if any section exceeds target chunk size.
3. **Generic fallback**: recursive character splitting on `\n\n` -> `\n` -> `. ` -> ` `.

Target chunk size: 512 tokens (~2048 chars). Overlap: 64 tokens (~256 chars) -- overlap
applies only to generic fallback (tier 3) and large-section recursion (tier 2). Structural
chunks (tier 1: individual steps/key_points) have no overlap since they are discrete semantic
units.

**Contextual headers:** For each chunk, a Flash LLM call generates a 1-2 sentence context
prefix that is prepended to the chunk text before embedding. The header is stored separately
in the payload for inspection. Togglable via `RAG_CONTEXTUAL_HEADERS`.

**Header generation details:**
- Headers are generated in parallel batches (asyncio.gather) during `build_index()`.
- Generated headers are persisted in the Qdrant payload (`context_header` field) so they
  survive restarts without regeneration.
- Estimated cost for current corpus: ~500 chunks x ~50 output tokens = ~25K tokens per
  full rebuild (~$0.01 with Flash). Latency: ~10-15s with batched calls.
- If header generation fails for a chunk, the chunk is indexed without a header (graceful
  degradation).

Example header: `"This chunk is from 'Parent Training in Behavior Therapy', a strong-evidence
strategy for preschool/school-age children. It describes step 3 of 6."`

Chunk text for embedding: `"{context_header} {raw_text}"`

#### Strategy 2: Semantic Chunking

Embedding-based boundary detection using sentence-level similarity.

1. Split document into sentences (for structured docs, each step/key_point is a sentence).
2. Embed each sentence independently via Gemini batch embed.
3. Compute cosine similarity between adjacent sentence embeddings.
4. Split where similarity drops below `RAG_SEMANTIC_SIMILARITY_THRESHOLD`.
5. Merge resulting chunks up to target size (512 tokens).

No LLM cost -- only embedding calls.

### Chunk Schema

```python
@dataclass
class Chunk:
    chunk_id: str            # "{doc_id}__{type}_{index}"
    parent_document_id: str
    text: str                # chunk content with context header (if applicable)
    raw_text: str            # chunk content without header
    context_header: str      # empty for semantic strategy
    chunk_type: str          # "description" | "step" | "key_point" | "text_segment"
    chunk_index: int
    metadata: dict           # tags, age_range, evidence_level, source, citations, etc.
```

All parent document metadata is copied to each chunk for filtering.

### Qdrant Point ID Strategy

Qdrant point IDs remain sequential integers (0, 1, 2, ...) matching the position in the
`self.chunks` list, same as today. The string-format `chunk_id` (e.g.,
`"working_memory__step_2"`) is stored in the Qdrant payload, not used as the point ID.

The existing `_build_results` pattern (`chunk = self._store.chunks[chunk_idx]`) continues to
work because `build_index()` assigns `id=i` during enumerate. The `chunk_id` in the payload
is for observability and debugging only.

### `full_doc` Attachment

Today each chunk dict carries a `full_doc` key because there is one chunk per document. With
chunking, individual chunk dicts will NOT carry `full_doc` to avoid duplicating large
documents across many chunks.

Instead, `full_doc` is attached at two points:
1. **`_chunk_to_result()`**: looks up `full_doc` via
   `self._store.get_document_by_id(chunk["document_id"])` instead of reading
   `chunk.get("full_doc")`. The chunk dict key remains `document_id` (not
   `parent_document_id`) for consistency with the existing code and the
   `RetrievalResult.document_id` field. The `Chunk` dataclass uses `parent_document_id`
   internally, but the chunk dict written to `self.chunks` maps it to `document_id`.
2. **Parent dedup step**: already has the resolved `full_doc` from step 1.

This is the only change to `_chunk_to_result` -- the rest of its behavior is unchanged.

### Architecture

**New module: `app/rag/chunker.py`**

```
ChunkingStrategy (Protocol)
    chunk(document: dict) -> list[Chunk]

RecursiveContextualChunker(ChunkingStrategy)
    - _chunk_structured(doc) -> list[Chunk]
    - _chunk_sectioned(text) -> list[Chunk]
    - _chunk_recursive(text) -> list[Chunk]
    - _add_contextual_headers(chunks, doc) -> list[Chunk]

SemanticChunker(ChunkingStrategy)
    - _embed_sentences(sentences) -> list[list[float]]
    - _find_boundaries(embeddings, threshold) -> list[int]
    - _merge_small_chunks(chunks, target_size) -> list[Chunk]
```

**Modified files:**

| File | Change |
|------|--------|
| `app/config.py` | Add 6 chunking settings (see Configuration section) |
| `app/rag/knowledge_store.py` | `_create_chunks()` delegates to configured `ChunkingStrategy`. `build_index()` uses strategy-specific collection name. |
| `app/rag/retriever.py` | Modify `_chunk_to_result` to look up `full_doc` by `parent_document_id`. Add parent document deduplication after reranking. Apply dedup to both `_build_results` and `_keyword_fallback` paths. |
| `app/models/schemas.py` | Add optional `chunk_id`, `chunk_type` fields to `RetrievalResult`. |
| `app/main.py` | Wire chunker into `KnowledgeStore` based on config. |

**Untouched files:** `graph.py`, `orchestrator.py`, `hooks.py`, `prompts.py`, `tools.py`,
`memory.py`, `validator.py`, API routes, frontend.

### Configuration

New settings in `app/config.py`:

```python
RAG_CHUNKING_STRATEGY: str = "none"  # "recursive_contextual" | "semantic" | "none"
RAG_CHUNK_SIZE_TOKENS: int = 512
RAG_CHUNK_OVERLAP_TOKENS: int = 64
RAG_SEMANTIC_SIMILARITY_THRESHOLD: float = 0.75
RAG_CONTEXTUAL_HEADERS: bool = True
RAG_PARENT_DEDUP: bool = True
```

### Collection Management

Each strategy writes to its own Qdrant collection:
- `"none"` -> `adhd_knowledge` (current behavior)
- `"recursive_contextual"` -> `adhd_knowledge_recursive_contextual`
- `"semantic"` -> `adhd_knowledge_semantic`

All collections live in the same Qdrant instance. Switching strategies is a config change +
restart. Both collections can coexist simultaneously.

The collection name is derived deterministically from the strategy setting. The
`_collection_is_current()` check uses the strategy-specific collection name, so switching
strategies always triggers a rebuild into the correct collection.

### Parent Document Deduplication

Added to `retriever.py` after reranking, before facet computation:

1. Group candidates by `parent_document_id`.
2. For each group, keep the chunk with the highest score.
3. Attach `full_doc` via `knowledge_store.get_document_by_id()`.
4. Re-sort by winning chunk scores.

**Why highest-score-wins:** Score aggregation (sum/average) biases toward longer documents
with more chunks. Highest-score-wins measures how well the best-matching piece of a document
answers the query.

Togglable via `RAG_PARENT_DEDUP`. When `"none"` strategy is active, dedup is a no-op
(one chunk per doc already).

### Data Flow

```
Document (JSON or preprocessed text)
    |
    v
ChunkingStrategy.chunk(document)
    -> list[Chunk] (with metadata, chunk_ids, parent refs)
    |
    v
KnowledgeStore.build_index()
    -> Embed chunks (Gemini batch embed)
    -> Build sparse vectors
    -> Upsert to strategy-specific Qdrant collection
    |
    v
[Query time]
    |
    v
HybridRetriever._hybrid_search()
    -> Dense + sparse + RRF (search mechanics unchanged, but sparse vector
       vocabulary reflects chunked texts -- smaller chunks produce different
       term frequencies. Evaluation must compare sparse effectiveness.)
    |
    v
Reranker (cross-encoder, unchanged)
    |
    v
Parent document deduplication (NEW)
    -> Collapse chunks to one result per document
    -> Attach full_doc
    |
    v
Outcome boost + facets + trim (unchanged)
    |
    v
RetrievalResult with full_doc (agent sees same contract as today)
```

## Testing

### Unit Tests (`tests/test_chunker.py`)

- Structured doc with steps -> correct chunk count, chunk_ids, metadata
- Structured doc with key_points -> same
- Long text -> recursive splits respect target size and overlap
- Sectioned text -> splits on headings, recurses large sections
- Semantic chunker -> finds boundaries at topic shifts, merges small chunks
- Context header generation -> header prepended, raw_text preserved
- Edge cases: empty doc, description-only doc, single-step doc
- Generic fallback: unrecognized format falls through to recursive split

### Existing Test Additions

- `test_knowledge_store.py`: chunking strategy integration, collection naming, chunk count
- `test_retriever.py`: parent dedup -- multiple chunks collapse, highest score wins, full_doc attached

### Evaluation Script

Standalone script for strategy comparison using test queries:
- Recall@k: does the relevant document appear in top-k?
- MRR (Mean Reciprocal Rank): how high does the relevant document rank?
- Per-query comparison across strategies

Uses golden data from `eval/data/` or hand-curated query->expected_document pairs.

The eval script builds all three collections (none, recursive_contextual, semantic) in a
single run, then queries each with the same set of test queries for fair comparison.

All unit tests use mocks -- no real API keys required. The evaluation script requires a
GEMINI_API_KEY since it runs real embeddings and (for recursive_contextual) real LLM calls.

### Unaffected Methods

`KnowledgeStore.search_by_tags()`, `get_document_by_id()`, and `get_related_docs()` operate
on `self.documents` (the original document list), not `self.chunks`. They are unaffected by
chunking strategy changes.

## Strategy Comparison Summary

| Dimension | Recursive + Contextual | Semantic |
|-----------|----------------------|----------|
| Ingestion cost | LLM call per chunk (Flash) | Embed call per sentence |
| Boundary quality | Rule-based (structural) | Data-driven (similarity) |
| Retrieval precision | Higher (context headers) | Good (topic-coherent) |
| Determinism | Non-deterministic (LLM) | Deterministic |
| Long-form handling | May cut mid-topic | Finds natural boundaries |

## Migration

- `"none"` strategy is the default initially for backwards compatibility
- Existing tests pass unchanged
- Switch to `"recursive_contextual"` or `"semantic"` via env var
- Old collection remains until explicitly deleted
