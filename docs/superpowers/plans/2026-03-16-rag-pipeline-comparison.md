# RAG Pipeline Comparison Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 files/modifications that let a single command compare 6 RAG pipeline ablation configs (TF-IDF vs BM25 sparse, reranker on/off, ColBERT on/off, query rewriter on/off) and print a side-by-side metrics table.

**Architecture:** `PipelineConfig` dataclass defines the 6 configs. `KnowledgeStore` gains `sparse_mode` (TF-IDF or BM25) and `collection_name` params so each config can build its own indexed collection. `ColBERTIndex` wraps FastEmbed late-interaction embeddings and is passed into `build_index` and `HybridRetriever`. A new `comparison_runner` builds 4 distinct indexes once, wires each config, and runs the eval loop from `retrieval_runner.py`.

**Tech Stack:** Python 3.12, Qdrant in-memory (>=1.10), FastEmbed `LateInteractionTextEmbedding` (colbert-ir/colbertv2.0 ~500 MB first download), existing `eval/metrics/retrieval.py` for scoring.

---

## Chunk 1: PipelineConfig + BM25 sparse encoding

### Task 1: PipelineConfig dataclass

**Files:**
- Create: `eval/pipeline_config.py`
- Test: `tests/test_pipeline_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_config.py
from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig


def test_pipeline_config_fields():
    cfg = PipelineConfig(
        name="test",
        sparse_mode="tfidf",
        use_reranker=False,
        use_colbert=False,
        use_query_rewriter=False,
    )
    assert cfg.name == "test"
    assert cfg.sparse_mode == "tfidf"


def test_ablation_configs_count():
    assert len(ABLATION_CONFIGS) == 6


def test_ablation_config_names():
    names = [c.name for c in ABLATION_CONFIGS]
    assert names == ["baseline", "+rewriter", "+reranker", "+bm25", "+colbert", "full"]


def test_baseline_has_nothing_on():
    baseline = ABLATION_CONFIGS[0]
    assert baseline.sparse_mode == "tfidf"
    assert not baseline.use_reranker
    assert not baseline.use_colbert
    assert not baseline.use_query_rewriter


def test_full_has_everything_on():
    full = ABLATION_CONFIGS[5]
    assert full.sparse_mode == "bm25"
    assert full.use_reranker
    assert full.use_colbert
    assert full.use_query_rewriter
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/test_pipeline_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'eval.pipeline_config'`

- [ ] **Step 3: Create `eval/pipeline_config.py`**

```python
"""Pipeline configuration dataclass and ablation config definitions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class PipelineConfig:
    name: str
    sparse_mode: Literal["tfidf", "bm25"]
    use_reranker: bool
    use_colbert: bool
    use_query_rewriter: bool


# Each config varies exactly one axis from baseline.
# 'full' combines all axes and serves as a best-possible reference.
ABLATION_CONFIGS: list[PipelineConfig] = [
    PipelineConfig("baseline",  sparse_mode="tfidf", use_reranker=False, use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+rewriter",  sparse_mode="tfidf", use_reranker=False, use_colbert=False, use_query_rewriter=True),
    PipelineConfig("+reranker",  sparse_mode="tfidf", use_reranker=True,  use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+bm25",      sparse_mode="bm25",  use_reranker=False, use_colbert=False, use_query_rewriter=False),
    PipelineConfig("+colbert",   sparse_mode="tfidf", use_reranker=False, use_colbert=True,  use_query_rewriter=False),
    PipelineConfig("full",       sparse_mode="bm25",  use_reranker=True,  use_colbert=True,  use_query_rewriter=True),
]
```

- [ ] **Step 4: Run to verify it passes**

```bash
python -m pytest tests/test_pipeline_config.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/pipeline_config.py tests/test_pipeline_config.py
git commit -m "Add PipelineConfig dataclass and ablation configs"
```

---

### Task 2: BM25 sparse encoding in KnowledgeStore

**Files:**
- Modify: `app/rag/knowledge_store.py`
- Test: `tests/test_rag.py` (add to existing file)

The changes to `knowledge_store.py`:
1. `__init__` gains `sparse_mode="tfidf"` and `collection_name: str | None = None`
2. `self._avgdl: float = 0.0` initialized before `_load_documents()`
3. `_build_vocabulary` also computes `_avgdl`
4. `_text_to_sparse` is replaced by `_text_to_sparse_doc` (index time) and `_text_to_sparse_query` (query time)
5. All callers updated: `build_index` uses `_text_to_sparse_doc`; `search_hybrid` uses `_text_to_sparse_query`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rag.py`)

These tests inject vocabulary and avgdl directly so they don't depend on which words appear in the ADHD knowledge files.

```python
# --- BM25 sparse encoding tests ---

def _store_with_controlled_vocab(sparse_mode: str) -> KnowledgeStore:
    """Return a KnowledgeStore with a hand-crafted vocab and avgdl for unit testing."""
    store = KnowledgeStore(sparse_mode=sparse_mode)
    # Override vocab so "hello" and "world" are always present at known indices
    store._vocab = {"hello": 0, "world": 1, "foo": 2}
    store._avgdl = 5.0  # pretend average doc length is 5 tokens
    return store


def test_tfidf_sparse_doc_uses_raw_counts():
    store = _store_with_controlled_vocab("tfidf")
    vec = store._text_to_sparse_doc("hello hello world")
    hello_pos = vec.indices.index(0)  # index 0 = "hello"
    assert vec.values[hello_pos] == 2.0


def test_bm25_sparse_doc_saturates_tf():
    store = _store_with_controlled_vocab("bm25")
    # Single occurrence
    vec1 = store._text_to_sparse_doc("hello world")
    val1 = vec1.values[vec1.indices.index(0)]

    # Ten occurrences — BM25 saturation should prevent 10x linear growth
    vec10 = store._text_to_sparse_doc(" ".join(["hello"] * 10 + ["world"]))
    val10 = vec10.values[vec10.indices.index(0)]

    assert val10 < val1 * 10, "BM25 TF should saturate — not linear in count"
    assert val10 > val1, "Higher count should still increase score"


def test_bm25_sparse_doc_differs_from_tfidf():
    tfidf_store = _store_with_controlled_vocab("tfidf")
    bm25_store = _store_with_controlled_vocab("bm25")
    text = "hello hello hello world"
    tfidf_vec = tfidf_store._text_to_sparse_doc(text)
    bm25_vec = bm25_store._text_to_sparse_doc(text)
    tfidf_val = tfidf_vec.values[tfidf_vec.indices.index(0)]
    bm25_val = bm25_vec.values[bm25_vec.indices.index(0)]
    # BM25 should produce a different (saturated) value than raw TF=3
    assert bm25_val != tfidf_val


def test_query_sparse_uses_raw_counts_in_bm25_mode():
    store = _store_with_controlled_vocab("bm25")
    vec = store._text_to_sparse_query("hello hello world")
    hello_pos = vec.indices.index(0)
    # Query-side: raw count regardless of sparse_mode
    assert vec.values[hello_pos] == 2.0


def test_collection_name_override():
    store = KnowledgeStore(collection_name="my_collection")
    assert store._collection == "my_collection"


def test_default_collection_name_uses_settings():
    from app.config import settings
    store = KnowledgeStore()
    assert store._collection == settings.QDRANT_COLLECTION


def test_avgdl_computed():
    # Requires knowledge JSON files to be present (standard test environment assumption)
    store = KnowledgeStore()
    assert store._avgdl > 0.0, "Expected avgdl > 0 with real knowledge docs loaded"
```

- [ ] **Step 2: Run to verify tests fail**

```bash
python -m pytest tests/test_rag.py::test_tfidf_sparse_doc_uses_raw_counts tests/test_rag.py::test_bm25_sparse_doc_saturates_tf tests/test_rag.py::test_avgdl_computed -v
```
Expected: `AttributeError` or similar — methods don't exist yet.

- [ ] **Step 3: Modify `app/rag/knowledge_store.py`**

In `__init__`, add `sparse_mode` and `collection_name` params and `_avgdl` init:

```python
def __init__(
    self,
    knowledge_dir: Path | None = None,
    sparse_mode: str = "tfidf",
    collection_name: str | None = None,
):
    self.knowledge_dir = knowledge_dir or (
        Path(__file__).parent.parent / "knowledge"
    )
    self.documents: list[dict] = []
    self.chunks: list[dict] = []
    self._client: QdrantClient | None = None
    self._collection = collection_name or settings.QDRANT_COLLECTION
    self._indexed = False
    self._vocab: dict[str, int] = {}
    self._avgdl: float = 0.0
    self._sparse_mode = sparse_mode

    self._load_documents()
```

Replace `_build_vocabulary` to also compute `_avgdl`:

```python
def _build_vocabulary(self):
    self._vocab = {}
    next_id = 0
    total_tokens = 0
    for chunk in self.chunks:
        tokens = self._tokenize(chunk["text"])
        total_tokens += len(tokens)
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = next_id
                next_id += 1
    self._avgdl = total_tokens / len(self.chunks) if self.chunks else 0.0
```

Replace `_text_to_sparse` with two methods:

```python
def _text_to_sparse_doc(self, text: str) -> SparseVector:
    """Sparse vector for indexing a document chunk.

    TF-IDF mode: raw token count as value.
    BM25 mode: saturated TF with k1=1.5, b=0.75 (Qdrant applies IDF server-side).
    """
    tokens = self._tokenize(text)
    counts = Counter(tokens)
    doc_len = len(tokens)
    k1, b = 1.5, 0.75
    avgdl = self._avgdl if self._avgdl > 0 else 1.0  # guard against division by zero

    indices = []
    values = []
    for token, count in sorted(counts.items()):
        if token not in self._vocab:
            continue
        indices.append(self._vocab[token])
        if self._sparse_mode == "bm25":
            tf_bm25 = count * (k1 + 1) / (count + k1 * (1 - b + b * doc_len / avgdl))
            values.append(tf_bm25)
        else:
            values.append(float(count))

    if not indices:
        return SparseVector(indices=[0], values=[0.0])
    return SparseVector(indices=indices, values=values)

def _text_to_sparse_query(self, text: str) -> SparseVector:
    """Sparse vector for a query string.

    Uses raw counts in both modes. BM25 saturation is near-identity for
    query terms (typically appear once) and would apply avgdl against the
    wrong length — raw counts are correct here.
    """
    tokens = self._tokenize(text)
    counts = Counter(tokens)

    indices = []
    values = []
    for token, count in sorted(counts.items()):
        if token not in self._vocab:
            continue
        indices.append(self._vocab[token])
        values.append(float(count))

    if not indices:
        return SparseVector(indices=[0], values=[0.0])
    return SparseVector(indices=indices, values=values)
```

Find the call to `self._text_to_sparse(chunk["text"])` inside `build_index` and replace it:
```python
sparse_vec = self._text_to_sparse_doc(chunk["text"])
```

Find the call to `self._text_to_sparse(query_text)` inside `search_hybrid` and replace it:
```python
sparse_vec = self._text_to_sparse_query(query_text)
```

(Do not rely on line numbers — they shift as earlier edits add lines. Search for the call text instead.)

- [ ] **Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_rag.py -v
```
Expected: all pass including the 6 new tests.

- [ ] **Step 5: Commit**

```bash
git add app/rag/knowledge_store.py tests/test_rag.py
git commit -m "Add BM25 sparse encoding and collection_name param to KnowledgeStore"
```

---

## Chunk 2: ColBERTIndex + HybridRetriever integration

### Task 3: ColBERTIndex

**Files:**
- Create: `app/rag/colbert_index.py`
- Test: `tests/test_colbert_index.py`

Tests mock `LateInteractionTextEmbedding` to avoid the 500 MB model download in CI.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_colbert_index.py
from unittest.mock import MagicMock, patch

import pytest


import numpy as np


def make_mock_model(n_tokens: int = 3, dim: int = 128):
    """Return a mock LateInteractionTextEmbedding that yields numpy token matrices.

    Uses np.array to match real FastEmbed output — ensures .tolist() conversion
    is tested rather than the trivially-correct list() on plain Python lists.
    """
    model = MagicMock()
    model.embed.return_value = iter([np.array([[0.1] * dim] * n_tokens)])
    model.query_embed.return_value = iter([np.array([[0.2] * dim] * 2)])
    return model


@patch("app.rag.colbert_index.LateInteractionTextEmbedding")
def test_embed_chunks_returns_one_matrix_per_chunk(mock_cls):
    mock_cls.return_value = make_mock_model(n_tokens=3)
    from app.rag.colbert_index import ColBERTIndex
    index = ColBERTIndex()
    chunks = [{"text": "hello world"}, {"text": "foo bar"}]
    # Reset mock to return 2 numpy matrices
    index._model.embed.return_value = iter([
        np.array([[0.1] * 128] * 3),
        np.array([[0.2] * 128] * 5),
    ])
    result = index.embed_chunks(chunks)
    assert len(result) == 2
    assert len(result[0]) == 3       # 3 tokens
    assert len(result[0][0]) == 128
    # Values must be plain Python floats, not numpy scalars
    assert isinstance(result[0][0][0], float)


@patch("app.rag.colbert_index.LateInteractionTextEmbedding")
def test_make_prefetch_returns_prefetch_object(mock_cls):
    mock_cls.return_value = make_mock_model()
    from app.rag.colbert_index import ColBERTIndex
    from qdrant_client.models import Prefetch
    index = ColBERTIndex()
    index._model.query_embed.return_value = iter([np.array([[0.2] * 128] * 2)])
    prefetch = index.make_prefetch("what strategies help with homework", limit=10)
    assert isinstance(prefetch, Prefetch)
    # Verify query vector elements are plain Python floats, not numpy scalars
    assert isinstance(prefetch.query[0][0], float)
```

- [ ] **Step 2: Run to verify tests fail**

```bash
python -m pytest tests/test_colbert_index.py -v
```
Expected: `ModuleNotFoundError: No module named 'app.rag.colbert_index'`

- [ ] **Step 3: Create `app/rag/colbert_index.py`**

```python
"""ColBERT late-interaction index using FastEmbed LateInteractionTextEmbedding.

Provides per-token embeddings for Qdrant multi-vector storage (MaxSim scoring).
Does NOT manage the Qdrant collection — KnowledgeStore.build_index owns that.
"""
import logging

from qdrant_client.models import Prefetch

logger = logging.getLogger(__name__)

# Import lazily so the 500 MB model download is deferred until first use
try:
    from fastembed.late_interaction import LateInteractionTextEmbedding
except ImportError:
    LateInteractionTextEmbedding = None  # type: ignore[assignment,misc]


class ColBERTIndex:
    """Wraps colbert-ir/colbertv2.0 for per-token embedding generation.

    embed_chunks: called once at index time by KnowledgeStore.build_index.
    make_prefetch: called per query by HybridRetriever._hybrid_search.
    """

    MODEL_NAME = "colbert-ir/colbertv2.0"
    DIM = 128

    def __init__(self, model_name: str = MODEL_NAME):
        if LateInteractionTextEmbedding is None:
            raise ImportError(
                "fastembed late_interaction support not installed. "
                "Run: pip install fastembed"
            )
        logger.info("Loading ColBERT model %s (~500 MB first download)...", model_name)
        self._model = LateInteractionTextEmbedding(model_name)
        logger.info("ColBERT model loaded.")

    def embed_chunks(self, chunks: list[dict]) -> list[list[list[float]]]:
        """Generate per-token embeddings for all chunks.

        Returns a list of token matrices, one per chunk.
        Each matrix is List[List[float]] with shape [n_tokens, 128].
        Pass directly as vector["colbert"] in a PointStruct — do NOT flatten.
        """
        texts = [chunk["text"] for chunk in chunks]
        # LateInteractionTextEmbedding.embed is synchronous.
        # Callers inside async code must wrap with asyncio.to_thread.
        matrices = list(self._model.embed(texts))
        # Use .tolist() to convert numpy arrays to plain Python floats.
        # list(numpy_array) produces numpy scalar elements, which Qdrant rejects.
        return [
            [token_vec.tolist() for token_vec in matrix]
            for matrix in matrices
        ]

    def make_prefetch(self, query_text: str, limit: int) -> Prefetch:
        """Build a Qdrant Prefetch for the 'colbert' named vector.

        limit should equal the prefetch_limit already computed in
        HybridRetriever._hybrid_search (min(top_k * 3, n_chunks)).
        """
        query_matrices = list(self._model.query_embed([query_text]))
        # Use .tolist() — list() on a numpy array produces numpy scalar elements
        # which Qdrant rejects. .tolist() recurses and returns plain Python floats.
        query_matrix = [token_vec.tolist() for token_vec in query_matrices[0]]
        return Prefetch(
            query=query_matrix,
            using="colbert",
            limit=limit,
        )
```

- [ ] **Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_colbert_index.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag/colbert_index.py tests/test_colbert_index.py
git commit -m "Add ColBERTIndex for per-token late-interaction embeddings"
```

---

### Task 4: Wire ColBERT into KnowledgeStore.build_index

**Files:**
- Modify: `app/rag/knowledge_store.py`
- Test: `tests/test_rag.py` (add tests)

The Qdrant imports needed: `MultiVectorParams`, `MultiVectorComparator` (add to existing import block).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rag.py`)

```python
# --- ColBERT collection schema tests ---

from unittest.mock import AsyncMock, MagicMock, patch


def make_mock_colbert():
    """ColBERTIndex mock that returns 2-token 128-dim matrices."""
    colbert = MagicMock()
    colbert.embed_chunks.return_value = [[[0.1] * 128] * 2] * 60  # one per chunk
    return colbert


def make_mock_gemini():
    """GeminiClient mock that returns 768-dim embeddings."""
    gemini = MagicMock()
    gemini.embed_batch = AsyncMock(return_value=[[0.1] * 768] * 60)
    return gemini


@pytest.mark.asyncio
async def test_build_index_with_colbert_creates_multivector_collection():
    from qdrant_client import QdrantClient
    store = KnowledgeStore(collection_name="test_colbert")
    store._client = QdrantClient(location=":memory:")

    colbert = make_mock_colbert()
    colbert.embed_chunks.return_value = [[[0.1] * 128] * 2] * len(store.chunks)
    gemini = make_mock_gemini()
    gemini.embed_batch = AsyncMock(return_value=[[0.1] * 768] * len(store.chunks))

    await store.build_index(gemini, colbert_index=colbert)

    info = store._client.get_collection("test_colbert")
    assert info.config.params.multi_vectors_config is not None
    assert "colbert" in info.config.params.multi_vectors_config


@pytest.mark.asyncio
async def test_build_index_without_colbert_has_no_multivector():
    from qdrant_client import QdrantClient
    store = KnowledgeStore(collection_name="test_no_colbert")
    store._client = QdrantClient(location=":memory:")

    gemini = make_mock_gemini()
    gemini.embed_batch = AsyncMock(return_value=[[0.1] * 768] * len(store.chunks))

    await store.build_index(gemini)

    info = store._client.get_collection("test_no_colbert")
    # multi_vectors_config should be absent or empty
    mvc = info.config.params.multi_vectors_config
    assert not mvc or "colbert" not in mvc
```

- [ ] **Step 2: Run to verify tests fail**

```bash
python -m pytest tests/test_rag.py::test_build_index_with_colbert_creates_multivector_collection tests/test_rag.py::test_build_index_without_colbert_has_no_multivector -v
```
Expected: fail — `build_index` doesn't accept `colbert_index` yet.

- [ ] **Step 3: Modify `app/rag/knowledge_store.py`**

Add to imports at the top:
```python
from qdrant_client.models import (
    ...  # existing imports
    MultiVectorComparator,
    MultiVectorParams,
)
```

Add optional `colbert_index` type hint import (avoid circular import — use `TYPE_CHECKING`):
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.rag.colbert_index import ColBERTIndex
```

Update `build_index` signature:
```python
async def build_index(self, gemini_client, colbert_index: ColBERTIndex | None = None):
```

Update `recreate_collection` call inside `build_index` to conditionally include `multi_vectors_config`:
```python
multi_vec_cfg = {}
if colbert_index is not None:
    # Use literal 128 — ColBERTIndex is TYPE_CHECKING-only and not accessible
    # at runtime. colbertv2.0 always outputs 128-dim token embeddings.
    multi_vec_cfg = {
        "colbert": MultiVectorParams(
            size=128,
            distance=Distance.COSINE,
            multivec_comparator=MultiVectorComparator.MAX_SIM,
        )
    }

self._client.recreate_collection(
    collection_name=self._collection,
    vectors_config={
        "dense": VectorParams(size=dim, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(modifier=Modifier.IDF),
    },
    **({"multi_vectors_config": multi_vec_cfg} if multi_vec_cfg else {}),
)
```

Before the `upsert` call, generate ColBERT embeddings (via `asyncio.to_thread` to avoid blocking event loop):
```python
colbert_matrices: list[list[list[float]]] | None = None
if colbert_index is not None:
    import asyncio
    colbert_matrices = await asyncio.to_thread(colbert_index.embed_chunks, self.chunks)
```

Update `PointStruct` construction to include `"colbert"` when present:
```python
for i, (embedding, chunk) in enumerate(zip(raw_embeddings, self.chunks)):
    sparse_vec = self._text_to_sparse_doc(chunk["text"])
    vector: dict = {
        "dense": embedding,
        "sparse": sparse_vec,
    }
    if colbert_matrices is not None:
        vector["colbert"] = colbert_matrices[i]
    points.append(PointStruct(id=i, vector=vector, payload={...}))
```

Update `_collection_is_current` to check for ColBERT multi-vector when `colbert_index` is provided. Change the method signature to accept optional `colbert_index`:
```python
def _collection_is_current(self, colbert_index=None) -> bool:
    try:
        ...  # existing checks unchanged up to the point count check
        if colbert_index is not None:
            mvc = info.config.params.multi_vectors_config
            if not mvc or "colbert" not in mvc:
                logger.info("Collection missing ColBERT multi-vectors — rebuilding")
                return False
        if info.points_count == len(self.chunks):
            return True
        ...
    except Exception as e:
        ...
```

Update the call site in `build_index`:
```python
if self._collection_is_current(colbert_index):
```

Update log message at end:
```python
logger.info(
    f"Qdrant collection built: {len(points)} vectors, "
    f"dim={dim}, vocab={len(self._vocab)}, colbert={'yes' if colbert_index else 'no'}"
)
```

- [ ] **Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_rag.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/rag/knowledge_store.py tests/test_rag.py
git commit -m "Wire ColBERT multi-vector support into KnowledgeStore.build_index"
```

---

### Task 5: Wire ColBERT into HybridRetriever

**Files:**
- Modify: `app/rag/retriever.py`
- Test: `tests/test_rag.py` (add tests)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rag.py`)

```python
# --- ColBERT prefetch in HybridRetriever ---

def test_hybrid_retriever_accepts_colbert_index():
    from app.rag.colbert_index import ColBERTIndex
    from app.rag.retriever import HybridRetriever
    colbert = MagicMock(spec=ColBERTIndex)
    retriever = HybridRetriever(
        knowledge_store=KnowledgeStore(),
        gemini_client=None,
        colbert_index=colbert,
    )
    assert retriever._colbert is colbert


def test_hybrid_retriever_without_colbert_has_none():
    retriever = HybridRetriever(knowledge_store=KnowledgeStore(), gemini_client=None)
    assert retriever._colbert is None
```

- [ ] **Step 2: Run to verify tests fail**

```bash
python -m pytest tests/test_rag.py::test_hybrid_retriever_accepts_colbert_index tests/test_rag.py::test_hybrid_retriever_without_colbert_has_none -v
```
Expected: fail — `HybridRetriever.__init__` doesn't accept `colbert_index`.

- [ ] **Step 3: Modify `app/rag/retriever.py`**

Add `TYPE_CHECKING` import for `ColBERTIndex` (avoid circular import):
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.rag.colbert_index import ColBERTIndex
```

Add `colbert_index` param to `__init__`:
```python
def __init__(
    self,
    knowledge_store: KnowledgeStore,
    gemini_client=None,
    query_rewriter: QueryRewriter | None = None,
    reranker: FastEmbedReranker | None = None,
    colbert_index: ColBERTIndex | None = None,
):
    self._store = knowledge_store
    self._gemini = gemini_client
    self._rewriter = query_rewriter
    self._reranker = reranker
    self._colbert = colbert_index
```

Update `_hybrid_search` to pass ColBERT prefetch when available:
```python
async def _hybrid_search(
    self,
    query: str,
    top_k: int,
    filters: RetrievalFilters | None = None,
) -> list[RetrievalResult]:
    query_tags = set(query.lower().split())

    if self._store.has_sparse and self._gemini:
        query_vector = await self._gemini.embed(query, timeout=settings.RAG_EMBED_TIMEOUT_S)
        prefetch_limit = min(top_k * 3, len(self._store.chunks))

        colbert_prefetch = None
        if self._colbert is not None:
            import asyncio
            colbert_prefetch = await asyncio.to_thread(
                self._colbert.make_prefetch, query, prefetch_limit
            )

        hybrid_results = self._store.search_hybrid(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            filters=filters,
            colbert_prefetch=colbert_prefetch,
        )
        if hybrid_results:
            return self._build_results(hybrid_results, query_tags)

    return self._keyword_fallback(query, top_k)
```

Note: `search_hybrid` in `knowledge_store.py` already accepts `colbert_prefetch` from Task 4 changes. Add it to `search_hybrid`'s parameter list and prefetch assembly:

In `knowledge_store.py` `search_hybrid`, add param and use it:
```python
def search_hybrid(
    self,
    query_vector: list[float],
    query_text: str,
    top_k: int = 10,
    filters: RetrievalFilters | None = None,
    colbert_prefetch=None,
) -> list[tuple[int, float, dict]]:
    ...
    prefetches = [
        Prefetch(query=query_vector, using="dense", limit=prefetch_limit),
        Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit),
    ]
    if colbert_prefetch is not None:
        prefetches.append(colbert_prefetch)

    results = self._client.query_points(
        collection_name=self._collection,
        prefetch=prefetches,
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=qdrant_filter,
        limit=top_k,
    ).points
    ...
```

- [ ] **Step 4: Run to verify tests pass**

```bash
python -m pytest tests/test_rag.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add app/rag/retriever.py app/rag/knowledge_store.py tests/test_rag.py
git commit -m "Wire ColBERT prefetch into HybridRetriever and search_hybrid"
```

---

## Chunk 3: Comparison runner

### Task 6: comparison_runner

**Files:**
- Create: `eval/runners/comparison_runner.py`
- Test: `tests/test_comparison_runner.py`

The runner builds 4 KnowledgeStore instances once, then wires each of the 6 ablation configs and runs the eval loop.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_comparison_runner.py
"""Unit tests for comparison_runner — mocks all external dependencies."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig


def make_mock_retrieval_response(doc_ids: list[str]):
    from app.models.schemas import RetrievalResponse, RetrievalResult
    results = [
        RetrievalResult(
            document_id=doc_id,
            document_name="test",
            content="test content",
            score=0.9,
            match_type="hybrid",
            source="",
            tags=[],
            evidence_level="",
            document_type="",
            age_range=[],
            citations=[],
            full_doc={},
        )
        for doc_id in doc_ids
    ]
    return RetrievalResponse(results=results, facets=MagicMock(), rewritten_query=None)


def test_index_key_mapping():
    """Each config maps to the right (sparse_mode, use_colbert) store key."""
    from eval.runners.comparison_runner import _index_key

    assert _index_key(ABLATION_CONFIGS[0]) == ("tfidf", False)  # baseline
    assert _index_key(ABLATION_CONFIGS[3]) == ("bm25", False)   # +bm25
    assert _index_key(ABLATION_CONFIGS[4]) == ("tfidf", True)   # +colbert
    assert _index_key(ABLATION_CONFIGS[5]) == ("bm25", True)    # full


def test_all_configs_have_valid_index_keys():
    from eval.runners.comparison_runner import _index_key

    valid_keys = {("tfidf", False), ("bm25", False), ("tfidf", True), ("bm25", True)}
    for cfg in ABLATION_CONFIGS:
        assert _index_key(cfg) in valid_keys


@pytest.mark.asyncio
async def test_run_single_config_produces_metrics():
    """_run_config returns a dict with MRR and recall keys."""
    from eval.runners.comparison_runner import _run_config

    mock_retriever = MagicMock()
    mock_retriever.retrieve = AsyncMock(
        return_value=make_mock_retrieval_response(["doc_001", "doc_002"])
    )

    dataset = [
        {"question": "what helps with homework?", "expected_doc_ids": ["doc_001"], "question_type": "fact_single"},
        {"question": "emotional regulation tips?", "expected_doc_ids": ["doc_999"], "question_type": "reasoning"},
    ]

    result = await _run_config(mock_retriever, dataset, ks=[1, 3, 5])
    assert "mrr" in result["overall"]
    assert "recall@5" in result["overall"]
    assert "fact_single" in result["by_question_type"]
```

- [ ] **Step 2: Run to verify tests fail**

```bash
python -m pytest tests/test_comparison_runner.py -v
```
Expected: `ModuleNotFoundError: No module named 'eval.runners.comparison_runner'`

- [ ] **Step 3: Create `eval/runners/comparison_runner.py`**

```python
"""Run all 6 ablation pipeline configs against the golden retrieval dataset.

Builds 4 distinct KnowledgeStore indexes once (keyed by sparse_mode x colbert),
then wires each config and runs the eval loop. Prints a side-by-side table.

Usage (from project root):
    python -m eval.runners.comparison_runner [--k 1 3 5] [--types fact_single reasoning]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict

from app.config import settings
from app.llm.client import GeminiClient
from app.models.schemas import SessionState
from app.rag.knowledge_store import KnowledgeStore
from app.rag.query_rewriter import QueryRewriter
from app.rag.reranker import FastEmbedReranker
from app.rag.retriever import HybridRetriever
from eval.config import GOLDEN_RETRIEVAL_PATH, RESULTS_DIR
from eval.metrics.retrieval import aggregate, mrr, ndcg_at_k, precision_at_k, recall_at_k
from eval.pipeline_config import ABLATION_CONFIGS, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Collection names — one per (sparse_mode, use_colbert) combination
_COLLECTION_NAMES: dict[tuple[str, bool], str] = {
    ("tfidf", False): "rag_tfidf",
    ("bm25",  False): "rag_bm25",
    ("tfidf", True):  "rag_tfidf_colbert",
    ("bm25",  True):  "rag_bm25_colbert",
}


def _index_key(cfg: PipelineConfig) -> tuple[str, bool]:
    return (cfg.sparse_mode, cfg.use_colbert)


async def _build_indexes(
    gemini: GeminiClient,
    configs: list[PipelineConfig],
) -> tuple[dict[tuple[str, bool], KnowledgeStore], "ColBERTIndex | None"]:
    """Build one KnowledgeStore per unique (sparse_mode, use_colbert) combination.

    Returns (stores, colbert_instance). colbert_instance is None if no config
    uses ColBERT. The same instance is reused across all colbert-enabled configs
    to avoid loading the 500 MB model multiple times.
    """
    from app.rag.colbert_index import ColBERTIndex

    needed_keys = {_index_key(cfg) for cfg in configs}
    stores: dict[tuple[str, bool], KnowledgeStore] = {}

    # Instantiate ColBERTIndex once — shared across colbert-enabled configs
    colbert: ColBERTIndex | None = None
    if any(key[1] for key in needed_keys):
        logger.info("Loading ColBERT model (first run downloads ~500 MB)...")
        colbert = ColBERTIndex()

    for key in needed_keys:
        sparse_mode, use_colbert = key
        collection_name = _COLLECTION_NAMES[key]
        logger.info("Building index: sparse=%s colbert=%s collection=%s",
                    sparse_mode, use_colbert, collection_name)
        store = KnowledgeStore(sparse_mode=sparse_mode, collection_name=collection_name)
        await store.build_index(gemini, colbert_index=colbert if use_colbert else None)
        stores[key] = store

    return stores, colbert


async def _run_config(
    retriever: HybridRetriever,
    dataset: list[dict],
    ks: list[int],
) -> dict:
    """Run the eval loop for one pipeline config. Returns aggregated results."""
    per_query: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)

    # Minimal stub state so query rewriter condition is satisfied when enabled
    stub_state = SessionState(session_id="eval", conversation_history=[
        {"role": "user", "content": "stub"}
    ])

    for item in dataset:
        question = item["question"]
        expected_ids = set(item.get("expected_doc_ids", []))
        q_type = item.get("question_type", "unknown")
        if not expected_ids:
            continue

        response = await retriever.retrieve(question, state=stub_state)
        retrieved_ids = [r.document_id for r in response.results]

        scores: dict = {"question_type": q_type, "mrr": mrr(retrieved_ids, expected_ids)}
        for k in ks:
            scores[f"recall@{k}"] = recall_at_k(retrieved_ids, expected_ids, k)
            scores[f"precision@{k}"] = precision_at_k(retrieved_ids, expected_ids, k)
            scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, expected_ids, k)

        per_query.append(scores)
        per_type[q_type].append(scores)

    return {
        "overall": aggregate(per_query),
        "by_question_type": {t: aggregate(s) for t, s in per_type.items()},
        "n_questions": len(per_query),
    }


async def run(ks: list[int], filter_types: list[str] | None) -> dict:
    if not GOLDEN_RETRIEVAL_PATH.exists():
        logger.error("Golden dataset not found at %s — run dataset_builder first",
                     GOLDEN_RETRIEVAL_PATH)
        sys.exit(1)

    with open(GOLDEN_RETRIEVAL_PATH) as f:
        dataset = json.load(f)
    if filter_types:
        dataset = [q for q in dataset if q.get("question_type") in filter_types]
    logger.info("Loaded %d questions from golden dataset", len(dataset))

    # Force query rewriting to be usable (runner controls on/off via QueryRewriter instance)
    settings.RAG_USE_QUERY_REWRITE = True

    gemini = GeminiClient()
    reranker = FastEmbedReranker()
    query_rewriter = QueryRewriter(gemini)
    stores, shared_colbert = await _build_indexes(gemini, ABLATION_CONFIGS)

    all_results: dict[str, dict] = {}

    for cfg in ABLATION_CONFIGS:
        logger.info("Running config: %s", cfg.name)
        store = stores[_index_key(cfg)]

        # Reuse the single ColBERTIndex instance loaded during _build_indexes.
        # Do NOT instantiate a new ColBERTIndex here — that would load the 500 MB
        # model a second time.
        colbert = shared_colbert if cfg.use_colbert else None

        retriever = HybridRetriever(
            knowledge_store=store,
            gemini_client=gemini,
            query_rewriter=query_rewriter if cfg.use_query_rewriter else None,
            reranker=reranker if cfg.use_reranker else None,
            colbert_index=colbert,
        )

        result = await _run_config(retriever, dataset, ks)
        all_results[cfg.name] = result
        logger.info("Config '%s' done — MRR=%.3f", cfg.name, result["overall"].get("mrr", 0))

    _print_report(all_results, ks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"comparison_{int(time.time())}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


def _print_report(results: dict[str, dict], ks: list[int]) -> None:
    header_cols = ["MRR"] + [f"R@{k}" for k in ks] + [f"NDCG@{k}" for k in ks]
    col_w = 7
    name_w = 14

    print("\n" + "=" * (name_w + len(header_cols) * (col_w + 3) + 2))
    print(f"{'Pipeline':<{name_w}} | " + " | ".join(f"{h:>{col_w}}" for h in header_cols))
    print("-" * (name_w + len(header_cols) * (col_w + 3) + 2))

    for name, data in results.items():
        overall = data.get("overall", {})
        vals = [overall.get("mrr", 0)]
        for k in ks:
            vals.append(overall.get(f"recall@{k}", 0))
        for k in ks:
            vals.append(overall.get(f"ndcg@{k}", 0))
        row = " | ".join(f"{v:>{col_w}.3f}" for v in vals)
        print(f"{name:<{name_w}} | {row}")

    print("=" * (name_w + len(header_cols) * (col_w + 3) + 2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RAG pipeline ablation configs")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5], metavar="K")
    parser.add_argument("--types", nargs="+", default=None, metavar="TYPE")
    args = parser.parse_args()
    asyncio.run(run(ks=args.k, filter_types=args.types))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_comparison_runner.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run full test suite to check nothing broken**

```bash
python -m pytest tests/ -v -m "not integration"
```
Expected: all pass. Fix any regressions before committing.

- [ ] **Step 6: Commit**

```bash
git add eval/runners/comparison_runner.py tests/test_comparison_runner.py
git commit -m "Add comparison_runner for ablation pipeline eval"
```

---

## Final verification

- [ ] Confirm the runner is importable end-to-end:

```bash
python -c "from eval.runners.comparison_runner import run; print('OK')"
```

- [ ] Confirm CLI help works:

```bash
python -m eval.runners.comparison_runner --help
```

- [ ] If a golden dataset exists, do a smoke run with 5 questions to check for runtime errors (does NOT require full golden dataset):

```bash
python -c "
import asyncio, json
from pathlib import Path
from eval.config import GOLDEN_RETRIEVAL_PATH
data = json.loads(GOLDEN_RETRIEVAL_PATH.read_text())[:5]
tmp = Path('/tmp/golden_smoke.json')
tmp.write_text(json.dumps(data))

# Import the module first, then patch the module-level name.
# Patching eval.config won't work — comparison_runner binds the name locally.
import eval.runners.comparison_runner as cr
cr.GOLDEN_RETRIEVAL_PATH = tmp

asyncio.run(cr.run(ks=[1, 3], filter_types=None))
"
```
Expected: table printed, no exceptions.
