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
