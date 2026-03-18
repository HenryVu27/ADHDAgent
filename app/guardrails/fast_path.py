"""Semantic fast path for InputGate — local embedding-based classifier.

Encodes a set of benign example utterances at startup into a normalized
numpy matrix. At inference, computes cosine similarity (dot product on
L2-normalized vectors) and short-circuits the Gemini InputGate call if
max_score >= threshold.

Every failure mode returns None, deferring to the Gemini gate.
"""

import logging

import numpy as np

from app.models.schemas import InputCheckResult

logger = logging.getLogger(__name__)

# Benign example utterances — one per semantic anchor.
# Keep these short and unambiguous: no mixed messages, no partial coaching queries.
# Four clusters: greetings, acknowledgments, thank-yous, brief follow-ups.
BENIGN_EXAMPLES = [
    # Greetings
    "hey",
    "hi",
    "hello",
    "hey Ally",
    "hi there",
    "good morning",
    # Acknowledgments / confirmations
    "ok",
    "okay",
    "got it",
    "makes sense",
    "I see",
    "understood",
    # Thank-yous
    "thanks",
    "thank you",
    "thank you so much",
    # Brief follow-ups
    "sure",
    "sounds good",
    # NOTE: "yes" and "no" are intentionally excluded.
    # The spec requires threshold validation before including single-word affirmatives
    # (e.g. "no he's still hurting" must not score above threshold). Until the eval
    # runner confirms they are safe at the configured threshold, keep them out.
]


class SemanticFastPath:
    """Local cosine-similarity classifier that fast-paths clearly benign messages.

    Usage:
        fp = SemanticFastPath(model_name="all-MiniLM-L6-v2", threshold=0.82)
        fp.build_index()          # call once at startup — blocks ~1-2s on first run
        result = fp.classify(msg) # returns InputCheckResult or None
    """

    def __init__(self, model_name: str, threshold: float) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._embedder = None   # fastembed.TextEmbedding, set by build_index()
        self._index: np.ndarray | None = None  # shape (N, D), L2-normalized rows

    def build_index(self) -> None:
        """Load model and encode BENIGN_EXAMPLES into a normalized matrix.

        Called synchronously at app startup before the lifespan yield.
        Blocks ~1-2s on first run (model download + ONNX load); subsequent
        starts are faster once the model is cached by fastembed.

        On any failure, leaves _index = None so classify() falls through to Gemini.
        """
        try:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=self._model_name)
            vectors = list(self._embedder.embed(BENIGN_EXAMPLES))
            matrix = np.array(vectors, dtype=np.float32)  # (N, D)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            # Avoid division by zero for any zero-norm rows (shouldn't happen in practice)
            norms = np.where(norms == 0, 1.0, norms)
            self._index = matrix / norms
            logger.info(
                "SemanticFastPath: index built (%d examples, model=%s, threshold=%.2f)",
                len(BENIGN_EXAMPLES),
                self._model_name,
                self._threshold,
            )
        except Exception as e:
            logger.error("SemanticFastPath: build_index failed — fast path disabled: %s", e)
            self._index = None
            self._embedder = None

    def classify(self, message: str) -> InputCheckResult | None:
        """Return InputCheckResult if message is clearly benign, else None.

        Returns None (fall through to Gemini) in all error/uncertainty cases.
        Never raises.
        """
        if self._index is None or self._embedder is None:
            return None
        try:
            vecs = list(self._embedder.embed([message]))
            vec = np.array(vecs[0], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0.0:
                return None
            vec = vec / norm
            scores = self._index @ vec  # (N,) — dot product = cosine similarity
            max_score = float(np.max(scores))
            if max_score >= self._threshold:
                return InputCheckResult(
                    is_allowed=True,
                    route="flash",
                    fast_path_bypassed=True,
                    fast_path_score=round(max_score, 4),
                )
            return None
        except Exception as e:
            logger.warning("SemanticFastPath.classify failed (falling through to Gemini): %s", e)
            return None
