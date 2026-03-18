"""Tests for SemanticFastPath classifier."""
import numpy as np
import pytest

from app.models.schemas import InputCheckResult


class TestSemanticFastPath:
    """Tests that do NOT require the real fastembed model (unit tests)."""

    def _make_fast_path(self, threshold=0.82):
        """Build a SemanticFastPath with a fake pre-built index (no model needed)."""
        from app.guardrails.fast_path import SemanticFastPath
        fp = SemanticFastPath(model_name="test-model", threshold=threshold)
        # Inject a synthetic index: 2 unit vectors
        fp._index = np.array([
            [1.0, 0.0, 0.0],  # "hello" direction
            [0.0, 1.0, 0.0],  # "thanks" direction
        ], dtype=np.float32)
        # Inject a fake embedder that maps strings to known vectors
        class FakeEmbedder:
            def embed(self, texts):
                mapping = {
                    "hi there": np.array([0.99, 0.01, 0.0], dtype=np.float32),
                    "thank you": np.array([0.01, 0.99, 0.0], dtype=np.float32),
                    "my son won't sleep": np.array([0.3, 0.3, 0.9], dtype=np.float32),
                    "zero vector": np.array([0.0, 0.0, 0.0], dtype=np.float32),
                }
                return (mapping.get(t, np.array([0.5, 0.5, 0.0])) for t in texts)
        fp._embedder = FakeEmbedder()
        return fp

    def test_classify_benign_returns_result(self):
        fp = self._make_fast_path(threshold=0.82)
        result = fp.classify("hi there")
        assert result is not None
        assert isinstance(result, InputCheckResult)
        assert result.is_allowed is True
        assert result.route == "flash"
        assert result.fast_path_bypassed is True
        assert result.fast_path_score is not None
        assert result.fast_path_score >= 0.82

    def test_classify_benign_thank_you(self):
        fp = self._make_fast_path(threshold=0.82)
        result = fp.classify("thank you")
        assert result is not None
        assert result.fast_path_bypassed is True

    def test_classify_complex_returns_none(self):
        fp = self._make_fast_path(threshold=0.82)
        result = fp.classify("my son won't sleep")
        assert result is None

    def test_classify_zero_vector_returns_none(self):
        """Zero-norm embedding should not raise, should return None."""
        fp = self._make_fast_path(threshold=0.82)
        result = fp.classify("zero vector")
        assert result is None

    def test_classify_no_index_returns_none(self):
        """If build_index() was never called or failed, classify returns None."""
        from app.guardrails.fast_path import SemanticFastPath
        fp = SemanticFastPath(model_name="test-model", threshold=0.82)
        # _index and _embedder are None (not built)
        result = fp.classify("hey")
        assert result is None

    def test_classify_score_stored_on_result(self):
        fp = self._make_fast_path(threshold=0.50)  # low threshold so both fire
        result = fp.classify("hi there")
        assert result is not None
        assert isinstance(result.fast_path_score, float)
        assert 0.0 <= result.fast_path_score <= 1.0

    def test_embedder_exception_returns_none(self):
        """If embedder throws, classify must return None (not raise)."""
        from app.guardrails.fast_path import SemanticFastPath
        fp = SemanticFastPath(model_name="test-model", threshold=0.82)
        fp._index = np.eye(3, dtype=np.float32)
        class BrokenEmbedder:
            def embed(self, texts):
                raise RuntimeError("embed failed")
        fp._embedder = BrokenEmbedder()
        result = fp.classify("hey")
        assert result is None

    def test_threshold_boundary_exact_match(self):
        """Score exactly at threshold should bypass."""
        fp = self._make_fast_path(threshold=0.82)
        result_hi = fp.classify("hi there")
        score = result_hi.fast_path_score
        fp._threshold = score  # exact boundary
        result = fp.classify("hi there")
        assert result is not None

    def test_threshold_just_below_returns_none(self):
        """Score just below threshold should NOT bypass."""
        fp = self._make_fast_path(threshold=0.82)
        result = fp.classify("hi there")
        score = result.fast_path_score
        fp._threshold = score + 0.01  # just above score
        result2 = fp.classify("hi there")
        assert result2 is None

    def test_build_index_failure_leaves_index_none(self):
        """If fastembed raises during build_index, _index must remain None and classify returns None."""
        from unittest.mock import patch
        from app.guardrails.fast_path import SemanticFastPath
        fp = SemanticFastPath(model_name="bad-model", threshold=0.82)
        # Patch fastembed.TextEmbedding at the import site inside build_index
        with patch("fastembed.TextEmbedding", side_effect=RuntimeError("model not found")):
            fp.build_index()
        assert fp._index is None
        assert fp._embedder is None
        result = fp.classify("hey")
        assert result is None
