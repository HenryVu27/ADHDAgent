"""Tests for model_router — complexity classification and model selection."""

import pytest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.model_router import classify_complexity, create_model_selector


def _make_state(message: str, turn_count: int = 1, extra_messages=None):
    """Build a minimal CoachingState dict for classification."""
    messages = list(extra_messages or [])
    messages.append(HumanMessage(content=message))
    return {
        "messages": messages,
        "session_id": "test",
        "turn_count": turn_count,
        "model_tier": "standard",
        "input_blocked": False,
        "block_response": "",
        "trace_steps": [],
    }


class TestClassifyComplexity:
    """Test the rule-based complexity classifier."""

    # Rule 1: Short trivial patterns -> fast
    def test_yes_is_fast(self):
        state = _make_state("Yes", turn_count=2)
        assert classify_complexity(state) == "fast"

    def test_ok_is_fast(self):
        state = _make_state("ok", turn_count=3)
        assert classify_complexity(state) == "fast"

    def test_thanks_is_fast(self):
        state = _make_state("Thanks!", turn_count=5)
        assert classify_complexity(state) == "fast"

    def test_nope_is_fast(self):
        state = _make_state("Nope", turn_count=2)
        assert classify_complexity(state) == "fast"

    def test_trivial_on_turn_1_is_standard(self):
        """Turn 1 trivial messages should NOT be fast (first message matters)."""
        state = _make_state("Yes", turn_count=1)
        # turn_count must be > 1 for fast
        assert classify_complexity(state) == "standard"

    def test_trivial_but_long_is_standard(self):
        """Message matching trivial pattern but > 20 chars is standard."""
        state = _make_state("Yes I think that sounds good", turn_count=2)
        assert classify_complexity(state) == "standard"

    # Rule 2: Safety keywords -> complex
    def test_medication_name_is_complex(self):
        state = _make_state("Should we try Adderall?", turn_count=2)
        assert classify_complexity(state) == "complex"

    def test_ritalin_is_complex(self):
        state = _make_state("He's on Ritalin currently", turn_count=3)
        assert classify_complexity(state) == "complex"

    def test_diagnosis_keyword_is_complex(self):
        state = _make_state("How do we get an ADHD diagnosis?", turn_count=2)
        assert classify_complexity(state) == "complex"

    def test_harm_keyword_is_complex(self):
        state = _make_state("I'm worried he might hurt himself", turn_count=2)
        assert classify_complexity(state) == "complex"

    def test_medication_generic_is_complex(self):
        state = _make_state("What about medication for focus?", turn_count=2)
        assert classify_complexity(state) == "complex"

    def test_legal_keyword_is_complex(self):
        state = _make_state("What are his IEP rights at school?", turn_count=2)
        assert classify_complexity(state) == "complex"

    # Rule 3: Long conversation -> complex
    def test_turn_11_is_complex(self):
        state = _make_state("How should we adjust the routine?", turn_count=11)
        assert classify_complexity(state) == "complex"

    def test_turn_15_is_complex(self):
        state = _make_state("Any other suggestions?", turn_count=15)
        assert classify_complexity(state) == "complex"

    # Rule 4: Normal -> standard
    def test_normal_parenting_question_is_standard(self):
        state = _make_state("My son has trouble with homework every night", turn_count=2)
        assert classify_complexity(state) == "standard"

    def test_strategy_request_is_standard(self):
        state = _make_state("What strategies can help with bedtime?", turn_count=3)
        assert classify_complexity(state) == "standard"

    def test_outcome_report_is_standard(self):
        state = _make_state("The visual timer worked really well yesterday", turn_count=5)
        assert classify_complexity(state) == "standard"

    # Edge cases
    def test_no_human_message_is_standard(self):
        state = {
            "messages": [AIMessage(content="Hello")],
            "session_id": "test",
            "turn_count": 1,
            "model_tier": "standard",
        }
        assert classify_complexity(state) == "standard"

    def test_empty_messages_is_standard(self):
        state = {
            "messages": [],
            "session_id": "test",
            "turn_count": 0,
            "model_tier": "standard",
        }
        assert classify_complexity(state) == "standard"

    # Priority: safety beats trivial
    def test_safety_overrides_short(self):
        """Even short messages with safety keywords should be complex."""
        state = _make_state("meds?", turn_count=3)
        # "meds" is < 20 chars but contains safety keyword
        assert classify_complexity(state) == "complex"


class TestCreateModelSelector:
    """Test the model selector callable."""

    @patch("app.agent.model_router.settings")
    def test_returns_model_for_standard_tier(self, mock_settings):
        mock_settings.GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
        mock_settings.GEMINI_MODEL_STANDARD = "gemini-2.5-flash"
        mock_settings.GEMINI_MODEL_COMPLEX = "gemini-2.5-pro"
        mock_settings.GEMINI_API_KEY = "fake-key"

        selector = create_model_selector()
        model = selector({"model_tier": "standard"})
        assert "gemini-2.5-flash" in model.model

    @patch("app.agent.model_router.settings")
    def test_returns_model_for_fast_tier(self, mock_settings):
        mock_settings.GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
        mock_settings.GEMINI_MODEL_STANDARD = "gemini-2.5-flash"
        mock_settings.GEMINI_MODEL_COMPLEX = "gemini-2.5-pro"
        mock_settings.GEMINI_API_KEY = "fake-key"

        selector = create_model_selector()
        model = selector({"model_tier": "fast"})
        assert "gemini-2.5-flash-lite" in model.model

    @patch("app.agent.model_router.settings")
    def test_returns_model_for_complex_tier(self, mock_settings):
        mock_settings.GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
        mock_settings.GEMINI_MODEL_STANDARD = "gemini-2.5-flash"
        mock_settings.GEMINI_MODEL_COMPLEX = "gemini-2.5-pro"
        mock_settings.GEMINI_API_KEY = "fake-key"

        selector = create_model_selector()
        model = selector({"model_tier": "complex"})
        assert "gemini-2.5-pro" in model.model

    @patch("app.agent.model_router.settings")
    def test_caches_model_instances(self, mock_settings):
        mock_settings.GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
        mock_settings.GEMINI_MODEL_STANDARD = "gemini-2.5-flash"
        mock_settings.GEMINI_MODEL_COMPLEX = "gemini-2.5-pro"
        mock_settings.GEMINI_API_KEY = "fake-key"

        selector = create_model_selector()
        model1 = selector({"model_tier": "standard"})
        model2 = selector({"model_tier": "standard"})
        assert model1 is model2  # Same cached instance

    @patch("app.agent.model_router.settings")
    def test_defaults_to_standard_on_unknown_tier(self, mock_settings):
        mock_settings.GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
        mock_settings.GEMINI_MODEL_STANDARD = "gemini-2.5-flash"
        mock_settings.GEMINI_MODEL_COMPLEX = "gemini-2.5-pro"
        mock_settings.GEMINI_API_KEY = "fake-key"

        selector = create_model_selector()
        model = selector({"model_tier": "unknown"})
        assert "gemini-2.5-flash" in model.model

    @patch("app.agent.model_router.settings")
    def test_defaults_when_no_tier_in_state(self, mock_settings):
        mock_settings.GEMINI_MODEL_FAST = "gemini-2.5-flash-lite"
        mock_settings.GEMINI_MODEL_STANDARD = "gemini-2.5-flash"
        mock_settings.GEMINI_MODEL_COMPLEX = "gemini-2.5-pro"
        mock_settings.GEMINI_API_KEY = "fake-key"

        selector = create_model_selector()
        model = selector({})  # No model_tier key
        assert "gemini-2.5-flash" in model.model
