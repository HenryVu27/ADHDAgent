"""Tests for guardrail gate classifiers."""

import pytest
from app.models.schemas import (
    InputCheckResult,
    InputClassification,
    OutputCheckResult,
    OutputClassification,
    GuardrailsError,
)


class TestSchemas:
    def test_input_classification_defaults(self):
        c = InputClassification()
        assert c.crisis is False
        assert c.jailbreak is False
        assert c.reasoning == ""

    def test_input_classification_crisis(self):
        c = InputClassification(crisis=True, reasoning="mentions self-harm")
        assert c.crisis is True

    def test_output_classification_defaults(self):
        c = OutputClassification()
        assert c.medication_recommendation is False
        assert c.diagnosis_claim is False
        assert c.scope_violation is False

    def test_output_classification_medication(self):
        c = OutputClassification(medication_recommendation=True, reasoning="recommends Adderall")
        assert c.medication_recommendation is True

    def test_input_check_result_unchanged(self):
        r = InputCheckResult(is_allowed=True)
        assert r.is_allowed is True
        assert r.blocked_reason is None

    def test_output_check_result_unchanged(self):
        r = OutputCheckResult(is_valid=True)
        assert r.is_valid is True
        assert r.violation_type is None
