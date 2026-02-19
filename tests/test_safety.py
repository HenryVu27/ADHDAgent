"""Tests for safety monitor."""

import pytest
from unittest.mock import AsyncMock

from app.agents.safety import SafetyMonitor
from app.models.schemas import SafetyLevel


def _mock_gemini(level: str = "safe", topic: str = None):
    """Create a mock Gemini client that returns the given safety level."""
    client = AsyncMock()
    client.extract_json = AsyncMock(return_value={"level": level, "detected_topic": topic})
    return client


@pytest.fixture
def safe_gemini():
    return _mock_gemini("safe")


@pytest.fixture
def crisis_gemini():
    return _mock_gemini("crisis", "self-harm")


@pytest.fixture
def out_of_scope_gemini():
    return _mock_gemini("out_of_scope", "medication")


# --- No Gemini fallback ---

@pytest.mark.asyncio
async def test_no_gemini_defaults_to_safe():
    """Without Gemini client, all messages pass as safe."""
    safety = SafetyMonitor(gemini_client=None)
    result = await safety.check("Should I try medication?")
    assert result.level == SafetyLevel.safe


# --- Safe messages ---

@pytest.mark.asyncio
async def test_safe_message(safe_gemini):
    safety = SafetyMonitor(gemini_client=safe_gemini)
    result = await safety.check("My son has trouble with homework")
    assert result.level == SafetyLevel.safe
    assert result.response_override is None


@pytest.mark.asyncio
async def test_safe_behavioral_question(safe_gemini):
    safety = SafetyMonitor(gemini_client=safe_gemini)
    result = await safety.check("What strategies work for bedtime routines?")
    assert result.level == SafetyLevel.safe


@pytest.mark.asyncio
async def test_diagnosed_as_context_is_safe(safe_gemini):
    """The exact bug: 'diagnosed' used as biographical context should be safe."""
    safety = SafetyMonitor(gemini_client=safe_gemini)
    result = await safety.check("my child is 12 years old, he was diagnosed 2 years ago")
    assert result.level == SafetyLevel.safe


# --- Crisis detection ---

@pytest.mark.asyncio
async def test_crisis_detection(crisis_gemini):
    safety = SafetyMonitor(gemini_client=crisis_gemini)
    result = await safety.check("I'm worried my child might harm themselves")
    assert result.level == SafetyLevel.crisis
    assert "988" in result.response_override


@pytest.mark.asyncio
async def test_crisis_suicide(crisis_gemini):
    safety = SafetyMonitor(gemini_client=crisis_gemini)
    result = await safety.check("My child mentioned suicide")
    assert result.level == SafetyLevel.crisis


@pytest.mark.asyncio
async def test_crisis_response_has_resources(crisis_gemini):
    safety = SafetyMonitor(gemini_client=crisis_gemini)
    result = await safety.check("I'm afraid I might harm my child")
    assert "911" in result.response_override
    assert "988" in result.response_override
    assert "741741" in result.response_override


# --- Out of scope ---

@pytest.mark.asyncio
async def test_out_of_scope_medication(out_of_scope_gemini):
    safety = SafetyMonitor(gemini_client=out_of_scope_gemini)
    result = await safety.check("Should I try medication for my child?")
    assert result.level == SafetyLevel.out_of_scope
    assert "medication" in result.detected_topic


@pytest.mark.asyncio
async def test_out_of_scope_diagnosis():
    gemini = _mock_gemini("out_of_scope", "diagnosis")
    safety = SafetyMonitor(gemini_client=gemini)
    result = await safety.check("Can you diagnose my child with ADHD?")
    assert result.level == SafetyLevel.out_of_scope


@pytest.mark.asyncio
async def test_out_of_scope_legal():
    gemini = _mock_gemini("out_of_scope", "custody")
    safety = SafetyMonitor(gemini_client=gemini)
    result = await safety.check("I need advice about custody arrangements")
    assert result.level == SafetyLevel.out_of_scope


@pytest.mark.asyncio
async def test_out_of_scope_specific_medication():
    gemini = _mock_gemini("out_of_scope", "ritalin")
    safety = SafetyMonitor(gemini_client=gemini)
    result = await safety.check("What do you think about Ritalin?")
    assert result.level == SafetyLevel.out_of_scope


# --- Gemini failure fallback ---

@pytest.mark.asyncio
async def test_gemini_failure_defaults_to_safe():
    """If Gemini call fails, default to safe rather than blocking."""
    gemini = AsyncMock()
    gemini.extract_json = AsyncMock(side_effect=Exception("API error"))
    safety = SafetyMonitor(gemini_client=gemini)
    result = await safety.check("My child was diagnosed last year")
    assert result.level == SafetyLevel.safe


# --- Conversation context tests ---

@pytest.mark.asyncio
async def test_safety_check_receives_conversation_context():
    """Safety prompt should include conversation context when provided."""
    gemini = _mock_gemini("safe")
    safety = SafetyMonitor(gemini_client=gemini)
    result = await safety.check(
        "It's getting worse every day",
        conversation_context="Parent: Things have been tough.\nCoach: I hear you.",
    )
    prompt = gemini.extract_json.call_args[0][0]
    assert "Things have been tough" in prompt


@pytest.mark.asyncio
async def test_safety_check_without_context_still_works():
    """Safety check should work without conversation context (backward-compatible)."""
    gemini = _mock_gemini("safe")
    safety = SafetyMonitor(gemini_client=gemini)
    result = await safety.check("My son has trouble focusing")
    assert result.level == SafetyLevel.safe
