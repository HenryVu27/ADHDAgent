"""Tests for predicate extraction (Layer 1)."""

import pytest

from app.predicates.extractor import PredicateExtractor


@pytest.fixture
def extractor():
    """Extractor with no Gemini client — uses keyword fallback."""
    return PredicateExtractor(gemini_client=None)


@pytest.mark.asyncio
async def test_extract_behavior_keywords(extractor):
    result = await extractor.extract("My son won't do his homework and keeps getting distracted")
    subjects = [p.subject for p in result.predicates]
    assert "avoidance" in subjects
    assert "distraction" in subjects
    assert "academic_concern" in subjects
    assert result.method.value == "keyword_fallback"


@pytest.mark.asyncio
async def test_extract_emotional_keywords(extractor):
    result = await extractor.extract("He had a total meltdown and was so angry")
    subjects = [p.subject for p in result.predicates]
    assert "emotional_dysregulation" in subjects


@pytest.mark.asyncio
async def test_extract_age(extractor):
    result = await extractor.extract("My 7 year old daughter was diagnosed last year")
    age_preds = [p for p in result.predicates if p.predicate == "child_age"]
    assert len(age_preds) == 1
    assert age_preds[0].subject == "7"


@pytest.mark.asyncio
async def test_extract_age_hyphenated(extractor):
    result = await extractor.extract("My 9-year-old keeps getting in trouble at school")
    age_preds = [p for p in result.predicates if p.predicate == "child_age"]
    assert len(age_preds) == 1
    assert age_preds[0].subject == "9"


@pytest.mark.asyncio
async def test_extract_parent_concern(extractor):
    result = await extractor.extract("I'm so frustrated and exhausted, I've tried everything")
    subjects = [p.subject for p in result.predicates]
    assert "parent_frustration" in subjects
    assert "parent_burnout" in subjects
    assert "exhausted_options" in subjects


@pytest.mark.asyncio
async def test_empty_message(extractor):
    result = await extractor.extract("")
    assert result.predicates == []


@pytest.mark.asyncio
async def test_greeting_no_predicates(extractor):
    result = await extractor.extract("Hi there, nice to meet you!")
    # Greetings may extract "seeking_help" from "help", but shouldn't crash
    assert isinstance(result.predicates, list)


@pytest.mark.asyncio
async def test_extraction_result_has_raw_text(extractor):
    msg = "My kid has trouble with transitions"
    result = await extractor.extract(msg)
    assert result.raw_text == msg


@pytest.mark.asyncio
async def test_predicates_have_confidence(extractor):
    result = await extractor.extract("My son won't do homework")
    for p in result.predicates:
        assert 0.0 <= p.confidence <= 1.0


@pytest.mark.asyncio
async def test_gemini_extraction_with_mock(mock_gemini_with_predicates):
    """Test that Gemini extraction works with a mock client."""
    extractor = PredicateExtractor(gemini_client=mock_gemini_with_predicates)
    result = await extractor.extract("My 7 year old won't do homework")
    assert result.method.value == "gemini"
    assert len(result.predicates) == 2
    assert result.predicates[0].subject == "avoidance"


# --- Conversation context tests ---

@pytest.mark.asyncio
async def test_gemini_extraction_receives_conversation_context(mock_gemini_with_predicates):
    """Extraction prompt should include conversation context when provided."""
    extractor = PredicateExtractor(gemini_client=mock_gemini_with_predicates)
    context = "Parent: My son Jamie is 7.\nCoach: Thank you for sharing."
    result = await extractor.extract("He won't do his homework", conversation_context=context)
    prompt = mock_gemini_with_predicates.extract_json_calls[-1]
    assert "Jamie" in prompt


@pytest.mark.asyncio
async def test_extraction_without_context_still_works(mock_gemini_with_predicates):
    """Extraction should work without conversation context (backward-compatible)."""
    extractor = PredicateExtractor(gemini_client=mock_gemini_with_predicates)
    result = await extractor.extract("My 7 year old won't do homework")
    assert result.method.value == "gemini"
    assert len(result.predicates) == 2


@pytest.mark.asyncio
async def test_keyword_extraction_ignores_context(extractor):
    """Keyword fallback doesn't use context, but shouldn't break with it."""
    result = await extractor.extract(
        "My son won't do homework",
        conversation_context="Parent: He's 7.\nCoach: Got it.",
    )
    assert result.method.value == "keyword_fallback"
    subjects = [p.subject for p in result.predicates]
    assert "avoidance" in subjects
