from app.predicates.extractor import PredicateExtractor


def test_extract_behavior_keywords():
    extractor = PredicateExtractor(use_llm=False)
    predicates = extractor.extract("My son won't do his homework and keeps getting distracted")

    subjects = [p["subject"] for p in predicates]
    assert "avoidance" in subjects
    assert "distraction" in subjects
    assert "academic_concern" in subjects


def test_extract_emotional_keywords():
    extractor = PredicateExtractor(use_llm=False)
    predicates = extractor.extract("He had a total meltdown and was so angry")

    subjects = [p["subject"] for p in predicates]
    assert "emotional_dysregulation" in subjects


def test_extract_age():
    extractor = PredicateExtractor(use_llm=False)
    predicates = extractor.extract("My 7 year old daughter was diagnosed last year")

    age_preds = [p for p in predicates if p["predicate"] == "child_age"]
    assert len(age_preds) == 1
    assert age_preds[0]["subject"] == "7"


def test_extract_parent_concern():
    extractor = PredicateExtractor(use_llm=False)
    predicates = extractor.extract("I'm so frustrated and exhausted, I've tried everything")

    subjects = [p["subject"] for p in predicates]
    assert "parent_frustration" in subjects
    assert "parent_burnout" in subjects
    assert "exhausted_options" in subjects


def test_empty_message():
    extractor = PredicateExtractor(use_llm=False)
    predicates = extractor.extract("")
    assert predicates == []
