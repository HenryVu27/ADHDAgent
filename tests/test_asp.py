from app.asp.engine import ASPEngine


def test_initial_state_is_intake():
    engine = ASPEngine()
    directives = engine.reason(predicates=[], session_id="test1")
    assert "gather_info" in directives


def test_safety_override():
    engine = ASPEngine()
    predicates = [{"predicate": "crisis", "subject": "harm", "category": "safety"}]
    directives = engine.reason(predicates=predicates, session_id="test2")
    assert "safety_redirect" in directives


def test_session_isolation():
    engine = ASPEngine()
    engine.reason(predicates=[], session_id="session_a")
    engine.reason(predicates=[], session_id="session_b")

    facts_a = engine.get_facts("session_a")
    facts_b = engine.get_facts("session_b")

    assert facts_a is not facts_b


def test_add_fact():
    engine = ASPEngine()
    engine.add_fact("test3", "context(child_age)")
    facts = engine.get_facts("test3")
    assert "context(child_age)" in facts
