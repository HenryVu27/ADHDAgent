# Shared test fixtures

import pytest

from app.models.schemas import (
    ConversationPhase,
    FamilyProfile,
    SessionState,
)
from app.rag.knowledge_store import KnowledgeStore
from app.rag.retriever import HybridRetriever


class MockGeminiClient:
    # Deterministic Gemini client for testing — no API calls

    def __init__(self, generate_response="Mock coaching response.", extract_json_response=None):
        self._generate_response = generate_response
        self._extract_json_response = extract_json_response or []
        self.generate_calls = []
        self.extract_json_calls = []

    async def generate(self, prompt, temperature=0.7):
        self.generate_calls.append(prompt)
        return self._generate_response

    async def extract_json(self, prompt, temperature=0.0):
        self.extract_json_calls.append(prompt)
        return self._extract_json_response

    def embed(self, text):
        return [0.1] * 768

    def embed_batch(self, texts):
        return [[0.1] * 768 for _ in texts]


@pytest.fixture
def mock_gemini():
    return MockGeminiClient()


@pytest.fixture
def knowledge_store():
    return KnowledgeStore()


@pytest.fixture
def retriever(knowledge_store):
    # No Gemini -> keyword fallback
    return HybridRetriever(
        knowledge_store=knowledge_store,
        gemini_client=None,
    )


@pytest.fixture
def clean_session():
    return SessionState(session_id="test_session")


@pytest.fixture
def intake_complete_session():
    return SessionState(
        session_id="test_session",
        phase=ConversationPhase.intake,
        turn_count=5,
        intake_question_index=5,
        family_profile=FamilyProfile(
            child_age="7",
            challenge_areas=["homework", "emotion"],
            attempted_strategies=["timeout", "rewards"],
            hardest_situations=["homework", "bedtime"],
        ),
    )


@pytest.fixture
def strategy_session():
    return SessionState(
        session_id="test_session",
        phase=ConversationPhase.strategy,
        turn_count=6,
        family_profile=FamilyProfile(
            child_age="7",
            challenge_areas=["homework"],
            attempted_strategies=["timeout"],
        ),
        active_strategies=["timer_technique"],
    )
