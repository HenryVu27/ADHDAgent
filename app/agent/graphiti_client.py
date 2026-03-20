"""Graphiti client factory and custom entity/edge type definitions.

Creates a configured Graphiti instance with Gemini LLM and embedder,
custom ADHD coaching domain types, and Neo4j connection.
"""

import logging

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)


# --- Custom Entity Types ---

class Child(BaseModel):
    """A child being discussed in coaching sessions."""
    name: str = ""
    age: str = ""
    diagnosis_status: str = ""
    adhd_subtype: str = ""


class Parent(BaseModel):
    """The parent participating in coaching."""
    name: str = ""


class Strategy(BaseModel):
    """An ADHD management strategy mentioned or tried."""
    name: str = ""
    source: str = ""
    evidence_level: str = ""


class Challenge(BaseModel):
    """A specific difficulty area the family faces."""
    name: str = ""
    severity: str = ""


class EmotionalState(BaseModel):
    """A detected emotional state of the parent."""
    emotion: str = ""
    intensity: str = ""


class Goal(BaseModel):
    """A goal the parent has set."""
    description: str = ""
    status: str = ""


# --- Custom Edge Types ---

class TriedStrategy(BaseModel):
    """Parent tried a strategy with an outcome."""
    outcome: str = ""
    notes: str = ""


class HasChallenge(BaseModel):
    """Child has a specific challenge area."""
    context: str = ""


class ExperiencedEmotion(BaseModel):
    """Parent experienced an emotional state."""
    trigger: str = ""


class AddressesChallenge(BaseModel):
    """Strategy addresses a specific challenge."""
    effectiveness: str = ""


class SetGoal(BaseModel):
    """Parent set a specific goal."""
    motivation: str = ""


# --- Dict containers for add_episode() ---

ADHD_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Child": Child,
    "Parent": Parent,
    "Strategy": Strategy,
    "Challenge": Challenge,
    "EmotionalState": EmotionalState,
    "Goal": Goal,
}

ADHD_EDGE_TYPES: dict[str, type[BaseModel]] = {
    "TriedStrategy": TriedStrategy,
    "HasChallenge": HasChallenge,
    "ExperiencedEmotion": ExperiencedEmotion,
    "AddressesChallenge": AddressesChallenge,
    "SetGoal": SetGoal,
}


async def create_graphiti_client():
    """Create and initialize a Graphiti client with Gemini and Neo4j.

    Returns None if GRAPHITI_ENABLED is False or Neo4j credentials are missing.
    Falls back to None on connection failure (logged as error).
    """
    if not settings.GRAPHITI_ENABLED or not settings.NEO4J_PASSWORD:
        logger.info("Graphiti disabled (enabled=%s, has_password=%s)",
                     settings.GRAPHITI_ENABLED, bool(settings.NEO4J_PASSWORD))
        return None

    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.gemini_client import GeminiClient
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig

        client = Graphiti(
            settings.NEO4J_URI,
            settings.NEO4J_USER,
            settings.NEO4J_PASSWORD,
            llm_client=GeminiClient(
                config=LLMConfig(
                    api_key=settings.GEMINI_API_KEY,
                    model=settings.GRAPHITI_LLM_MODEL,
                )
            ),
            embedder=GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=settings.GEMINI_API_KEY,
                    embedding_model=settings.GRAPHITI_EMBEDDING_MODEL,
                )
            ),
        )
        await client.build_indices_and_constraints()
        logger.info(
            "Graphiti client initialized (neo4j=%s, llm=%s, embedder=%s)",
            settings.NEO4J_URI, settings.GRAPHITI_LLM_MODEL, settings.GRAPHITI_EMBEDDING_MODEL,
        )
        return client

    except Exception as e:
        logger.error("Graphiti initialization failed — falling back to SQLite memory: %s", e)
        return None
