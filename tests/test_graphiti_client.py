"""Tests for the Graphiti client factory and custom types."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.graphiti_client import (
    create_graphiti_client,
    ADHD_ENTITY_TYPES,
    ADHD_EDGE_TYPES,
)


@pytest.mark.asyncio
async def test_create_graphiti_client_returns_none_when_disabled():
    """When GRAPHITI_ENABLED=False, factory returns None."""
    with patch("app.agent.graphiti_client.settings") as mock_settings:
        mock_settings.GRAPHITI_ENABLED = False
        mock_settings.NEO4J_PASSWORD = "test"
        result = await create_graphiti_client()
        assert result is None


@pytest.mark.asyncio
async def test_create_graphiti_client_returns_none_when_no_password():
    """When NEO4J_PASSWORD is empty, factory returns None."""
    with patch("app.agent.graphiti_client.settings") as mock_settings:
        mock_settings.GRAPHITI_ENABLED = True
        mock_settings.NEO4J_PASSWORD = ""
        result = await create_graphiti_client()
        assert result is None


def test_entity_types_defined():
    """Custom entity types are Pydantic models with expected fields."""
    assert "Child" in ADHD_ENTITY_TYPES
    assert "Strategy" in ADHD_ENTITY_TYPES
    assert "EmotionalState" in ADHD_ENTITY_TYPES


def test_edge_types_defined():
    """Custom edge types are Pydantic models with expected fields."""
    assert "TriedStrategy" in ADHD_EDGE_TYPES
    assert "HasChallenge" in ADHD_EDGE_TYPES
    assert "ExperiencedEmotion" in ADHD_EDGE_TYPES
