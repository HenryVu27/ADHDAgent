"""Tests for GeminiClient.search_web() -- grounded web search via GoogleSearch tool."""

import asyncio

import pytest
from unittest.mock import MagicMock, patch

from app.llm.client import GeminiClient


def _make_client(mock_genai_client):
    """Create a GeminiClient with __init__ bypassed, wiring mock internals."""
    with patch.object(GeminiClient, "__init__", lambda self: None):
        client = GeminiClient()
    client._client = mock_genai_client
    client._model = "gemini-2.5-flash"
    return client


def _make_grounding_chunk(title: str, uri: str):
    chunk = MagicMock()
    chunk.web = MagicMock()
    chunk.web.title = title
    chunk.web.uri = uri
    return chunk


class TestSearchWebReturnsAnswerAndSources:
    """search_web() returns (answer_text, sources) when grounding metadata is present."""

    @patch("app.llm.client.genai")
    def test_returns_answer_and_sources(self, mock_genai):
        chunk1 = _make_grounding_chunk("CHADD Article", "https://chadd.org/article")
        chunk2 = _make_grounding_chunk("ADDitude Mag", "https://additudemag.com/tips")

        grounding_metadata = MagicMock()
        grounding_metadata.grounding_chunks = [chunk1, chunk2]

        candidate = MagicMock()
        candidate.grounding_metadata = grounding_metadata

        mock_response = MagicMock()
        mock_response.text = "ADHD strategies for kids include visual schedules."
        mock_response.candidates = [candidate]

        mock_api_client = MagicMock()
        mock_api_client.models.generate_content.return_value = mock_response

        client = _make_client(mock_api_client)
        answer, sources = asyncio.get_event_loop().run_until_complete(
            client.search_web("ADHD strategies for kids")
        )

        assert answer == "ADHD strategies for kids include visual schedules."
        assert len(sources) == 2
        assert sources[0] == {"title": "CHADD Article", "url": "https://chadd.org/article"}
        assert sources[1] == {"title": "ADDitude Mag", "url": "https://additudemag.com/tips"}


class TestSearchWebEmptySources:
    """search_web() returns empty sources list when grounding_metadata is None."""

    @patch("app.llm.client.genai")
    def test_returns_empty_sources_when_no_grounding(self, mock_genai):
        candidate = MagicMock()
        candidate.grounding_metadata = None

        mock_response = MagicMock()
        mock_response.text = "Some answer without sources."
        mock_response.candidates = [candidate]

        mock_api_client = MagicMock()
        mock_api_client.models.generate_content.return_value = mock_response

        client = _make_client(mock_api_client)
        answer, sources = asyncio.get_event_loop().run_until_complete(
            client.search_web("any query")
        )

        assert answer == "Some answer without sources."
        assert sources == []


class TestSearchWebNoText:
    """search_web() returns ("", []) when response has no text and no candidates."""

    @patch("app.llm.client.genai")
    def test_returns_empty_on_no_text(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = []

        mock_api_client = MagicMock()
        mock_api_client.models.generate_content.return_value = mock_response

        client = _make_client(mock_api_client)
        answer, sources = asyncio.get_event_loop().run_until_complete(
            client.search_web("empty query")
        )

        assert answer == ""
        assert sources == []


class TestSearchWebRaisesOnError:
    """search_web() propagates exceptions from the underlying API call."""

    @patch("app.llm.client.genai")
    def test_raises_on_api_error(self, mock_genai):
        mock_api_client = MagicMock()
        mock_api_client.models.generate_content.side_effect = ConnectionError("network down")

        client = _make_client(mock_api_client)
        with pytest.raises(ConnectionError, match="network down"):
            asyncio.get_event_loop().run_until_complete(
                client.search_web("some query")
            )
