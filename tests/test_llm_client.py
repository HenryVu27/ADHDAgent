"""Tests for GeminiClient -- retry logic and max_output_tokens parameter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.client import GeminiClient


class TestGenerateMaxOutputTokens:
    """Verify max_output_tokens is passed through to the API config."""

    @patch("app.llm.client.genai")
    def test_generate_uses_custom_max_tokens(self, mock_genai):
        """generate() should pass max_output_tokens to GenerateContentConfig."""
        mock_response = MagicMock()
        mock_response.text = "response text"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        import asyncio
        client = GeminiClient()
        result = asyncio.get_event_loop().run_until_complete(
            client.generate("test prompt", max_output_tokens=256)
        )

        # Verify the config passed to generate_content
        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.max_output_tokens == 256

    @patch("app.llm.client.genai")
    def test_generate_uses_default_max_tokens(self, mock_genai):
        """generate() without explicit max_output_tokens uses default (2048)."""
        mock_response = MagicMock()
        mock_response.text = "response text"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        import asyncio
        client = GeminiClient()
        result = asyncio.get_event_loop().run_until_complete(
            client.generate("test prompt")
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.max_output_tokens == 2048


class TestExtractJsonMaxOutputTokens:

    @patch("app.llm.client.genai")
    def test_extract_json_uses_custom_max_tokens(self, mock_genai):
        mock_response = MagicMock()
        mock_response.text = '{"key": "value"}'
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        import asyncio
        client = GeminiClient()
        result = asyncio.get_event_loop().run_until_complete(
            client.extract_json("test prompt", max_output_tokens=512)
        )

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.max_output_tokens == 512


class TestRetryBehavior:
    """Verify tenacity retry on transient errors."""

    @patch("app.llm.client.genai")
    def test_generate_retries_on_connection_error(self, mock_genai):
        """generate() should retry on ConnectionError then succeed."""
        mock_response = MagicMock()
        mock_response.text = "success after retry"
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            ConnectionError("network blip"),
            mock_response,
        ]
        mock_genai.Client.return_value = mock_client

        import asyncio
        client = GeminiClient()
        result = asyncio.get_event_loop().run_until_complete(
            client.generate("test prompt")
        )
        assert result == "success after retry"
        assert mock_client.models.generate_content.call_count == 2

    @patch("app.llm.client.genai")
    def test_generate_does_not_retry_on_value_error(self, mock_genai):
        """generate() should NOT retry on ValueError (client error)."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ValueError("bad request")
        mock_genai.Client.return_value = mock_client

        import asyncio
        client = GeminiClient()
        with pytest.raises(ValueError, match="bad request"):
            asyncio.get_event_loop().run_until_complete(
                client.generate("test prompt")
            )
        assert mock_client.models.generate_content.call_count == 1
