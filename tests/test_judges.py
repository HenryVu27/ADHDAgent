import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from eval.judges.base import JudgeBase


class ConcreteJudge(JudgeBase):
    """Minimal concrete subclass for testing the base class."""
    pass


@pytest.fixture
def mock_gen_client():
    client = MagicMock()
    client.json = AsyncMock(return_value={"score": 5})
    return client


class TestJudgeBase:

    def test_init_default_concurrency(self):
        judge = ConcreteJudge()
        assert judge._semaphore._value == 5

    def test_init_custom_concurrency(self):
        judge = ConcreteJudge(max_concurrency=10)
        assert judge._semaphore._value == 10

    @pytest.mark.asyncio
    async def test_judge_json_calls_gen_client(self, mock_gen_client):
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ConcreteJudge()
        result = await judge.judge_json("test prompt")
        assert result == {"score": 5}
        mock_gen_client.json.assert_called_once()

    @pytest.mark.asyncio
    async def test_judge_json_respects_semaphore(self, mock_gen_client):
        """Verify semaphore limits concurrency."""
        call_count = 0
        max_concurrent = 0

        async def slow_json(*args, **kwargs):
            nonlocal call_count, max_concurrent
            call_count += 1
            current = call_count
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            call_count -= 1
            return {"score": 1}

        mock_gen_client.json = slow_json

        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ConcreteJudge(max_concurrency=2)

        tasks = [judge.judge_json("prompt") for _ in range(5)]
        await asyncio.gather(*tasks)
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_judge_json_returns_empty_on_failure(self, mock_gen_client):
        mock_gen_client.json = AsyncMock(side_effect=Exception("API error"))
        with patch("eval.judges.base.GenClient", return_value=mock_gen_client):
            judge = ConcreteJudge()
        result = await judge.judge_json("test prompt")
        assert result == {}
