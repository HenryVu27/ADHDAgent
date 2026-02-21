"""
LangChain BaseLLM adapter for our GeminiClient.

NeMo Guardrails expects a LangChain-compatible LLM. This wraps
our existing GeminiClient so NeMo can use Gemini as its engine
without introducing a separate connection or API key.
"""

import asyncio
import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult

logger = logging.getLogger(__name__)


class GeminiLangChainLLM(BaseLLM):
    """LangChain wrapper around our GeminiClient for NeMo Guardrails."""

    client: Any = None  # The GeminiClient instance

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "gemini-langchain-adapter"

    def _generate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Synchronous generation — runs async method in event loop."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(
            self._agenerate(prompts, stop=stop, run_manager=None, **kwargs)
        )

    async def _agenerate(
        self,
        prompts: list[str],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Async generation using our GeminiClient."""
        generations = []
        for prompt in prompts:
            try:
                text = await self.client.generate(prompt, temperature=0.0)
                if stop:
                    for s in stop:
                        idx = text.find(s)
                        if idx != -1:
                            text = text[:idx]
                generations.append([Generation(text=text)])
            except Exception as e:
                logger.error(f"Gemini LangChain adapter error: {e}")
                generations.append([Generation(text="")])
        return LLMResult(generations=generations)


def make_gemini_llm(gemini_client) -> GeminiLangChainLLM:
    """Create a LangChain-compatible LLM from our GeminiClient."""
    return GeminiLangChainLLM(client=gemini_client)
