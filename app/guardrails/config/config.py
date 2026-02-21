"""
Custom NeMo Guardrails actions for multi-rail input + output checking.

Each action uses llm_task_manager to render the prompt template from
config.yml, then calls the LLM to classify. Both llm_task_manager and
llm are injected by NeMo's runtime via named parameter matching.
"""

import logging

from langchain_core.language_models.llms import BaseLLM

logger = logging.getLogger(__name__)


async def _run_check(
    task_name: str,
    context_key: str,
    context_value: str,
    llm_task_manager=None,
    llm: BaseLLM | None = None,
) -> str:
    """Shared logic for all rail checks.

    1. Render the prompt template via llm_task_manager
    2. Call the LLM with the rendered prompt
    3. Parse yes/no answer
    """
    if llm_task_manager is None or llm is None:
        logger.warning("Missing llm_task_manager or llm — cannot run %s", task_name)
        return "no"

    try:
        logger.debug("NeMo rail [%s] input %s=%r", task_name, context_key, context_value)
        prompt = llm_task_manager.render_task_prompt(
            task=task_name,
            context={context_key: context_value},
        )
        logger.debug("NeMo rail [%s] rendered prompt: %s", task_name, prompt[:200])
        result = await llm.ainvoke(prompt)
        answer = result.strip().lower() if isinstance(result, str) else str(result).strip().lower()
        logger.debug("NeMo rail [%s] LLM answer: %r", task_name, answer)
        if answer.startswith("yes"):
            logger.info("NeMo rail [%s] triggered", task_name)
            return "yes"
        return "no"
    except Exception as e:
        logger.error("NeMo rail [%s] failed: %s", task_name, e)
        return "no"


# ---- Input Rail Actions ----

async def self_check_language(user_input: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_language", "user_input", user_input, llm_task_manager, llm)


async def self_check_jailbreak(user_input: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_jailbreak", "user_input", user_input, llm_task_manager, llm)


async def self_check_crisis(user_input: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_crisis", "user_input", user_input, llm_task_manager, llm)


async def self_check_out_of_scope(user_input: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_out_of_scope", "user_input", user_input, llm_task_manager, llm)


async def self_check_content_safety(user_input: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_content_safety", "user_input", user_input, llm_task_manager, llm)


async def self_check_topic_boundaries(user_input: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_topic_boundaries", "user_input", user_input, llm_task_manager, llm)


# ---- Output Rail Actions ----

async def self_check_medication(bot_response: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_medication", "bot_response", bot_response, llm_task_manager, llm)


async def self_check_diagnosis(bot_response: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_diagnosis", "bot_response", bot_response, llm_task_manager, llm)


async def self_check_scope_violation(bot_response: str = "", llm_task_manager=None, llm=None) -> str:
    return await _run_check("self_check_scope_violation", "bot_response", bot_response, llm_task_manager, llm)


def init(app):
    """Register all actions with NeMo Guardrails."""
    app.register_action(self_check_language, "self_check_language")
    app.register_action(self_check_jailbreak, "self_check_jailbreak")
    app.register_action(self_check_crisis, "self_check_crisis")
    app.register_action(self_check_out_of_scope, "self_check_out_of_scope")
    app.register_action(self_check_content_safety, "self_check_content_safety")
    app.register_action(self_check_topic_boundaries, "self_check_topic_boundaries")
    app.register_action(self_check_medication, "self_check_medication")
    app.register_action(self_check_diagnosis, "self_check_diagnosis")
    app.register_action(self_check_scope_violation, "self_check_scope_violation")
