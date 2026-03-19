"""Record live agent traces into conversation JSON files for eval.

Usage:
    python -m eval.runners.recorder --input eval/data/scripts/homework_scenario.json --output eval/data/conversations/
    python -m eval.runners.recorder --input eval/data/scripts/homework_scenario.json --output eval/data/conversations/ --variant-name "v2_prompt"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def record_conversation(
    messages: list[str],
    conversation_id: str,
    output_dir: Path,
    variant_name: str = "default",
    persona_id: str = "",
) -> Path:
    """Run a list of user messages through the live agent and save the trace."""
    # Import here to avoid loading app at module level
    from app.main import build_dependencies

    deps = await build_dependencies()
    orchestrator = deps["orchestrator"]
    session_store = deps["session_store"]

    session_id = f"rec_{conversation_id}_{int(time.time())}"

    turns = []
    for i, user_msg in enumerate(messages, 1):
        logger.info("[%d/%d] Sending: %s", i, len(messages), user_msg[:80])

        result = await orchestrator.process(message=user_msg, session_id=session_id)

        # Get the trace for this turn
        traces = await session_store.get_traces(session_id)
        trace = traces[-1] if traces else None

        tool_calls = []
        if trace and trace.tool_calls:
            for tc in trace.tool_calls:
                tool_calls.append({
                    "name": tc.name,
                    "args": tc.args,
                    "result": tc.result[:1000] if tc.result else "",
                })

        response_text = result.response or ""
        if not response_text:
            stored_messages = await session_store.get_messages(session_id)
            assistant_msgs = [m for m in stored_messages if m.get("role") == "assistant"]
            response_text = assistant_msgs[-1]["content"] if assistant_msgs else ""

        turns.append({
            "turn": i,
            "user_message": user_msg,
            "assistant_response": response_text,
            "tool_calls": tool_calls,
            "trace": trace.model_dump() if trace else {},
            "input_blocked": trace.input_blocked if trace else False,
        })

    conversation = {
        "conversation_id": conversation_id,
        "source": "recorded",
        "persona_id": persona_id,
        "variant": variant_name,
        "turns": turns,
        "metadata": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_variant": variant_name,
            "model": settings.GEMINI_AGENT_MODEL,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{conversation_id}.json"
    out_path.write_text(json.dumps(conversation, indent=2))
    logger.info("Saved conversation to %s", out_path)

    # Clean up
    if deps.get("db_conn"):
        await deps["db_conn"].close()

    return out_path


async def run(input_path: Path, output_dir: Path, variant_name: str) -> None:
    """Load a script file and record conversations."""
    with open(input_path) as f:
        script = json.load(f)

    # Script format: {"conversations": [{"id": "...", "persona_id": "...", "messages": ["...", ...]}]}
    for conv_script in script.get("conversations", []):
        await record_conversation(
            messages=conv_script["messages"],
            conversation_id=conv_script["id"],
            output_dir=output_dir,
            variant_name=variant_name,
            persona_id=conv_script.get("persona_id", ""),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record agent conversations for eval")
    parser.add_argument("--input", type=Path, required=True, help="Script JSON file with user messages")
    parser.add_argument("--output", type=Path, default=Path("eval/data/conversations"), help="Output directory")
    parser.add_argument("--variant-name", default="default", help="Pipeline variant name")
    args = parser.parse_args()
    asyncio.run(run(input_path=args.input, output_dir=args.output, variant_name=args.variant_name))
