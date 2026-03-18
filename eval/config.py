"""Standalone eval config — no imports from app/."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GENERATOR_MODEL: str = "gemini-2.5-flash"   # cheap model for dataset generation
JUDGE_MODEL: str = "gemini-2.5-flash"       # quality filtering judge
EVAL_JUDGE_MODEL: str = "gemini-2.5-pro"  # stronger model for response quality judging
EMBEDDING_MODEL: str = "gemini-embedding-001"

KNOWLEDGE_DIR = Path(__file__).parent.parent / "app" / "knowledge"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "results"

GOLDEN_RETRIEVAL_PATH = DATA_DIR / "golden_retrieval.json"
GOLDEN_MEMORY_PATH = DATA_DIR / "golden_memory.json"
GOLDEN_INPUT_GATE_PATH = DATA_DIR / "golden_input_gate.json"

# Quality filtering thresholds
ANSWERABILITY_MIN_SCORE: int = 2        # 1-3 scale; discard if below this
DIVERSITY_SIM_THRESHOLD: float = 0.85  # discard if cosine sim to any kept question exceeds this
DIFFICULTY_EASY_THRESHOLD: float = 0.92 # flag (not discard) if question-chunk sim exceeds this

# Target question type distribution across 60 docs
QUESTIONS_PER_STATEMENT: int = 3        # three questions per statement for higher volume
STATEMENTS_PER_CHUNK: int = 10          # extract this many statements per chunk
QUESTION_TYPE_WEIGHTS: dict[str, float] = {
    "fact_single": 0.30,
    "reasoning": 0.25,
    "multi_context": 0.18,
    "out_of_scope": 0.10,
    "procedure": 0.12,
    "comparative": 0.05,
}

# Conversations to generate per persona
CONVERSATIONS_PER_PERSONA: int = 8
MAX_CRITIC_RETRIES: int = 3
