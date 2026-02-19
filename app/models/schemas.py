"""
Data contracts for the ADHDAgent pipeline.

All Pydantic models are defined here first — contracts before implementation.
Every component in the pipeline consumes and produces these typed models.
"""

from enum import Enum
from pydantic import BaseModel, Field


# --- Enums ---

class ConversationPhase(str, Enum):
    intake = "intake"
    strategy = "strategy"
    progress = "progress"
    followup = "followup"


# --- Guardrails ---

class InputCheckResult(BaseModel):
    is_allowed: bool
    blocked_reason: str | None = None  # "jailbreak" | "crisis" | "out_of_scope" | "content" | "off_topic"
    override_response: str | None = None  # Pre-built response for crisis/OOS
    duration_ms: float = 0.0


class OutputCheckResult(BaseModel):
    is_valid: bool
    violation_type: str | None = None  # "medication" | "diagnosis" | "scope"
    duration_ms: float = 0.0


class GuardrailsError(Exception):
    """Raised when NeMo Guardrails fails — no silent fallback."""
    pass


# --- Phase Manager ---

class PhaseDecision(BaseModel):
    agent: str = Field(description="Which agent should handle this turn")
    phase: ConversationPhase = ConversationPhase.intake
    directives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list, description="Active constraints for response gen")
    phase_changed: bool = False


# --- RAG Retrieval ---

class RetrievalResult(BaseModel):
    document_id: str
    document_name: str
    content: str
    score: float
    match_type: str = "vector"
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    # Context engineering metadata
    evidence_level: str = ""
    document_type: str = ""
    age_range: list[str] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)


class FacetCounts(BaseModel):
    document_type: dict[str, int] = Field(default_factory=dict)
    tags: dict[str, int] = Field(default_factory=dict)
    age_range: dict[str, int] = Field(default_factory=dict)
    evidence_level: dict[str, int] = Field(default_factory=dict)
    source: dict[str, int] = Field(default_factory=dict)


class RetrievalFilters(BaseModel):
    document_type: str | None = None
    tags: list[str] | None = None
    age_range: str | None = None
    evidence_level: str | None = None
    source: str | None = None


class RetrievalResponse(BaseModel):
    results: list[RetrievalResult] = Field(default_factory=list)
    facets: FacetCounts = Field(default_factory=FacetCounts)
    rewritten_query: str | None = None


# --- Session State ---

class FamilyProfile(BaseModel):
    child_age: str | None = None
    child_name: str | None = None
    diagnosis_status: str | None = None
    challenge_areas: list[str] = Field(default_factory=list)
    attempted_strategies: list[str] = Field(default_factory=list)
    good_day_description: str | None = None
    hardest_situations: list[str] = Field(default_factory=list)


class Goal(BaseModel):
    description: str
    strategy_id: str | None = None
    created_turn: int = 0
    status: str = "active"


class Outcome(BaseModel):
    goal_description: str
    signal: str  # "positive" or "negative"
    detail: str = ""
    turn: int = 0


class SessionState(BaseModel):
    session_id: str
    phase: ConversationPhase = ConversationPhase.intake
    turn_count: int = 0
    family_profile: FamilyProfile = Field(default_factory=FamilyProfile)
    active_strategies: list[str] = Field(default_factory=list)
    recommended_strategies: list[str] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    outcomes: list[Outcome] = Field(default_factory=list)
    intake_question_index: int = 0
    conversation_history: list[dict] = Field(default_factory=list)


# --- Pipeline Trace ---

class PipelineStep(BaseModel):
    name: str
    duration_ms: float = 0.0
    detail: dict = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    steps: list[PipelineStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    input_check: InputCheckResult | None = None
    phase_decision: PhaseDecision | None = None
    retrieval_results: list[RetrievalResult] = Field(default_factory=list)
    agent_used: str = ""
    rewritten_query: str | None = None


# --- API Contracts ---

class SeedSessionRequest(BaseModel):
    """Sent from the frontend after onboarding to pre-populate the session."""
    session_id: str
    child_name: str = ""
    child_age: str = ""
    challenges: list[str] = Field(default_factory=list)
    tried_strategies: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    phase: ConversationPhase
    pipeline_trace: PipelineTrace
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    phase: ConversationPhase
    turn_count: int
    family_profile: FamilyProfile
    active_strategies: list[str]
    goals: list[Goal]


class OutcomesResponse(BaseModel):
    session_id: str
    outcomes: list[Outcome]
    recommended_strategies: list[str]
    goals: list[Goal]
