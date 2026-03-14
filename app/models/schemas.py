"""
Data contracts for the ADHDAgent pipeline.

All Pydantic models are defined here first — contracts before implementation.
Every component in the pipeline consumes and produces these typed models.
"""

import re
from enum import Enum
from pydantic import BaseModel, Field, field_validator


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
    route: str = "pro"  # "pro" (default/safe) or "flash" (simple messages)


class OutputCheckResult(BaseModel):
    is_valid: bool
    violation_type: str | None = None  # "medication" | "diagnosis" | "scope"
    duration_ms: float = 0.0


class InputClassification(BaseModel):
    """Structured output from the input gate classifier."""
    crisis: bool = False
    jailbreak: bool = False
    complexity: str = "complex"  # "simple" or "complex" — defaults complex for safety
    reasoning: str = ""


class OutputClassification(BaseModel):
    """Structured output from the output gate classifier."""
    medication_recommendation: bool = False
    diagnosis_claim: bool = False
    scope_violation: bool = False
    reasoning: str = ""


class GuardrailsError(Exception):
    """Raised when a guardrail gate check fails or times out."""
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
    # Full original document for structured formatting (excluded from API serialization)
    full_doc: dict = Field(default_factory=dict, exclude=True)


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
    adhd_subtype: str | None = None
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
    strategy_name: str
    signal: str  # "positive" or "negative"
    detail: str = ""
    turn: int = 0


class SessionSummary(BaseModel):
    summary: str
    covers_through_turn: int


class EpisodicMemory(BaseModel):
    event_type: str  # "outcome_reported" | "goal_set" | "goal_completed" | "breakthrough"
    summary: str
    outcome: str = ""  # "positive" | "negative" | "mixed"
    strategies_involved: list[str] = Field(default_factory=list)
    emotional_context: str = ""
    turn_range_start: int = 0
    turn_range_end: int = 0


class ProfileChange(BaseModel):
    field: str
    old_value: str
    new_value: str
    turn: int = 0
    created_at: str = ""


class EpisodeLink(BaseModel):
    source_id: int
    target_id: int
    link_type: str   # "same_strategy" | "same_emotion" | "same_goal"
    link_reason: str = ""
    created_at: str = ""


class StoredToolResult(BaseModel):
    tool_name: str
    query: str = ""
    result_text: str
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
    diagnosis_status: str = ""
    adhd_subtype: str = ""
    challenges: list[str] = Field(default_factory=list)
    tried_strategies: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: str = "default"

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or whitespace only")
        return stripped

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', v):
            raise ValueError("session_id must be 1-128 alphanumeric, underscore, or hyphen characters")
        return v


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


# --- Observability ---

class ToolCallRecord(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)
    result: str = ""
    duration_ms: float = 0.0


class AgentReasoningStep(BaseModel):
    step_index: int
    thought: str = ""
    tool_call: ToolCallRecord | None = None
    is_final: bool = False


class EnrichedTrace(BaseModel):
    session_id: str
    turn: int
    timestamp: str = ""
    pipeline_steps: list[PipelineStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    reasoning_steps: list[AgentReasoningStep] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    model_tier: str = "standard"
    input_blocked: bool = False
    blocked_reason: str = ""
    agent_used: str = ""


class AnalysisFlag(BaseModel):
    flag_type: str
    severity: str = "warning"
    description: str = ""
    evidence: str = ""


class TurnAnalysis(BaseModel):
    session_id: str
    turn: int
    flags: list[AnalysisFlag] = Field(default_factory=list)
    quality_score: float = 1.0
    summary: str = ""
    tool_call_assessment: str = ""
    timestamp: str = ""


class ObservabilityEvent(BaseModel):
    category: str
    event_type: str
    session_id: str = ""
    turn: int = 0
    timestamp: str = ""
    duration_ms: float = 0.0
    detail: dict = Field(default_factory=dict)
    level: str = "info"


class SessionOverview(BaseModel):
    session_id: str
    turn_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    total_flags: int = 0
    avg_quality_score: float = 1.0
    tool_calls_count: int = 0
    blocked_count: int = 0


class SessionDetailResponse(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)
    traces: list[EnrichedTrace] = Field(default_factory=list)
    analyses: list[TurnAnalysis] = Field(default_factory=list)
    events: list[ObservabilityEvent] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    sessions: list[SessionOverview] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 50


# --- Main App Session List ---

class SessionListItem(BaseModel):
    session_id: str
    turn_count: int = 0
    phase: str = "intake"
    created_at: str = ""
    updated_at: str = ""


class SessionsResponse(BaseModel):
    sessions: list[SessionListItem] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 50


class MessagesResponse(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 50
