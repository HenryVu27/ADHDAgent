// TypeScript types mirroring app/models/schemas.py

export type ConversationPhase = "intake" | "strategy" | "progress" | "followup"

// Guardrails
export interface InputCheckResult {
  is_allowed: boolean
  blocked_reason: string | null
  override_response: string | null
  duration_ms: number
}

// Phase Manager
export interface PhaseDecision {
  agent: string
  phase: ConversationPhase
  directives: string[]
  constraints: string[]
  phase_changed: boolean
}

// RAG Retrieval
export interface RetrievalResult {
  document_id: string
  document_name: string
  content: string
  score: number
  match_type: string
  source: string
  tags: string[]
}

// Session State
export interface FamilyProfile {
  child_age: string | null
  child_name: string | null
  diagnosis_status: string | null
  challenge_areas: string[]
  attempted_strategies: string[]
  good_day_description: string | null
  hardest_situations: string[]
}

export interface Goal {
  description: string
  strategy_id: string | null
  created_turn: number
  status: string
}

export interface Outcome {
  goal_description: string
  signal: string
  detail: string
  turn: number
}

// Pipeline Trace
export interface PipelineStep {
  name: string
  duration_ms: number
  detail: Record<string, unknown>
}

export interface PipelineTrace {
  steps: PipelineStep[]
  total_duration_ms: number
  input_check: InputCheckResult | null
  phase_decision: PhaseDecision | null
  retrieval_results: RetrievalResult[]
  agent_used: string
}

// API Contracts
export interface ChatRequest {
  message: string
  session_id: string
}

export interface ChatResponse {
  response: string
  agent_used: string
  phase: ConversationPhase
  pipeline_trace: PipelineTrace
  session_id: string
}

export interface SessionResponse {
  session_id: string
  phase: ConversationPhase
  turn_count: number
  family_profile: FamilyProfile
  active_strategies: string[]
  goals: Goal[]
}

export interface OutcomesResponse {
  session_id: string
  outcomes: Outcome[]
  recommended_strategies: string[]
  goals: Goal[]
}

// Knowledge documents (for resource library)
export interface KnowledgeDocument {
  id: string
  name: string
  description: string
  tags: string[]
  steps?: string[]
  key_points?: string[]
  evidence_level?: string
  source?: string
}

// Frontend-only types
export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  agentUsed?: string
  pipelineTrace?: PipelineTrace
}

export interface User {
  name: string
  email: string
  createdAt: string
}

export interface OnboardingData {
  childName: string
  childAge: string
  challenges: string[]
  triedStrategies: string[]
  goals: string[]
}

// Observability types
export interface ToolCallRecord {
  name: string
  args: Record<string, unknown>
  result: string
  duration_ms: number
}

export interface AgentReasoningStep {
  step_index: number
  thought: string
  tool_call: ToolCallRecord | null
  is_final: boolean
}

export interface EnrichedTrace {
  session_id: string
  turn: number
  timestamp: string
  pipeline_steps: PipelineStep[]
  total_duration_ms: number
  reasoning_steps: AgentReasoningStep[]
  tool_calls: ToolCallRecord[]
  model_tier: string
  input_blocked: boolean
  blocked_reason: string
  agent_used: string
}

export interface AnalysisFlag {
  flag_type: string
  severity: string
  description: string
  evidence: string
}

export interface TurnAnalysis {
  session_id: string
  turn: number
  flags: AnalysisFlag[]
  quality_score: number
  summary: string
  tool_call_assessment: string
  timestamp: string
}

export interface ObservabilityEvent {
  category: string
  event_type: string
  session_id: string
  turn: number
  timestamp: string
  duration_ms: number
  detail: Record<string, unknown>
  level: string
}

export interface SessionOverview {
  session_id: string
  turn_count: number
  created_at: string
  updated_at: string
  total_flags: number
  avg_quality_score: number
  tool_calls_count: number
  blocked_count: number
}

export interface SessionDetailResponse {
  session_id: string
  messages: Array<{
    role: string
    content: string
    turn: number
    blocked?: boolean
    blocked_reason?: string
  }>
  traces: EnrichedTrace[]
  analyses: TurnAnalysis[]
  events: ObservabilityEvent[]
}

export interface SessionListResponse {
  sessions: SessionOverview[]
}

// Main app session list
export interface SessionListItem {
  session_id: string
  turn_count: number
  phase: string
  created_at: string
  updated_at: string
}

export interface SessionsResponse {
  sessions: SessionListItem[]
}

export interface MessagesResponse {
  session_id: string
  messages: Array<{
    role: string
    content: string
    turn?: number
    blocked?: boolean
  }>
}
