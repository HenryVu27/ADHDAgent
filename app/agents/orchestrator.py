"""
Pipeline Orchestrator.

Coordinates the full pipeline:
1. NeMo INPUT RAILS (jailbreak, crisis, out-of-scope, content, topic)
2. Phase manager decision (agent routing) + profile update
3. RAG retrieval
4. Agent processing + Gemini response
5. NeMo OUTPUT RAILS (medication, diagnosis, scope)
6. Build PipelineTrace
7. Update session state + conversation history
"""

import logging
import time

from app.agents.context import format_conversation_window
from app.agents.intake import IntakeAgent
from app.agents.progress import ProgressAgent
from app.agents.strategy import StrategyAgent
from app.guardrails.validator import GuardrailsValidator
from app.models.schemas import (
    ChatResponse,
    ConversationPhase,
    Goal,
    GuardrailsError,
    PipelineStep,
    PipelineTrace,
    RetrievalResponse,
    RetrievalResult,
    SeedSessionRequest,
    SessionState,
)
from app.phase_manager import PhaseManager
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates the full pipeline."""

    def __init__(
        self,
        guardrails: GuardrailsValidator,
        phase_manager: PhaseManager,
        retriever: HybridRetriever,
        intake: IntakeAgent,
        strategy: StrategyAgent,
        progress: ProgressAgent,
    ):
        self._guardrails = guardrails
        self._phase_manager = phase_manager
        self._retriever = retriever
        self._agents = {
            "intake": intake,
            "strategy": strategy,
            "progress": progress,
        }
        self._sessions: dict[str, SessionState] = {}

    def get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def seed_session(self, request: SeedSessionRequest) -> None:
        """Pre-populate a session with onboarding data, skipping intake."""
        state = self.get_session(request.session_id)
        state.family_profile.child_name = request.child_name or None
        state.family_profile.child_age = request.child_age or None
        state.family_profile.challenge_areas = request.challenges
        state.family_profile.attempted_strategies = request.tried_strategies
        for goal_text in request.goals:
            state.goals.append(Goal(description=goal_text))
        # Onboarding data satisfies intake requirements — advance to strategy
        state.phase = ConversationPhase.strategy
        state.intake_question_index = 5  # mark intake as done
        logger.info(
            "Session %s seeded: age=%s, challenges=%s, phase=%s",
            request.session_id,
            request.child_age,
            request.challenges,
            state.phase.value,
        )

    async def process(self, message: str, session_id: str) -> ChatResponse:
        """Run the full pipeline."""
        pipeline_start = time.time()
        trace = PipelineTrace()
        state = self.get_session(session_id)
        state.turn_count += 1

        # --- Step 1: NeMo INPUT RAILS ---
        step_start = time.time()
        try:
            input_check = await self._guardrails.check_input(
                message, context={"phase": state.phase.value}
            )
            input_ms = (time.time() - step_start) * 1000
            trace.input_check = input_check
            trace.steps.append(PipelineStep(
                name="input_rails",
                duration_ms=input_ms,
                detail={
                    "is_allowed": input_check.is_allowed,
                    "blocked_reason": input_check.blocked_reason,
                },
            ))

            if not input_check.is_allowed:
                response_text = input_check.override_response or ""
                trace.agent_used = "guardrails"
                trace.total_duration_ms = (time.time() - pipeline_start) * 1000
                self._record_turn(state, message, response_text, "guardrails")
                return ChatResponse(
                    response=response_text,
                    agent_used="guardrails",
                    phase=state.phase,
                    pipeline_trace=trace,
                    session_id=session_id,
                )
        except GuardrailsError as e:
            logger.error("Input rails failed: %s", e)
            input_ms = (time.time() - step_start) * 1000
            trace.steps.append(PipelineStep(
                name="input_rails",
                duration_ms=input_ms,
                detail={"error": str(e)},
            ))
            # Continue without input rails on error

        # --- Step 2: Progressive profiling + Phase Manager Decision ---
        if state.phase != ConversationPhase.intake:
            self._phase_manager.update_profile(message, state)

        step_start = time.time()
        decision = self._phase_manager.decide(message, state)
        phase_ms = (time.time() - step_start) * 1000
        trace.phase_decision = decision
        trace.steps.append(PipelineStep(
            name="phase_manager",
            duration_ms=phase_ms,
            detail={
                "agent": decision.agent,
                "phase": decision.phase.value,
                "directives": decision.directives,
                "constraints": decision.constraints,
                "phase_changed": decision.phase_changed,
            },
        ))

        # --- Step 3: RAG Retrieval ---
        retrieval_results: list[RetrievalResult] = []
        if decision.agent in ("strategy", "progress"):
            step_start = time.time()
            response: RetrievalResponse = await self._retriever.retrieve(
                query=message,
                state=state,
            )
            retrieval_results = response.results
            rag_ms = (time.time() - step_start) * 1000
            trace.retrieval_results = retrieval_results
            trace.rewritten_query = response.rewritten_query
            trace.steps.append(PipelineStep(
                name="rag_retrieval",
                duration_ms=rag_ms,
                detail={
                    "results_count": len(retrieval_results),
                    "rewritten_query": response.rewritten_query,
                    "facets": response.facets.model_dump(),
                    "documents": [
                        {
                            "name": r.document_name,
                            "score": round(r.score, 3),
                            "type": r.match_type,
                        }
                        for r in retrieval_results
                    ],
                },
            ))

        # Format RAG context for the agent
        rag_context = self._format_rag_context(retrieval_results)

        # Track recommended strategies
        for r in retrieval_results:
            if r.document_id and r.document_id not in state.recommended_strategies:
                state.recommended_strategies.append(r.document_id)

        # --- Step 4: Agent Processing + Response Generation ---
        agent = self._agents.get(decision.agent, self._agents["strategy"])
        step_start = time.time()
        response_text = await agent.process(
            message=message,
            decision=decision,
            state=state,
            rag_context=rag_context,
        )
        agent_ms = (time.time() - step_start) * 1000
        trace.agent_used = agent.name
        trace.steps.append(PipelineStep(
            name="response_generation",
            duration_ms=agent_ms,
            detail={"agent": agent.name},
        ))

        # --- Step 5: NeMo OUTPUT RAILS ---
        step_start = time.time()
        validation_ms = 0.0
        validation_detail: dict = {}
        try:
            output_check = await self._guardrails.check_output(
                response_text,
                context={"phase": state.phase.value, "agent": agent.name},
            )
            validation_ms = output_check.duration_ms
            validation_detail = {
                "valid": output_check.is_valid,
                "violation_type": output_check.violation_type,
            }

            if not output_check.is_valid:
                logger.warning("Output rails blocked response: %s", output_check.violation_type)
                response_text = (
                    "I want to make sure I give you helpful, appropriate guidance. "
                    "Could you tell me more about what you're looking for help with today?"
                )
        except GuardrailsError as e:
            logger.error("Output rails failed: %s", e)
            validation_ms = (time.time() - step_start) * 1000
            validation_detail = {"error": str(e)}

        trace.steps.append(PipelineStep(
            name="output_rails",
            duration_ms=validation_ms,
            detail=validation_detail,
        ))

        # --- Finalize ---
        trace.total_duration_ms = (time.time() - pipeline_start) * 1000
        self._record_turn(state, message, response_text, agent.name)

        return ChatResponse(
            response=response_text,
            agent_used=agent.name,
            phase=state.phase,
            pipeline_trace=trace,
            session_id=session_id,
        )

    def _format_rag_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as structured XML for the LLM."""
        if not results:
            return ""

        parts = [f'<retrieval_results count="{len(results)}">']

        for i, r in enumerate(results, 1):
            attrs = [
                f'rank="{i}"',
                f'score="{r.score:.2f}"',
                f'match_type="{r.match_type}"',
            ]
            if r.evidence_level:
                attrs.append(f'evidence_level="{r.evidence_level}"')
            if r.document_type:
                attrs.append(f'document_type="{r.document_type}"')

            parts.append(f'  <result {" ".join(attrs)}>')
            parts.append(f'    <name>{r.document_name}</name>')
            parts.append(f'    <source>{r.source}</source>')
            if r.tags:
                parts.append(f'    <tags>{", ".join(r.tags)}</tags>')
            parts.append(f'    <content>{r.content}</content>')
            parts.append('  </result>')

        parts.append('  <system_instruction>')
        parts.append('    Prefer strategies with evidence_level="strong" when available.')
        parts.append('    Reference the source for credibility (e.g., "According to AAP guidelines...").')
        parts.append('    When multiple results share tags, the parent\'s concern likely spans these areas.')
        parts.append('    Match strategy to child\'s age range when age information is available.')
        parts.append('  </system_instruction>')
        parts.append('</retrieval_results>')

        return "\n".join(parts)

    def _record_turn(self, state: SessionState, message: str, response: str, agent: str):
        """Record the conversation turn in session history."""
        state.conversation_history.append({
            "turn": state.turn_count,
            "user_message": message,
            "agent_response": response,
            "agent": agent,
            "phase": state.phase.value,
        })
