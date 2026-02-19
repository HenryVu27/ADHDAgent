"""
Pipeline Orchestrator.

Coordinates the full 3-layer pipeline:
1. Extract predicates (Layer 1)
2. Safety check
3. Rules engine decision (Layer 2)
4. RAG retrieval
5. Agent processing + Gemini response (Layer 3)
6. Post-generation validation
7. Build PipelineTrace
8. Update session state + conversation history
"""

import logging
import time

from app.agents.context import format_conversation_window
from app.agents.intake import IntakeAgent
from app.agents.progress import ProgressAgent
from app.agents.safety import SafetyMonitor
from app.agents.strategy import StrategyAgent
from app.models.schemas import (
    ChatResponse,
    ConversationPhase,
    ExtractionResult,
    Goal,
    PipelineStep,
    PipelineTrace,
    RetrievalResponse,
    RetrievalResult,
    SafetyLevel,
    SeedSessionRequest,
    SessionState,
)
from app.predicates.extractor import PredicateExtractor
from app.rag.retriever import HybridRetriever
from app.rules.interface import RulesEngine

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Coordinates the full 3-layer pipeline."""

    def __init__(
        self,
        extractor: PredicateExtractor,
        safety: SafetyMonitor,
        rules_engine: RulesEngine,
        retriever: HybridRetriever,
        intake: IntakeAgent,
        strategy: StrategyAgent,
        progress: ProgressAgent,
        guardrails_validator=None,
    ):
        self._extractor = extractor
        self._safety = safety
        self._rules = rules_engine
        self._retriever = retriever
        self._guardrails = guardrails_validator
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
        """Run the full 3-layer pipeline."""
        pipeline_start = time.time()
        trace = PipelineTrace()
        state = self.get_session(session_id)
        state.turn_count += 1

        # Format recent context for extraction and safety (short window)
        recent_context = format_conversation_window(state.conversation_history, max_turns=2)

        # --- Step 1: Predicate Extraction (Layer 1) ---
        step_start = time.time()
        extraction = await self._extractor.extract(message, conversation_context=recent_context)
        extraction_ms = (time.time() - step_start) * 1000
        trace.extraction = extraction
        trace.steps.append(PipelineStep(
            name="predicate_extraction",
            duration_ms=extraction_ms,
            detail={
                "method": extraction.method.value,
                "count": len(extraction.predicates),
                "predicates": [p.model_dump() for p in extraction.predicates],
            },
        ))

        # --- Step 2: Safety Check ---
        step_start = time.time()
        safety_result = await self._safety.check(message, conversation_context=recent_context)
        safety_ms = (time.time() - step_start) * 1000
        trace.safety = safety_result
        trace.steps.append(PipelineStep(
            name="safety_check",
            duration_ms=safety_ms,
            detail={
                "level": safety_result.level.value,
                "detected_topic": safety_result.detected_topic,
            },
        ))

        # If safety triggered, short-circuit the pipeline
        if safety_result.level != SafetyLevel.safe:
            response_text = safety_result.response_override or ""
            trace.agent_used = "safety"
            trace.total_duration_ms = (time.time() - pipeline_start) * 1000
            self._record_turn(state, message, response_text, "safety")
            return ChatResponse(
                response=response_text,
                agent_used="safety",
                phase=state.phase,
                pipeline_trace=trace,
                session_id=session_id,
            )

        # --- Step 3: Rules Engine Decision (Layer 2) ---
        step_start = time.time()
        decision = self._rules.decide(extraction, state)
        rules_ms = (time.time() - step_start) * 1000
        trace.rules_decision = decision
        trace.steps.append(PipelineStep(
            name="rules_engine",
            duration_ms=rules_ms,
            detail={
                "agent": decision.agent,
                "phase": decision.phase.value,
                "directives": decision.directives,
                "constraints": decision.constraints,
                "phase_changed": decision.phase_changed,
                "reasoning": decision.reasoning,
            },
        ))

        # --- Step 4: RAG Retrieval ---
        retrieval_results: list[RetrievalResult] = []
        relevance_confident = True
        if decision.agent in ("strategy", "progress"):
            step_start = time.time()
            response: RetrievalResponse = await self._retriever.retrieve(
                query=message,
                predicates=extraction.predicates,
                state=state,
            )
            retrieval_results = response.results
            relevance_confident = response.relevance_confident
            rag_ms = (time.time() - step_start) * 1000
            trace.retrieval_results = retrieval_results
            trace.rewritten_query = response.rewritten_query
            trace.relevance_confident = response.relevance_confident
            trace.steps.append(PipelineStep(
                name="rag_retrieval",
                duration_ms=rag_ms,
                detail={
                    "results_count": len(retrieval_results),
                    "rewritten_query": response.rewritten_query,
                    "relevance_confident": response.relevance_confident,
                    "facets": response.facets.model_dump(),
                    "documents": [
                        {
                            "name": r.document_name,
                            "score": round(r.score, 3),
                            "type": r.match_type,
                            "rerank_score": round(r.rerank_score, 3) if r.rerank_score is not None else None,
                        }
                        for r in retrieval_results
                    ],
                },
            ))

        # Format RAG context for the agent
        if relevance_confident:
            rag_context = self._format_rag_context(retrieval_results)
        else:
            rag_context = self._format_uncertain_context(retrieval_results)

        # Track recommended strategies
        for r in retrieval_results:
            if r.document_id and r.document_id not in state.recommended_strategies:
                state.recommended_strategies.append(r.document_id)

        # --- Step 5: Agent Processing + Response Generation (Layer 3) ---
        agent = self._agents.get(decision.agent, self._agents["strategy"])
        step_start = time.time()
        response_text = await agent.process(
            message=message,
            extraction=extraction,
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

        # --- Step 6: Post-generation Validation ---
        step_start = time.time()
        if self._guardrails:
            validation = await self._guardrails.validate(
                response_text,
                context={"phase": state.phase.value, "agent": agent.name},
            )
            is_valid = validation.is_valid
            validation_ms = validation.duration_ms
            validation_detail = {
                "valid": is_valid,
                "method": validation.method,
                "violation_type": validation.violation_type,
                "detail": validation.detail,
            }
        else:
            is_valid = self._rules.validate_response(response_text, decision)
            validation_ms = (time.time() - step_start) * 1000
            validation_detail = {"valid": is_valid, "method": "keyword_rules_engine"}
        trace.steps.append(PipelineStep(
            name="response_validation",
            duration_ms=validation_ms,
            detail=validation_detail,
        ))

        if not is_valid:
            logger.warning("Response failed validation — using safety fallback")
            response_text = (
                "I want to make sure I give you helpful, appropriate guidance. "
                "Could you tell me more about what you're looking for help with today?"
            )

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
        """Format retrieval results as structured XML for the LLM.

        Applies context engineering: metadata teaches the LLM how to use
        the results (evidence level, match quality, source attribution).
        """
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

    def _format_uncertain_context(self, results: list[RetrievalResult]) -> str:
        """Format context when retrieval confidence is low (CRAG-lite).

        Instructs the LLM to acknowledge uncertainty and ask clarifying
        questions rather than presenting low-confidence information as fact.
        """
        parts = ['<retrieval_results confidence="low">']
        parts.append('  <system_instruction>')
        parts.append('    IMPORTANT: The retrieval system has LOW CONFIDENCE in these results.')
        parts.append('    Do NOT present this information as directly answering the parent\'s question.')
        parts.append('    Instead:')
        parts.append('    1. Acknowledge what the parent asked about')
        parts.append('    2. Share that you want to make sure you give the most relevant guidance')
        parts.append('    3. Ask a clarifying question to better understand their specific situation')
        parts.append('    4. You may briefly mention a general related strategy, but frame it tentatively')
        parts.append('  </system_instruction>')

        if results:
            parts.append('  <low_confidence_context>')
            for r in results[:2]:
                parts.append(f'    <result name="{r.document_name}" score="{r.score:.2f}">')
                parts.append(f'      {r.content[:200]}...')
                parts.append('    </result>')
            parts.append('  </low_confidence_context>')

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
