from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.orchestrator import AgentOrchestrator
from app.predicates.extractor import PredicateExtractor
from app.asp.engine import ASPEngine

router = APIRouter()

orchestrator = AgentOrchestrator()
extractor = PredicateExtractor()
asp_engine = ASPEngine()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    predicates: list[dict]
    asp_directives: list[str]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Processes parent input through the full pipeline:
    1. Extract predicates from the message
    2. Run ASP reasoning to determine valid conversation moves
    3. Route to the appropriate agent
    4. Generate and return response
    """
    # Step 1: Extract predicates from parent utterance
    predicates = extractor.extract(request.message)

    # Step 2: ASP reasoning - determine what conversation moves are valid
    asp_directives = asp_engine.reason(
        predicates=predicates,
        session_id=request.session_id,
    )

    # Step 3: Orchestrator routes to the right agent based on ASP output
    result = orchestrator.process(
        message=request.message,
        predicates=predicates,
        asp_directives=asp_directives,
        session_id=request.session_id,
    )

    return ChatResponse(
        response=result["response"],
        agent_used=result["agent"],
        predicates=predicates,
        asp_directives=asp_directives,
    )


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/session/{session_id}/state")
async def get_session_state(session_id: str):
    """Returns current conversation state for a session."""
    return {
        "session_id": session_id,
        "conversation_state": orchestrator.get_state(session_id),
    }
