# ADHDAgent

An agentic framework for parent-facing ADHD coaching that combines LLM flexibility with Answer Set Programming (ASP) guardrails to deliver clinician-informed, structured conversations.

## Motivation

Millions of families with children diagnosed with ADHD struggle to access and sustain psychosocial treatment. While LLMs can engage in natural conversation, they lack the logical rigor needed for clinical safety. This project explores a **hybrid architecture** where:

- **LLMs** handle natural language understanding, empathy, and generation
- **ASP (Answer Set Programming)** enforces conversation structure and clinician-defined boundaries
- **RAG** grounds responses in vetted ADHD parenting knowledge
- **Multi-agent orchestration** breaks complex coaching into specialized, composable agents

## Architecture

```
Parent Input (text)
       |
       v
[Predicate Extraction] -- LLM extracts structured predicates from free text
       |
       v
[ASP Reasoning Engine] -- Determines valid conversation moves given current state
       |
       v
[Agent Orchestrator] -- Routes to specialized agents based on ASP output
       |
       +--> [Intake Agent]       -- Gathers family context and child profile
       +--> [Strategy Agent]     -- Recommends evidence-based parenting strategies
       +--> [Progress Agent]     -- Tracks behavioral goals and outcomes
       +--> [Safety Monitor]     -- Enforces clinician-informed guardrails
       |
       v
[Response Generator] -- LLM produces empathetic, parent-friendly response
       |
       v
Parent Output (text)
```

## Key Components

### Predicate Extraction (`app/predicates/`)
Extracts structured predicates from parent utterances using LLMs. For example:
- *"My son won't do his homework and keeps getting distracted"* becomes:
  - `child_behavior(avoidance, homework)`
  - `child_behavior(distraction, homework)`
  - `parent_concern(academic_performance)`

### ASP Conversation Engine (`app/asp/`)
Logic programs defining valid conversation transitions, topic boundaries, and safety constraints. The ASP solver determines what the chatbot *should* do next based on conversation state.

### Multi-Agent System (`app/agents/`)
Specialized agents handle different aspects of coaching:
- **Intake Agent**: Builds family profile through guided questions
- **Strategy Agent**: Matches situations to evidence-based interventions
- **Progress Agent**: Tracks goals, celebrates wins, adjusts plans
- **Safety Monitor**: Validates all responses against clinical guardrails

### RAG Knowledge Base (`app/rag/`)
Retrieval-augmented generation over vetted ADHD parenting resources, behavioral strategies, and clinical guidelines.

## Tech Stack

- **Backend**: Python, FastAPI
- **LLM Integration**: OpenAI API / local models
- **ASP Solver**: Clingo (Potassco) / s(CASP)
- **Vector Store**: FAISS
- **Frontend**: HTML/CSS/JS (simple chat interface)

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/HenryVu27/ADHDAgent.git
cd ADHDAgent
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Running

```bash
# Start the backend server
uvicorn app.main:app --reload --port 8000

# Open the frontend
# Navigate to http://localhost:8000 in your browser
```

## Project Structure

```
ADHDAgent/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration and environment variables
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py             # Base agent class
│   │   ├── orchestrator.py     # Agent orchestration / supervisor
│   │   ├── intake.py           # Family intake agent
│   │   ├── strategy.py         # Strategy recommendation agent
│   │   ├── progress.py         # Progress tracking agent
│   │   └── safety.py           # Safety monitoring agent
│   ├── asp/
│   │   ├── __init__.py
│   │   ├── engine.py           # ASP solver interface
│   │   └── rules/
│   │       ├── conversation.lp # Conversation transition rules
│   │       ├── safety.lp       # Safety constraint rules
│   │       └── topics.lp       # Topic boundary definitions
│   ├── predicates/
│   │   ├── __init__.py
│   │   └── extractor.py        # LLM-based predicate extraction
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py        # RAG retrieval pipeline
│   │   └── knowledge_base.py   # Knowledge base management
│   ├── knowledge/
│   │   └── adhd_strategies.json # Vetted ADHD parenting strategies
│   └── api/
│       ├── __init__.py
│       └── routes.py           # API route definitions
├── frontend/
│   ├── index.html              # Chat interface
│   └── static/
│       ├── style.css           # Styling
│       └── app.js              # Frontend logic
├── tests/
│   ├── __init__.py
│   ├── test_predicates.py      # Predicate extraction tests
│   └── test_asp.py             # ASP engine tests
├── .env.example                # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

## Research Context

This project draws on the intersection of:
- **Answer Set Programming** for knowledge representation and non-monotonic reasoning
- **s(CASP)** goal-directed ASP for commonsense reasoning in dialog systems
- **Digital therapeutics** for ADHD behavioral intervention
- **Agentic AI** architectures for reliable, structured conversations

## License

MIT
