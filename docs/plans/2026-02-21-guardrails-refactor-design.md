# Guardrails Refactor: NeMo Removal + LangGraph-Native Gates

**Date:** 2026-02-21
**Status:** Approved
**Priority:** Simplicity / code cleanup (latency and cost improvements are side effects)

---

## Problem

The current guardrails implementation has three issues:

1. **NeMo is dead code.** NeMo Guardrails is initialized on startup but never used in the default path. The actual guardrails are 9 direct Gemini classifier calls (6 input + 3 output). NeMo adds startup time, memory usage, dependency complexity, and confusion.

2. **9 classifier calls per message is overkill.** Each call asks a single yes/no question. 75% of per-message LLM calls are binary classifiers. A typical knowledge-base query makes ~12 LLM calls (6 input guardrails + 1-2 agent + 3 output guardrails + 1 embedding), taking 6-8 seconds.

3. **Guardrails are invisible in the graph.** They're side effects inside `pre_model_hook` / `post_model_hook` closures, mixed in with context assembly, message trimming, and model routing. Hard to reason about, test independently, or trace.

### Industry context

No production healthcare company uses NeMo Guardrails. No frontier lab recommends it. The FDA's framework is technology-agnostic — they require demonstrable safety, not specific tools. The industry pattern is: strong system prompt + 1-2 structured classifiers + scoped tools.

---

## Solution

### Architecture: LangGraph-native guardrail nodes

Replace the pre/post hook guardrails pattern with explicit graph nodes:

```
START
  |
  v
input_gate (1 structured Gemini call: crisis + jailbreak)
  |--[blocked]--> END (with block response)
  |--[allowed]-->
  v
prepare_context (context assembly, message trimming, model routing)
  |
  v
agent (create_react_agent — internal ReAct loop unchanged)
  |
  v
output_gate (1 structured Gemini call: medication + diagnosis + scope)
  |--[violation]--> END (with safe fallback)
  |--[valid]-->
  v
END
```

The ReAct agent itself stays as `create_react_agent` — we wrap it in an outer graph, not rewrite it. The `pre_model_hook` is stripped to context assembly only. `post_model_hook` is removed.

### Consolidated classifiers

**Input gate (1 call):** Structured JSON output classifying crisis + jailbreak.

```python
class InputClassification(BaseModel):
    crisis: bool = False        # Self-harm, abuse, immediate danger
    jailbreak: bool = False     # Prompt injection, role-play escape
    reasoning: str = ""         # For tracing/debugging
```

**Output gate (1 call):** Structured JSON output classifying medication + diagnosis + scope.

```python
class OutputClassification(BaseModel):
    medication_recommendation: bool = False
    diagnosis_claim: bool = False
    scope_violation: bool = False
    reasoning: str = ""
```

### Soft checks move to system prompt

4 checks are removed as external classifiers and absorbed into an enhanced system prompt boundaries section:

| Check | Why safe to move | What catches failures |
|---|---|---|
| off-topic | Agent sees full conversation context. Current classifier has false positive issues requiring keyword-based overrides. | System prompt instruction + agent's natural conversation flow |
| out-of-scope | System prompt already has "Strict Boundaries" covering this. | System prompt + output gate catches medication/diagnosis/scope |
| content-safety | Gemini's built-in safety filters handle hate/harassment. | Built-in filters + system prompt |
| language | Simple instruction for the agent to follow. | System prompt instruction |

### Why crisis + jailbreak stay external

- **Crisis** has deterministic routing — must return specific phone numbers (988, 911, crisis text line). System prompt might paraphrase or omit resources.
- **Jailbreak** attempts are adversarial — the system prompt is the target. An independent classifier can't be manipulated through the conversation.

### Why output checks stay external

These are the highest-liability rails. If the agent recommends Adderall, that needs to be caught before reaching the parent. The system prompt has boundary instructions, but the output gate is the safety net for the cases where the model ignores those instructions.

### LLM calls per message: before vs after

| Scenario | Before | After | Savings |
|---|---|---|---|
| Simple greeting | 10 | 3 | 70% |
| Knowledge search | 12-13 | 5 | 58% |
| Blocked message | 6 | 1 | 83% |

---

## Enhanced System Prompt Boundaries

Replace the current "Strict Boundaries" section (5 rules) with:

```
## Strict Boundaries

**Scope** — You are an ADHD parenting coach. You ONLY help with: behavioral
strategies, daily routines, emotional regulation, communication skills, positive
reinforcement, transition planning, homework support, and parent self-care.

1. NEVER discuss medication, dosage, or specific medications.
2. NEVER make or suggest a diagnosis.
3. NEVER provide medical, legal, psychiatric, nutrition therapy, or OT advice.
4. If asked about medication, diagnosis, or any medical topic, warmly redirect:
   "That's an important question for your child's healthcare provider, who knows
   your family's specific situation. I can help with behavioral strategies —
   what challenges are you facing day-to-day?"
5. Stay focused on behavioral strategies, routines, and practical parenting.

**Language** — If the parent writes primarily in a language other than English,
respond: "I'm currently only available in English. Could you share what's going
on in English so I can help you with strategies for your child?"

**Staying on topic** — If a message is completely unrelated to children,
parenting, or ADHD (e.g., sports scores, politics, recipes), gently redirect:
"I'm specifically designed to help with ADHD parenting strategies. What's going
on with your child that I can help with?" Greetings, thanks, and emotional
context from parents are always on-topic.

**Content safety** — If a parent promotes harmful practices toward children
(physical punishment, emotional abuse, neglect), do not engage with the harmful
content. Redirect toward positive approaches. Note: parents expressing normal
frustration ("I'm so frustrated", "I want to scream") is completely normal —
validate their feelings and offer support.
```

---

## File Changes

### Delete entirely

| File | Reason |
|---|---|
| `app/guardrails/gemini_provider.py` | NeMo LangChain adapter — dead code |
| `app/guardrails/config/config.yml` | NeMo Colang config |
| `app/guardrails/config/rails.co` | NeMo Colang flows |
| `app/guardrails/config/config.py` | NeMo custom actions |
| `app/guardrails/config/` directory | Entire NeMo config dir |

### Rewrite

| File | What changes |
|---|---|
| `app/guardrails/validator.py` | New `InputGate` and `OutputGate` classes with single structured Gemini calls. No NeMo imports. ~150 lines (down from 392). |
| `app/agent/graph.py` | Custom `StateGraph` with input_gate → prepare_context → agent → output_gate nodes. |
| `app/agent/hooks.py` | Strip to context-assembly-only `prepare_context` function. Remove all guardrails logic. |
| `app/agent/state.py` | Add gate result fields. |
| `tests/test_guardrails.py` | Test new `InputGate` and `OutputGate` classes. |
| `tests/test_agent_hooks.py` | Test context-assembly-only function. Remove guardrails assertions. |

### Edit (minor)

| File | What changes |
|---|---|
| `app/main.py` | Remove NeMo init step. Wire new gate classes. Update log messages and FastAPI description. |
| `app/config.py` | Remove `NEMO_GUARDRAILS_TIMEOUT_MS`. Add `GUARDRAILS_TIMEOUT_S`. |
| `app/agent/prompts.py` | Enhanced "Strict Boundaries" section absorbing soft checks. |
| `app/models/schemas.py` | Update `GuardrailsError` docstring. Keep `InputCheckResult` / `OutputCheckResult`. |
| `app/agent/orchestrator.py` | Minor — `input_blocked` state field still works the same way. |
| `requirements.txt` | Remove `nemoguardrails>=0.11.0`. Check if `nest_asyncio` is used elsewhere; if not, remove. |

---

## Testing Strategy

1. **Unit tests for gates:** Mock `GeminiClient.generate` to return structured JSON. Test each classification scenario (crisis detected, jailbreak detected, medication recommendation caught, etc.).
2. **Unit tests for context assembly:** Already exist in `test_agent_hooks.py` — update to test the stripped-down function.
3. **Integration tests for graph flow:** Test the full graph with mocked gates to verify routing (blocked → END, allowed → agent → output gate → END).
4. **E2E tests:** Existing `test_e2e_conversations.py` continues to validate the full pipeline with real API calls.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| System prompt fails to catch off-topic messages | Low risk — agent has full conversation context. Monitor in production. Can always add a check back if needed. |
| Structured output parsing fails (malformed JSON) | Fallback: if JSON parse fails, default to "allowed" for input / "valid" for output (fail open, same as current behavior on errors). |
| Wrapping create_react_agent in outer graph causes state issues | The outer graph passes state through. The inner agent operates on `messages` as before. Test thoroughly. |
| Output gate misses a violation the 3-call version would have caught | Single call with structured output actually provides MORE context to the classifier (it sees all 3 checks in one prompt, can reason holistically). Should be equivalent or better. |
