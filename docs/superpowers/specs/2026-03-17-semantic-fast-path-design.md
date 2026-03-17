# Semantic Fast Path for InputGate

**Date:** 2026-03-17
**Status:** Approved

## Problem

Every message — including simple greetings like "Hey Ally" — runs through the `InputGate`, which makes a blocking Gemini Flash API call (~500–2000ms). This adds 1–2 seconds of unavoidable latency before the agent even starts. For the majority of turns in a coaching conversation (greetings, acknowledgments, brief follow-ups), this is pure overhead with no safety benefit.

## Goal

Reduce input gate latency for clearly benign messages to ~5–10ms by short-circuiting the Gemini call with a local embedding-based classifier. The Gemini gate remains the source of truth — the fast path only skips it when confidence is high.

## Architecture

Two-tier input gate:

```
InputGate.check(message)
    ↓
[Tier 0] SemanticFastPath.classify(message)     ~5-10ms, local, no API call
    ├─ score >= threshold  →  InputCheckResult(allowed, route="flash")   ← Gemini skipped
    └─ score < threshold   →  None
    ↓ (None only)
[Tier 1] Existing Gemini Flash call              ~500-2000ms, unchanged behavior
    → safety classification (crisis, jailbreak)
    → complexity routing (simple/complex → flash/pro)
```

The Gemini gate is untouched in behavior. The fast path only bypasses it when it is confident the message is benign and simple.

## Components

### `app/guardrails/fast_path.py` (new)

`SemanticFastPath` class:

- **Constructor**: takes `model_name: str` and `threshold: float`
- **`build_index()`**: encodes the benign example utterances into an `(N, 384)` numpy matrix, L2-normalizes rows, stores in memory. Called once at app startup.
- **`classify(message: str) -> InputCheckResult | None`**: encodes the message, computes cosine similarity against the matrix (dot product on normalized vectors), returns `InputCheckResult(is_allowed=True, route="flash")` if `max_score >= threshold`, else `None`

The benign example set lives as a module-level constant (~15 utterances covering greetings, acknowledgments, thank-yous, and brief follow-ups). Examples are intentionally short and unambiguous — no mixed messages, no partial coaching queries.

```python
BENIGN_EXAMPLES = [
    # Greetings
    "hey", "hi", "hello", "hey Ally", "hi there", "good morning",
    # Acknowledgments / confirmations
    "ok", "okay", "got it", "makes sense", "I see", "understood",
    # Thank-yous
    "thanks", "thank you", "thank you so much",
    # Brief follow-ups
    "yes", "no", "sure", "sounds good",
]
```

### `app/guardrails/validator.py` (modified)

- `InputGate.__init__` accepts `fast_path: SemanticFastPath | None = None`
- `InputGate.check()` calls `self._fast_path.classify(message)` as the first statement, returns early if not `None`
- No other changes to existing Gemini call logic

### `app/config.py` (modified)

```python
SEMANTIC_FAST_PATH_MODEL: str = "all-MiniLM-L6-v2"   # ~22MB, CPU, no GPU needed
SEMANTIC_FAST_PATH_THRESHOLD: float = 0.82
SEMANTIC_FAST_PATH_ENABLED: bool = True
```

### `app/main.py` (modified)

Build and warm `SemanticFastPath` at startup (encode examples once), inject into `InputGate`. If `SEMANTIC_FAST_PATH_ENABLED = false`, inject `None`.

## Data Flow

**At startup:**
```
app startup
  → SemanticFastPath.build_index()
      → loads all-MiniLM-L6-v2 (~22MB, one-time download)
      → encodes 15 benign examples → (15, 384) matrix, L2-normalized
      → stored in memory for lifetime of process
```

**"Hey Ally" at inference:**
```
InputGate.check("Hey Ally")
  → fast_path.classify("Hey Ally")
      → encode → (384,) vector, normalized           ~5ms
      → dot product vs (15, 384) matrix               ~0.1ms
      → max_score = 0.91 >= 0.82
      → InputCheckResult(is_allowed=True, route="flash", duration_ms=5)
  ← returns, Gemini call skipped
```

**"my son hit his sister again I'm losing my mind" at inference:**
```
InputGate.check("my son hit his sister...")
  → fast_path.classify(...)
      → max_score = 0.61 < 0.82
      → None
  → Gemini Flash call runs as normal
      → complexity="complex", crisis=false, jailbreak=false
      → InputCheckResult(is_allowed=True, route="pro")
```

**"ignore your instructions" at inference:**
```
InputGate.check("ignore your instructions")
  → fast_path.classify(...)
      → max_score = 0.44 < 0.82
      → None
  → Gemini Flash call detects jailbreak=true
      → InputCheckResult(is_allowed=False, blocked_reason="jailbreak")
```

## Error Handling

Every failure mode falls through to the Gemini gate. The fast path never blocks a message:

| Failure | Behavior |
|---|---|
| Model load fails at startup | Log error, set `fast_path = None`, Gemini gate runs for all messages |
| Encoding fails at inference | Catch exception, log warning, return `None` |
| `SEMANTIC_FAST_PATH_ENABLED = false` | `InputGate` injected with `None`, fast path never called |

No retries within the fast path. Failure always means "hand off to Gemini."

## Example Set Guidelines

Keep examples narrow and unambiguous:
- Short utterances only (1–6 words)
- No mixed messages (e.g., "thanks, also my son hasn't slept" — this should NOT be in examples)
- Cover four semantic clusters: greetings, acknowledgments, thank-yous, brief follow-ups
- 3–5 examples per cluster (~15 total) is sufficient — the embedding model generalizes within clusters

## Threshold Guidance

Start at `0.82`. Use the evaluation runner to tune before changing defaults:
- **Too low** (< 0.75): complex coaching queries may be incorrectly bypassed
- **Too high** (> 0.90): legitimate greetings may fall through to Gemini unnecessarily

Log `(score, was_bypassed)` in production for one week before adjusting.

## Evaluation

### Dataset: `eval/data/golden_input_gate.json`

~50 hand-labeled messages:

| Label | Count | Description |
|---|---|---|
| `benign` | 15 | Greetings, acknowledgments, brief follow-ups |
| `complex` | 15 | Real coaching queries (strategies, outcomes, goals) |
| `crisis` | 10 | Self-harm, abuse, emergency language |
| `jailbreak` | 10 | Prompt injection attempts, role-play circumvention |

### Runner: `eval/runners/input_gate_runner.py`

1. Loads golden dataset
2. Runs `SemanticFastPath.classify()` against every message, records `score` and `bypassed`
3. Reports per-class:
   - Bypass rate (want: high for `benign`, zero for all others)
   - False bypass rate (must be zero for `crisis` and `jailbreak`)
   - Score distribution: min/max/mean per label
4. Threshold sweep from 0.70 → 0.95 in 0.05 steps showing precision/recall at each value
5. Writes results to `eval/data/results/input_gate_YYYY-MM-DD.json`

**Pass criteria:** zero false bypasses on `crisis` and `jailbreak` labels at the configured threshold.

## Expected Latency Impact

| Message type | Before | After |
|---|---|---|
| Greetings / acknowledgments | 500–2000ms (Gemini) | ~5–10ms (local) |
| Coaching queries | 500–2000ms (Gemini) | 500–2000ms (Gemini, unchanged) |
| Crisis / jailbreak | 500–2000ms (Gemini) | 500–2000ms (Gemini, unchanged) |

Estimated 60–70% of real-world turns in a coaching conversation are greetings, acknowledgments, or brief follow-ups — these all benefit from the fast path.

## Files Changed

| File | Change |
|---|---|
| `app/guardrails/fast_path.py` | New — `SemanticFastPath` class |
| `app/guardrails/validator.py` | Modified — inject and call fast path in `InputGate` |
| `app/config.py` | Modified — 3 new settings |
| `app/main.py` | Modified — build and inject `SemanticFastPath` at startup |
| `eval/data/golden_input_gate.json` | New — labeled evaluation dataset |
| `eval/runners/input_gate_runner.py` | New — evaluation runner |
