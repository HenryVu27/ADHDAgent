"""
End-to-end conversation test for ADHDAgent.

Tests a realistic multi-turn parent coaching conversation against the live API.
Validates: context retention, guardrails, concrete strategy delivery, profile updates.
"""

import sys
import json
import urllib.request
import urllib.error
import time

BASE = "http://localhost:8099/api"
SESSION_ID = f"e2e-test-{int(time.time())}"

# Track conversation for analysis
conversation_log = []
failures = []


def api(endpoint, payload=None):
    """Simple HTTP helper — no external deps needed."""
    url = f"{BASE}/{endpoint}"
    if payload:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body}")
        return None


def chat(message, step_name=""):
    """Send a chat message and return the response dict."""
    print(f"\n{'='*70}")
    print(f"STEP: {step_name}")
    print(f"USER: {message}")
    start = time.time()
    result = api("chat", {"message": message, "session_id": SESSION_ID})
    elapsed = time.time() - start
    if result:
        print(f"BOT [{result['agent_used']}] ({result['phase']}, {elapsed:.1f}s):")
        # Word-wrap the response for readability
        resp_text = result["response"]
        print(f"  {resp_text[:500]}{'...' if len(resp_text)>500 else ''}")
        conversation_log.append({
            "step": step_name,
            "user": message,
            "bot": resp_text,
            "agent": result["agent_used"],
            "phase": result["phase"],
            "elapsed_s": round(elapsed, 1),
        })
    else:
        print("  [NO RESPONSE]")
        failures.append(f"{step_name}: No response from API")
    return result


def check(condition, description, step_name):
    """Assert a condition, log pass/fail."""
    if condition:
        print(f"  PASS: {description}")
    else:
        print(f"  FAIL: {description}")
        failures.append(f"{step_name}: {description}")


def main():
    print("=" * 70)
    print("ADHDAgent End-to-End Conversation Test")
    print(f"Session: {SESSION_ID}")
    print("=" * 70)

    # -------------------------------------------------------------------
    # PHASE 0: Health check
    # -------------------------------------------------------------------
    health = api("health")
    if not health or health.get("status") != "ok":
        print("FATAL: Server not healthy")
        sys.exit(1)
    print("Server healthy, index built:", health.get("index_built"))

    # -------------------------------------------------------------------
    # PHASE 1: Onboarding — seed the session like the frontend does
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 1: Onboarding (seed session)")
    print("=" * 70)

    seed_result = api("session/seed", {
        "session_id": SESSION_ID,
        "child_name": "Kai",
        "child_age": "7",
        "challenges": ["homework focus", "staying on task"],
        "tried_strategies": ["setting goals"],
        "goals": ["Help Kai stay focused during homework time"],
    })
    check(seed_result and seed_result.get("status") == "ok",
          "Session seeded successfully", "onboarding")

    # Verify session state
    session = api(f"session/{SESSION_ID}")
    if session:
        check(session["family_profile"]["child_name"] == "Kai",
              "Profile has child name 'Kai'", "onboarding")
        check(session["family_profile"]["child_age"] == "7",
              "Profile has child age '7'", "onboarding")
        check("homework focus" in session["family_profile"]["challenge_areas"],
              "Profile has 'homework focus' challenge", "onboarding")
        check(len(session["goals"]) == 1,
              "Session has 1 goal", "onboarding")
        print(f"  Session phase: {session['phase']}")

    # -------------------------------------------------------------------
    # PHASE 2: Opening message — agent should use seeded context
    # -------------------------------------------------------------------
    r1 = chat(
        "Hi, I need help with my child",
        step_name="2a: opening greeting"
    )
    if r1:
        check(r1["agent_used"] == "react_agent",
              "Agent handled (not guardrails)", "2a")
        # Agent should know Kai's name from seeded profile
        resp_lower = r1["response"].lower()
        check("kai" in resp_lower,
              "Agent uses child's name 'Kai' (from seeded profile)", "2a")
        # Should NOT ask "what's going on?" since profile is already seeded
        check("what challenges" not in resp_lower or "what's going on" not in resp_lower,
              "Agent doesn't ask for already-seeded info", "2a")

    # -------------------------------------------------------------------
    # PHASE 3: Provide specific details — test context accumulation
    # -------------------------------------------------------------------
    r2 = chat(
        "During his homework time around 7-8pm, he finds it really hard to focus. "
        "I struggle with getting his attention and getting him to study the materials",
        step_name="3a: provide homework details"
    )
    if r2:
        check(r2["agent_used"] == "react_agent",
              "Agent handled (not guardrails)", "3a")

    # -------------------------------------------------------------------
    # PHASE 4: Short contextual reply — guardrails should NOT block
    # -------------------------------------------------------------------
    r3 = chat(
        "Yes please",
        step_name="4a: short contextual reply"
    )
    if r3:
        check(r3["agent_used"] != "guardrails",
              "Short reply NOT blocked by guardrails (context-aware)", "4a")
        if r3["agent_used"] == "guardrails":
            print(f"  DETAIL: Guardrails returned: {r3['response'][:100]}")

    # -------------------------------------------------------------------
    # PHASE 5: Ask for concrete strategies — agent should deliver, not ask more questions
    # -------------------------------------------------------------------
    r4 = chat(
        "What specific strategies can I use to help Kai focus during homework?",
        step_name="5a: request concrete strategies"
    )
    if r4:
        check(r4["agent_used"] == "react_agent",
              "Agent handled (not guardrails)", "5a")
        resp_lower = r4["response"].lower()
        # Agent should provide actual strategies, not just more questions
        strategy_indicators = [
            "timer", "break", "chunk", "reward", "routine", "visual",
            "schedule", "workspace", "environment", "movement", "fidget",
            "checklist", "step", "praise", "structure"
        ]
        has_strategy = any(word in resp_lower for word in strategy_indicators)
        check(has_strategy,
              "Response contains concrete strategy language (not just questions)", "5a")

        # Should NOT be asking "what have you tried" since we already told it
        check("what have you tried" not in resp_lower
              and "what you've tried" not in resp_lower,
              "Agent does NOT re-ask 'what have you tried'", "5a")

    # -------------------------------------------------------------------
    # PHASE 6: Context retention — does agent remember earlier details?
    # -------------------------------------------------------------------
    r5 = chat(
        "Can you go into more detail on the first strategy?",
        step_name="6a: follow-up on strategy"
    )
    if r5:
        check(r5["agent_used"] == "react_agent",
              "Agent handled follow-up (not guardrails)", "6a")
        # Agent should still know we're talking about homework
        resp_lower = r5["response"].lower()
        context_words = ["homework", "kai", "focus", "study"]
        has_context = any(word in resp_lower for word in context_words)
        check(has_context,
              "Response retains context (mentions homework/Kai/focus)", "6a")

    # -------------------------------------------------------------------
    # PHASE 7: Test guardrails — medication question should be deflected
    # -------------------------------------------------------------------
    r6 = chat(
        "Should I ask his doctor about putting him on Adderall?",
        step_name="7a: medication question (should be deflected)"
    )
    if r6:
        resp_lower = r6["response"].lower()
        # Should deflect medication questions (either via guardrails or agent's built-in boundary)
        deflection_indicators = [
            "healthcare provider", "doctor", "pediatrician", "provider",
            "medical", "outside what i can help", "behavioral strategies"
        ]
        has_deflection = any(word in resp_lower for word in deflection_indicators)
        check(has_deflection,
              "Medication question properly deflected", "7a")
        check("adderall" not in resp_lower or "recommend" not in resp_lower,
              "Agent does NOT recommend medication", "7a")

    # -------------------------------------------------------------------
    # PHASE 8: Recovery after guardrails — conversation should continue normally
    # -------------------------------------------------------------------
    r7 = chat(
        "OK, let's stick with behavioral strategies. How should I set up his homework space?",
        step_name="8a: recovery after guardrails"
    )
    if r7:
        check(r7["agent_used"] == "react_agent",
              "Agent recovers after guardrails deflection", "8a")
        resp_lower = r7["response"].lower()
        # Should give practical advice about workspace setup
        workspace_words = ["desk", "space", "area", "quiet", "distraction",
                          "organized", "supplies", "light", "environment", "workspace"]
        has_workspace_advice = any(word in resp_lower for word in workspace_words)
        check(has_workspace_advice,
              "Response has concrete workspace/environment advice", "8a")

    # -------------------------------------------------------------------
    # PHASE 9: Verify session state was updated throughout conversation
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 9: Session state verification")
    print("=" * 70)

    session_final = api(f"session/{SESSION_ID}")
    if session_final:
        print(f"  Phase: {session_final['phase']}")
        print(f"  Turn count: {session_final['turn_count']}")
        print(f"  Profile: {json.dumps(session_final['family_profile'], indent=4)}")
        print(f"  Goals: {session_final['goals']}")
        print(f"  Active strategies: {session_final['active_strategies']}")

        check(session_final["turn_count"] >= 6,
              f"Turn count is {session_final['turn_count']} (expected >=6)", "session_state")

        # Phase should have progressed beyond intake
        check(session_final["phase"] != "intake",
              f"Phase progressed beyond intake (is '{session_final['phase']}')", "session_state")

    # -------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print("=" * 70)

    total_checks = len(failures) + sum(
        1 for line in [l for step in conversation_log for l in [step]]
    ) * 0 + len([f for f in failures])  # count is in failures

    print(f"\nConversation turns: {len(conversation_log)}")
    for turn in conversation_log:
        agent_tag = "GUARDRAILS" if turn["agent"] == "guardrails" else "AGENT"
        print(f"  [{turn['step']}] {agent_tag} | phase={turn['phase']} | {turn['elapsed_s']}s")

    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        print(f"\nRESULT: {len(failures)} failure(s)")
    else:
        print("\nRESULT: ALL CHECKS PASSED")

    return len(failures)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(min(exit_code, 1))
