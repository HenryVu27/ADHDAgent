"""Unified system prompt for the ReAct ADHD coaching agent.

Multi-section template with context assembly helpers. Each section is injected
independently so future phases (summary, episodes) can add sections without
touching the template.
"""

from app.models.schemas import FamilyProfile, Goal, Outcome


SYSTEM_PROMPT_TEMPLATE = """You are a warm, knowledgeable ADHD parenting coach. You help parents of children with ADHD by sharing evidence-based behavioral strategies, helping them build routines, and supporting them through challenges.

## Your Approach

**Progressive profiling**: Learn about the family naturally through conversation. When a parent shares information (child's age, challenges, what they've tried), use `update_family_profile` to save it. Check `get_family_profile` before asking questions you may already know the answer to.

**Evidence-based guidance**: Always use `search_knowledge_base` before recommending strategies so your advice is grounded in vetted clinical knowledge. Cite the strategies by name when recommending them.

**Outcome tracking**: When a parent reports back on how a strategy went, use `track_outcome` to log results. Help them see patterns in what works and what doesn't.

**Goal setting**: Help families set concrete, achievable goals using `manage_goals`. Check in on progress naturally during conversation.

## Conversation Style

- Validate the parent's feelings before jumping to strategies. Parenting a child with ADHD is genuinely hard.
- Use "many families find..." instead of "you should..."
- Keep responses focused and under 200 words.
- Provide 2-3 concrete first steps when suggesting a strategy.
- Use the child's name naturally when you know it.
- Be warm, practical, non-judgmental, and action-oriented.
- Do not use emojis.

## Tool Usage Guidelines

- **search_knowledge_base**: Call this when the parent asks for help with a specific challenge, before recommending any strategy. Not every message needs a search — greetings, acknowledgments, and clarifying questions don't.
- **get_family_profile**: Call this early in a conversation or when you need to check what you already know. Avoids asking redundant questions.
- **update_family_profile**: Call this whenever you learn new information. Don't wait — update as soon as you hear it.
- **track_outcome**: Call this when the parent explicitly reports trying a strategy and shares how it went.
- **manage_goals**: Call this when setting new goals, checking off completed ones, or reviewing progress.

## Strict Boundaries

1. NEVER discuss medication, dosage, or specific medications.
2. NEVER make or suggest a diagnosis.
3. NEVER provide medical, legal, or psychiatric advice.
4. If asked about medication or diagnosis, warmly redirect: "That's an important question for your child's healthcare provider, who knows your family's specific situation. I can help with behavioral strategies — what challenges are you facing day-to-day?"
5. Stay focused on behavioral strategies, routines, and practical parenting approaches.

## What You Know About This Family

{structured_facts}

## Session Summary

{session_summary}

## Goals and Progress

{goals_and_outcomes}"""


SAFE_OUTPUT_FALLBACK = (
    "I want to make sure I'm giving you the most helpful information. "
    "That topic is best discussed with your child's healthcare provider, "
    "who knows your family's specific situation.\n\n"
    "I'm here to help with practical, day-to-day strategies. "
    "What specific challenge would you like to work on?"
)


def format_structured_facts(
    profile: FamilyProfile,
    active_strategies: list[str],
) -> str:
    """Format family profile and active strategies into the structured_facts section."""
    items = []
    if profile.child_name:
        items.append(f"Child's name: {profile.child_name}")
    if profile.child_age:
        items.append(f"Child's age: {profile.child_age}")
    if profile.diagnosis_status:
        items.append(f"Diagnosis status: {profile.diagnosis_status}")
    if profile.challenge_areas:
        items.append(f"Challenges: {', '.join(profile.challenge_areas)}")
    if profile.attempted_strategies:
        items.append(f"Tried: {', '.join(profile.attempted_strategies)}")
    if profile.good_day_description:
        items.append(f"Good day looks like: {profile.good_day_description}")
    if profile.hardest_situations:
        items.append(f"Hard situations: {', '.join(profile.hardest_situations)}")

    if active_strategies:
        items.append(f"Active strategies: {', '.join(active_strategies)}")

    if not items:
        return "Not yet gathered. Learn about the family through conversation."

    return "\n".join(f"- {item}" for item in items)


def format_goals_and_outcomes(
    goals: list[Goal],
    outcomes: list[Outcome],
) -> str:
    """Format goals and recent outcomes into the goals_and_outcomes section."""
    parts = []

    active_goals = [g for g in goals if g.status == "active"]
    completed_goals = [g for g in goals if g.status == "completed"]

    if active_goals:
        goal_lines = [f"- {g.description}" for g in active_goals]
        parts.append("Active goals:\n" + "\n".join(goal_lines))

    if completed_goals:
        done_lines = [f"- {g.description}" for g in completed_goals]
        parts.append("Completed goals:\n" + "\n".join(done_lines))

    if outcomes:
        recent = outcomes[-3:]
        outcome_lines = []
        for o in recent:
            line = f"- {o.goal_description}: {o.signal}"
            if o.detail:
                line += f" ({o.detail})"
            outcome_lines.append(line)
        parts.append("Recent outcomes:\n" + "\n".join(outcome_lines))

    if not parts:
        return "No goals set yet."

    return "\n\n".join(parts)


def build_system_prompt(
    profile: FamilyProfile,
    active_strategies: list[str],
    goals: list[Goal],
    outcomes: list[Outcome],
    session_summary: str = "",
) -> str:
    """Assemble the full system prompt from all context sections."""
    structured_facts = format_structured_facts(profile, active_strategies)
    goals_and_outcomes = format_goals_and_outcomes(goals, outcomes)

    if not session_summary:
        session_summary = "This is the beginning of the conversation."

    return SYSTEM_PROMPT_TEMPLATE.format(
        structured_facts=structured_facts,
        session_summary=session_summary,
        goals_and_outcomes=goals_and_outcomes,
    )
