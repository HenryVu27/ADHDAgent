"""Compare retrieval quality across chunking strategies.

Builds all three collections (none, recursive_contextual, semantic),
runs the same queries against each, and reports Recall@k and MRR.

Requires GEMINI_API_KEY. Run: ./adhd312/bin/python eval/runners/chunking_comparison.py
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Comprehensive eval queries covering different document types, topics, and query styles.
# Each query has expected document IDs (the documents most relevant to the query).
EVAL_QUERIES = [
    # --- Specific / direct queries ---
    {
        "query": "How can I set up a daily report card with my child's teacher?",
        "expected": ["homebased_daily_report_card_drc", "daily_behavior_report_cards", "daily_schoolhome_report_card"],
    },
    {
        "query": "What are the symptoms of inattentive ADHD?",
        "expected": ["adhd_predominately_inattentive_symptoms", "inattention_symptoms_in_adhd"],
    },
    {
        "query": "How do I get a 504 plan for my child with ADHD?",
        "expected": ["understanding_section_504_rights_for_students_with", "school_ef_accommodations_504_iep"],
    },
    {
        "query": "What is the ABC model of behavior modification?",
        "expected": ["implementing_behavior_modification_programs_abc_mo"],
    },
    {
        "query": "visual timer strategies for time blindness",
        "expected": ["time_blindness_interventions"],
    },
    {
        "query": "body doubling for homework focus",
        "expected": ["body_doubling_ef_support", "task_initiation_homework_refusal"],
    },
    {
        "query": "How do I organize my child's backpack and school materials?",
        "expected": ["organization_backpack_school_materials"],
    },
    {
        "query": "teaching social skills to children with ADHD",
        "expected": ["systematic_teaching_of_social_skills_for_children", "friendship_skills_groups", "social_scripts_training"],
    },
    {
        "query": "melatonin and sleep problems in ADHD kids",
        "expected": ["sleep_hygiene_adhd", "bedtime_winddown_ef_strategies"],
    },

    # --- Vague / broad queries ---
    {
        "query": "my kid can't sit still",
        "expected": ["hyperactivityimpulsivity_symptoms_in_adhd", "adhd_predominately_hyperactiveimpulsive_symptoms"],
    },
    {
        "query": "nothing works with my child's behavior",
        "expected": ["parent_training_in_behavior_therapy", "behavioral_treatments_for_adhd", "behavior_therapy_for_adhd"],
    },
    {
        "query": "school keeps calling about my son",
        "expected": ["schoolbased_interventions_for_adhd", "teacher_communication", "behavioral_classroom_management"],
    },
    {
        "query": "how do I help with morning chaos",
        "expected": ["morning_routine_ef_support", "time_blindness_interventions"],
    },
    {
        "query": "homework every night is a battle",
        "expected": ["homework_battles", "task_initiation_homework_refusal", "homework_and_organization_interventions_for_studen"],
    },

    # --- Emotional / parent-perspective queries ---
    {
        "query": "I feel like a terrible parent, I can't handle this anymore",
        "expected": ["parent_self_care"],
    },
    {
        "query": "my child has meltdowns in the grocery store and I'm so embarrassed",
        "expected": ["public_meltdowns", "emotional_dysregulation_coregulation"],
    },
    {
        "query": "my daughter says everyone hates her and she has no friends",
        "expected": ["peer_relationship_challenges", "rsd_kids", "self_esteem_adhd"],
    },
    {
        "query": "my child cries and rages when he loses at games",
        "expected": ["frustration_tolerance", "emotional_dysregulation_coregulation", "anger_management_adhd"],
    },
    {
        "query": "siblings are constantly fighting and I'm exhausted",
        "expected": ["sibling_conflict_adhd", "parent_self_care"],
    },

    # --- Multi-topic / cross-domain queries ---
    {
        "query": "I need help with bedtime routine and reducing screen time before sleep",
        "expected": ["bedtime_winddown_ef_strategies", "screen_time_gaming_adhd", "sleep_hygiene_adhd"],
    },
    {
        "query": "how to talk to my child about their ADHD diagnosis and build confidence",
        "expected": ["talking_adhd_diagnosis", "self_esteem_adhd", "adhd_strengths_reframing"],
    },
    {
        "query": "positive reinforcement at home and at school with a reward chart",
        "expected": ["implementing_a_rewards_chart", "using_positive_reinforcement_structure_and_discipl", "parent_training_in_behavior_therapy"],
    },
    {
        "query": "how to communicate effectively with my child's teacher about ADHD accommodations",
        "expected": ["teacher_communication", "familyschool_partnerships_for_adhd_management", "school_ef_accommodations_504_iep"],
    },

    # --- Age-specific queries ---
    {
        "query": "behavior therapy for my 4 year old with ADHD",
        "expected": ["behavior_therapy_for_young_children_with_adhd", "behavioral_parent_training_for_adhd"],
    },
    {
        "query": "my teenager with ADHD is struggling with risky behavior and substance use",
        "expected": ["increased_risks_during_adolescence"],
    },

    # --- Executive function queries ---
    {
        "query": "how to break big school projects into smaller steps",
        "expected": ["breaking_tasks_micro_steps", "planning_prioritization_visual_supports"],
    },
    {
        "query": "my child can't remember instructions and loses everything",
        "expected": ["working_memory_scaffolds_daily_tasks", "inattention_symptoms_in_adhd"],
    },
    {
        "query": "games and activities that build executive function skills",
        "expected": ["building_ef_skills_play_games"],
    },
    {
        "query": "my child gets stuck and can't switch between tasks or activities",
        "expected": ["cognitive_flexibility_task_shifting", "transition_warnings_scripts"],
    },
    {
        "query": "what apps or technology tools help ADHD kids stay organized",
        "expected": ["technology_tools_ef_support", "organization_backpack_school_materials"],
    },
]


async def evaluate_strategy(strategy: str, queries: list[dict], top_k: int = 5) -> dict:
    """Build index for a strategy and evaluate retrieval quality."""
    os.environ["RAG_CHUNKING_STRATEGY"] = strategy
    os.environ["RAG_CONTEXTUAL_HEADERS"] = "false"  # skip headers for fair comparison
    os.environ["QDRANT_URL"] = ":memory:"  # fresh in-memory index per strategy

    # Reload all modules that cache `settings` so env var changes take effect
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.rag.knowledge_store
    importlib.reload(app.rag.knowledge_store)
    import app.rag.retriever
    importlib.reload(app.rag.retriever)

    from app.config import settings
    from app.llm.client import GeminiClient
    from app.rag.knowledge_store import KnowledgeStore
    from app.rag.retriever import HybridRetriever

    logger.info("Strategy=%s, collection=%s, chunks will differ=%s",
                strategy, settings.RAG_CHUNKING_STRATEGY, strategy != "none")

    gemini = GeminiClient()
    store = KnowledgeStore()
    await store.build_index(gemini)

    retriever = HybridRetriever(knowledge_store=store, gemini_client=gemini)

    results = {"strategy": strategy, "queries": [], "recall_at_k": 0.0, "mrr": 0.0}
    total_recall = 0.0
    total_rr = 0.0
    evaluated = 0

    for q in queries:
        if not q["expected"]:
            continue
        evaluated += 1
        response = await retriever.retrieve(q["query"], top_k=top_k, skip_rewrite=True)
        retrieved_ids = [r.document_id for r in response.results]

        # Recall@k
        hits = sum(1 for eid in q["expected"] if eid in retrieved_ids)
        recall = hits / len(q["expected"])
        total_recall += recall

        # MRR
        rr = 0.0
        for eid in q["expected"]:
            if eid in retrieved_ids:
                rank = retrieved_ids.index(eid) + 1
                rr = max(rr, 1.0 / rank)
        total_rr += rr

        results["queries"].append({
            "query": q["query"],
            "expected": q["expected"],
            "retrieved": retrieved_ids[:top_k],
            "recall": recall,
            "rr": rr,
        })

    if evaluated > 0:
        results["recall_at_k"] = total_recall / evaluated
        results["mrr"] = total_rr / evaluated

    logger.info(
        "Strategy=%s  Recall@%d=%.3f  MRR=%.3f",
        strategy, top_k, results["recall_at_k"], results["mrr"],
    )
    return results


async def main():
    strategies = ["none", "recursive_contextual", "semantic"]
    all_results = []

    for strategy in strategies:
        logger.info("--- Evaluating strategy: %s ---", strategy)
        result = await evaluate_strategy(strategy, EVAL_QUERIES)
        all_results.append(result)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Strategy':<25} {'Recall@5':<12} {'MRR':<12}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['strategy']:<25} {r['recall_at_k']:<12.3f} {r['mrr']:<12.3f}")
    print("=" * 60)

    # Save results
    output_path = Path(__file__).parent.parent / "data" / "results" / "chunking_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
