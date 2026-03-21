"""MemoryManager -- ingests conversation turns into Graphiti knowledge graph.

Runs after the response is sent to the parent. Non-blocking.
Replaces the previous rolling summary / fact extraction / episodic memory system.
"""

import asyncio
import logging
import time

from langsmith import traceable

from app.agent.graphiti_client import ADHD_ENTITY_TYPES, ADHD_EDGE_TYPES
from app.agent.store_protocol import SessionStoreBase
from app.config import settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Ingests conversation turns into Graphiti and syncs profile to SQLite."""

    def __init__(self, session_store: SessionStoreBase, graphiti_client=None, event_bus=None):
        self._store = session_store
        self._graphiti = graphiti_client
        self._event_bus = event_bus

    @traceable(name="memory.post_turn_tasks", run_type="chain")
    async def post_turn_tasks(
        self,
        session_id: str,
        turn: int,
        user_message: str,
        assistant_response: str,
        user_id: int | None = None,
        **kwargs,
    ) -> None:
        """Ingest the conversation turn into Graphiti.

        Formats the turn as "Parent: ... Coach: ..." and calls
        graphiti.add_episode(). On failure, logs and drops -- the
        conversation text is always in SQLite regardless.
        """
        if not self._graphiti:
            return

        from graphiti_core.nodes import EpisodeType
        from datetime import datetime

        episode_body = f"Parent: {user_message}\nCoach: {assistant_response}"
        group_id = str(user_id) if user_id is not None else session_id

        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                self._graphiti.add_episode(
                    name=f"session_{session_id}_turn_{turn}",
                    episode_body=episode_body,
                    source=EpisodeType.message,
                    source_description=f"ADHD coaching session {session_id}",
                    reference_time=datetime.now(),
                    group_id=group_id,
                    entity_types=ADHD_ENTITY_TYPES,
                    edge_types=ADHD_EDGE_TYPES,
                ),
                timeout=settings.GRAPHITI_INGESTION_TIMEOUT_S,
            )
            duration_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Graphiti episode ingested: session=%s turn=%d (%.0fms)",
                session_id, turn, duration_ms,
            )
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", "graphiti_episode_ingested", session_id, turn,
                    duration_ms=duration_ms,
                )

            # Sync discovered entities to SQLite profile
            if user_id is not None:
                await self._sync_profile(session_id, user_id)

        except Exception as e:
            duration_ms = (time.monotonic() - t0) * 1000
            logger.error(
                "Graphiti ingestion failed: session=%s turn=%d error=%s (%.0fms)",
                session_id, turn, e, duration_ms,
            )
            if self._event_bus:
                await self._event_bus.emit(
                    "memory", "graphiti_ingestion_failed", session_id, turn,
                    detail={"error": str(e)},
                )

    async def _sync_profile(self, session_id: str, user_id: int) -> None:
        """Sync Graphiti-discovered facts into SQLite FamilyProfile.

        One-way: Graphiti -> SQLite. Only fills empty fields.
        Queries Neo4j for Child entity attributes via episode traversal.
        """
        try:
            query = """
            MATCH (e {group_id: $group_id})-[:MENTIONS]->(c)
            WHERE 'Entity' IN labels(c)
            RETURN c.name AS child_name, c.age AS child_age,
                   c.diagnosis_status AS diagnosis_status,
                   c.adhd_subtype AS adhd_subtype,
                   c.summary AS summary
            ORDER BY c.created_at DESC LIMIT 1
            """
            records = await self._graphiti.driver.execute_query(
                query, group_id=str(user_id)
            )
            if not records:
                return

            record = dict(records[0])
            current_profile = (await self._store.get(session_id)).family_profile

            updates = {}
            for field in ("child_name", "child_age", "diagnosis_status", "adhd_subtype"):
                graph_val = record.get(field)
                sqlite_val = getattr(current_profile, field, None)
                if graph_val and not sqlite_val:
                    updates[field] = graph_val

            if updates:
                await self._store.update_profile(session_id, **updates)
                logger.info("Profile synced from graph: session=%s fields=%s",
                            session_id, list(updates.keys()))
                if self._event_bus:
                    await self._event_bus.emit(
                        "memory", "graphiti_profile_synced", session_id, 0,
                        detail={"fields_updated": list(updates.keys())},
                    )
        except Exception as e:
            logger.warning("Profile sync failed (non-critical): %s", e)
