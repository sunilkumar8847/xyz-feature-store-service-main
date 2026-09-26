"""
feature-store-service/src/services/feature_store.py

Main orchestration service for the Feature Store.
Coordinates online store (Redis), offline store (S3), registry (PostgreSQL),
and feature computation engine.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.domain.models import (
    EntitySnapshot, FeatureVector, MaterializationJob,
    MaterializationStatus, OfflineFeatureRequest,
)
from src.repositories.feature_registry import FeatureRegistryRepository
from src.repositories.online_store import OnlineFeatureStore
from src.repositories.offline_store import OfflineFeatureStore
from src.services.feature_computation import FeatureComputationService
from src.core.metrics import (
    FEATURE_GET_LATENCY, FEATURE_GET_COUNTER,
    FEATURE_CACHE_HIT_RATE, MATERIALIZATION_DURATION,
)

logger = logging.getLogger(__name__)


class FeatureStoreService:
    """
    The primary service interface for the Feature Store.
    Implements the Hexagonal architecture Port: all API handlers call this,
    and it delegates to the appropriate adapters/repositories.
    """

    def __init__(
        self,
        online_store: OnlineFeatureStore,
        offline_store: OfflineFeatureStore,
        computation_service: FeatureComputationService,
        db_session: AsyncSession,
    ):
        self._online = online_store
        self._offline = offline_store
        self._compute = computation_service
        self._registry = FeatureRegistryRepository(db_session)

    async def get_features(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
        entity1: Optional[EntitySnapshot] = None,
        entity2: Optional[EntitySnapshot] = None,
    ) -> FeatureVector:
        """
        Get features for an entity pair.
        Strategy:
          1. Try online store (Redis) → <10ms
          2. If miss: compute on-the-fly → ~50-100ms
          3. Store computed features in online store (async)
        """
        start = time.perf_counter()

        # 1. Try online store
        cached = await self._online.get(entity_id_1, entity_id_2, tenant_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if cached is not None:
            FEATURE_GET_COUNTER.labels(status="hit", store="online").inc()
            FEATURE_GET_LATENCY.labels(store_type="cache").observe(elapsed_ms)
            return cached

        # 2. Cache miss — compute on-the-fly
        FEATURE_GET_COUNTER.labels(status="miss", store="online").inc()

        if entity1 is None or entity2 is None:
            raise ValueError(
                f"Cache miss for ({entity_id_1}, {entity_id_2}) and no entity snapshots provided. "
                f"Pass entity snapshots to compute features on-the-fly."
            )

        fv = self._compute.compute(entity1, entity2, settings.FEATURE_VERSION)

        # 3. Store in online cache (best-effort, non-blocking)
        await self._online.set(fv)

        # 4. Persist to offline store (best-effort)
        try:
            self._offline.write_features([fv])
        except Exception as e:
            logger.warning(f"Offline store write failed (non-critical): {e}")

        total_ms = (time.perf_counter() - start) * 1000
        FEATURE_GET_LATENCY.labels(store_type="compute").observe(total_ms)
        return fv

    async def get_features_batch(
        self,
        pairs: List[Dict],  # [{"entity_id_1": ..., "entity_id_2": ..., "tenant_id": ...}]
        entity_snapshots: Optional[Dict[str, EntitySnapshot]] = None,
    ) -> List[Optional[FeatureVector]]:
        """
        Batch feature retrieval for up to 100 entity pairs.
        Uses Redis pipeline for minimum round-trips.
        """
        if len(pairs) > settings.FEATURE_BATCH_SIZE:
            raise ValueError(f"Batch size {len(pairs)} exceeds maximum {settings.FEATURE_BATCH_SIZE}")

        # Batch get from online store
        raw_pairs = [(p["entity_id_1"], p["entity_id_2"], p["tenant_id"]) for p in pairs]
        cached_results = await self._online.get_batch(raw_pairs)

        results = []
        compute_needed = []

        for pair in pairs:
            e1, e2, tid = pair["entity_id_1"], pair["entity_id_2"], pair["tenant_id"]
            pair_key = f"{e1}:{e2}:{tid}"
            cached = cached_results.get(pair_key)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                compute_needed.append((len(results) - 1, pair))

        # Compute missing features
        if compute_needed and entity_snapshots:
            compute_results = []
            for idx, pair in compute_needed:
                e1_snap = entity_snapshots.get(pair["entity_id_1"])
                e2_snap = entity_snapshots.get(pair["entity_id_2"])
                if e1_snap and e2_snap:
                    fv = self._compute.compute(e1_snap, e2_snap, settings.FEATURE_VERSION)
                    results[idx] = fv
                    compute_results.append(fv)

            # Cache computed features
            if compute_results:
                await self._online.set_batch(compute_results)

        return results

    async def push_features(
        self,
        entity_id: str,
        tenant_id: str,
        entity: EntitySnapshot,
    ) -> int:
        """
        Push updated features for an entity to the online store.
        This invalidates all cached pairs involving this entity,
        then re-computes (lazily, on next request).
        """
        invalidated = await self._online.invalidate_entity(entity_id, tenant_id)
        logger.info(f"Invalidated {invalidated} cached vectors for entity {entity_id}")
        return invalidated

    async def get_offline_features(
        self,
        request: OfflineFeatureRequest,
    ):
        """
        Point-in-time correct feature retrieval for model training.
        Returns a Pandas DataFrame with all 50 feature columns.
        """
        return self._offline.read_point_in_time(request)

    async def trigger_materialization(
        self,
        tenant_id: Optional[str] = None,
        triggered_by: str = "api",
    ) -> MaterializationJob:
        """
        Create and enqueue a materialization job.
        The actual work is done by the materialization worker (background task).
        """
        job = MaterializationJob(
            tenant_id=tenant_id,
            lookback_days=settings.MATERIALIZATION_LOOKBACK_DAYS,
            triggered_by=triggered_by,
        )
        await self._registry.save_materialization_job(job)
        logger.info(f"Materialization job {job.job_id} created by {triggered_by}")
        return job

    async def get_job_status(self, job_id: UUID) -> Optional[MaterializationJob]:
        return await self._registry.get_materialization_job(job_id)

    async def list_recent_jobs(self, limit: int = 10) -> List[MaterializationJob]:
        return await self._registry.list_recent_jobs(limit)

    async def list_feature_definitions(
        self,
        version: Optional[str] = None,
    ):
        return await self._registry.list_features(version=version or settings.FEATURE_VERSION)

    async def get_online_store_stats(self) -> Dict:
        return await self._online.get_stats()
