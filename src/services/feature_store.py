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
from src.repositories.online_store import OnlineFeatureStore, OnlineStoreError
from src.repositories.offline_store import OfflineFeatureStore
from src.services.feature_computation import FeatureComputationService
from src.core.metrics import (
    FEATURE_GET_LATENCY, FEATURE_GET_COUNTER,
    FEATURE_CACHE_HIT_RATE, MATERIALIZATION_DURATION,
)

logger = logging.getLogger(__name__)


class FeatureNotFoundError(LookupError):
    """No servable feature vector is stored for this pair and tenant."""

    def __init__(self, entity_id_1: str, entity_id_2: str):
        self.entity_id_1, self.entity_id_2 = entity_id_1, entity_id_2
        super().__init__(
            f"No materialized features for pair ({entity_id_1}, {entity_id_2}). "
            f"Features are produced by materialization, not computed on request."
        )


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
        self._online_store = online_store
        self._offline = offline_store
        self._compute = computation_service
        self._registry = FeatureRegistryRepository(db_session)

    @property
    def _online(self) -> OnlineFeatureStore:
        if self._online_store is None:
            # Redis was unreachable at startup. Handlers used to dereference None and
            # fail with an AttributeError (HTTP 500).
            raise OnlineStoreError("Online store (Redis) is unavailable")
        return self._online_store

    async def get_features(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
    ) -> FeatureVector:
        """
        The MATERIALIZED feature vector of a pair, from the online store.

        Raises FeatureNotFoundError when the store holds no servable vector. There is
        deliberately no compute-on-miss here any more: that path computed from the
        three fields a caller put in the URL (name/email/phone — personal data in
        access logs, and no address or geo fields), cached the partial result for 24
        hours and wrote it to the offline store, so the same pair could have different
        features in training and in serving. Features are produced by materialization.
        """
        start = time.perf_counter()
        cached = await self._online.get(entity_id_1, entity_id_2, tenant_id)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if cached is None:
            FEATURE_GET_COUNTER.labels(status="miss", store="online").inc()
            raise FeatureNotFoundError(entity_id_1, entity_id_2)

        FEATURE_GET_COUNTER.labels(status="hit", store="online").inc()
        FEATURE_GET_LATENCY.labels(store_type="cache").observe(elapsed_ms)
        return cached

    async def get_features_batch(
        self,
        pairs: List[Dict],  # [{"entity_id_1": ..., "entity_id_2": ..., "tenant_id": ...}]
        entity_snapshots: Optional[Dict[str, EntitySnapshot]] = None,
    ) -> List[Optional[FeatureVector]]:
        """
        Batch feature retrieval for up to 100 entity pairs.
        Uses Redis pipeline for minimum round-trips.

        Pairs missing from the online store are computed only when the caller supplied
        both records in the request BODY. Such vectors are returned but NOT stored:
        the stores hold materialized vectors only, so a caller-supplied (possibly
        partial) record can never become the features another request is served.
        """
        if len(pairs) > settings.FEATURE_BATCH_SIZE:
            raise ValueError(f"Batch size {len(pairs)} exceeds maximum {settings.FEATURE_BATCH_SIZE}")

        raw_pairs = [(p["entity_id_1"], p["entity_id_2"], p["tenant_id"]) for p in pairs]
        cached_results = await self._online.get_batch(raw_pairs)

        results: List[Optional[FeatureVector]] = []
        for pair in pairs:
            e1, e2, tid = pair["entity_id_1"], pair["entity_id_2"], pair["tenant_id"]
            fv = cached_results.get(f"{e1}:{e2}:{tid}")
            if fv is None and entity_snapshots:
                e1_snap = entity_snapshots.get(e1)
                e2_snap = entity_snapshots.get(e2)
                if e1_snap and e2_snap:
                    fv = self._compute.compute(e1_snap, e2_snap, settings.FEATURE_VERSION)
            results.append(fv)
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

    async def get_job_status(
        self, job_id: UUID, tenant_id: Optional[str] = None, all_tenants: bool = False,
    ) -> Optional[MaterializationJob]:
        return await self._registry.get_materialization_job(
            job_id, tenant_id=tenant_id, all_tenants=all_tenants)

    async def list_recent_jobs(
        self, limit: int = 10, tenant_id: Optional[str] = None, all_tenants: bool = False,
    ) -> List[MaterializationJob]:
        return await self._registry.list_recent_jobs(
            limit, tenant_id=tenant_id, all_tenants=all_tenants)

    async def find_active_job(self, tenant_id: Optional[str]) -> Optional[MaterializationJob]:
        return await self._registry.find_active_job(tenant_id)

    async def list_feature_definitions(
        self,
        version: Optional[str] = None,
    ):
        return await self._registry.list_features(version=version or settings.FEATURE_VERSION)

    async def get_online_store_stats(self, tenant_id: str) -> Dict:
        return await self._online.get_stats(tenant_id)
