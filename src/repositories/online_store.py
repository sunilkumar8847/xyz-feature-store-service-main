"""
feature-store-service/src/repositories/online_store.py

Redis-backed online feature store.
Target: <10ms P99 for single feature retrieval.

Keys
  feature:{tenant_id}:{pair_hash}          MessagePack feature vector
  feature_idx:{tenant_id}:{entity_id}      SET of the feature keys this entity is part of
  feature_store:stats:{tenant_id}          HASH hit/miss counters for that tenant

Expiry: the caller decides. Materialized vectors are the authoritative online copy and
are written WITHOUT expiry (MATERIALIZED_FEATURE_TTL_HOURS = 0); they are replaced by the
next materialization or removed by invalidate_entity. Previously every key expired after
24 hours and nothing refreshed them.
"""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import msgpack
import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.core.config import settings
from src.core.metrics import REDIS_CONNECTION_ERRORS
from src.domain.feature_catalog import is_canonical
from src.domain.models import FeatureVector

logger = logging.getLogger(__name__)


class OnlineStoreError(RuntimeError):
    """Redis could not be read. NOT a cache miss — the caller must not treat it as one."""


def _ttl_seconds(hours: Optional[float]) -> Optional[int]:
    return int(hours * 3600) if hours and hours > 0 else None


class OnlineFeatureStore:
    """
    Redis-based online store for low-latency feature retrieval.
    Uses MessagePack for compact binary serialization (faster + smaller than JSON).
    """

    KEY_PREFIX = "feature"
    INDEX_PREFIX = "feature_idx"
    STATS_PREFIX = "feature_store:stats"

    def __init__(self, redis_client: Redis):
        self._redis = redis_client
        self._default_ttl = _ttl_seconds(settings.FEATURE_TTL_HOURS)

    @classmethod
    async def create(cls) -> "OnlineFeatureStore":
        """Factory: create Redis connection and return store."""
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
            socket_connect_timeout=settings.REDIS_CONNECT_TIMEOUT,
            decode_responses=False,  # Binary for msgpack
        )
        # Test connection
        await redis_client.ping()
        logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        return cls(redis_client)

    # ─── Read ────────────────────────────────────────────────────────────────

    async def get(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
    ) -> Optional[FeatureVector]:
        """
        Feature vector for a pair, or None when the store holds no SERVABLE vector:
        no key, a vector of another feature catalog version, or one whose names are
        not the catalog (written by an older build). Raises OnlineStoreError when
        Redis cannot be read — that used to be reported as a miss.
        """
        start = time.perf_counter()
        key = self._make_key(entity_id_1, entity_id_2, tenant_id)

        try:
            raw = await self._redis.get(key)
        except RedisError as e:
            REDIS_CONNECTION_ERRORS.inc()
            logger.error(f"Redis GET error for key {key}: {e}")
            raise OnlineStoreError(f"online store read failed: {type(e).__name__}") from e

        if raw is None:
            await self._record_stats("miss", tenant_id)
            return None

        fv = self._deserialize(msgpack.unpackb(raw, raw=False))
        if not self._servable(fv, tenant_id):
            await self._record_stats("miss", tenant_id)
            return None

        await self._record_stats("hit", tenant_id)
        logger.debug(f"Cache hit for {key} in {(time.perf_counter() - start) * 1000:.2f}ms")
        return fv

    async def get_batch(
        self,
        pairs: List[tuple],  # List of (entity_id_1, entity_id_2, tenant_id)
    ) -> Dict[str, Optional[FeatureVector]]:
        """
        Batch get using Redis pipeline (single round-trip).
        Returns dict: pair_key -> servable FeatureVector or None. Raises OnlineStoreError
        when Redis cannot be read.
        """
        if not pairs:
            return {}

        keys = [self._make_key(e1, e2, tid) for e1, e2, tid in pairs]
        try:
            pipeline = self._redis.pipeline(transaction=False)
            for key in keys:
                pipeline.get(key)
            results = await pipeline.execute()
        except RedisError as e:
            REDIS_CONNECTION_ERRORS.inc()
            logger.error(f"Redis batch GET error: {e}")
            raise OnlineStoreError(f"online store read failed: {type(e).__name__}") from e

        output: Dict[str, Optional[FeatureVector]] = {}
        hits = 0
        for (e1, e2, tid), raw in zip(pairs, results):
            fv = self._deserialize(msgpack.unpackb(raw, raw=False)) if raw is not None else None
            if fv is not None and not self._servable(fv, tid):
                fv = None
            output[f"{e1}:{e2}:{tid}"] = fv
            hits += fv is not None
        logger.debug(f"Batch GET: {hits}/{len(pairs)} hits")
        return output

    @staticmethod
    def _servable(fv: FeatureVector, tenant_id: str) -> bool:
        if fv.tenant_id != tenant_id:
            # Keys are tenant-prefixed, so this can only happen through a hash
            # collision or a corrupted value. Never serve it.
            logger.error("Online vector tenant does not match its key; not served")
            return False
        if fv.feature_version != settings.FEATURE_VERSION:
            return False
        return is_canonical(fv.features.keys())

    # ─── Write ───────────────────────────────────────────────────────────────

    async def set(self, feature_vector: FeatureVector, ttl_hours: Optional[float] = None) -> bool:
        """Store one vector. Returns True on success."""
        return await self.set_batch([feature_vector], ttl_hours=ttl_hours) == 1

    async def set_batch(
        self,
        feature_vectors: List[FeatureVector],
        ttl_hours: Optional[float] = None,
    ) -> int:
        """
        Store vectors and index them under both of their entities (so an entity change
        can find and invalidate them). Returns the number of vectors written; a Redis
        error is reported as 0 and logged — the caller compares the count.

        ttl_hours: None -> FEATURE_TTL_HOURS; 0 -> no expiry.
        Only vectors with exactly the catalog names are accepted.
        """
        if not feature_vectors:
            return 0
        for fv in feature_vectors:
            if not is_canonical(fv.features.keys()):
                raise ValueError(
                    f"refusing to store a non-catalog feature vector for "
                    f"{fv.entity_id_1}:{fv.entity_id_2}"
                )
        ttl = self._default_ttl if ttl_hours is None else _ttl_seconds(ttl_hours)

        try:
            pipeline = self._redis.pipeline(transaction=False)
            for fv in feature_vectors:
                key = self._make_key(fv.entity_id_1, fv.entity_id_2, fv.tenant_id)
                raw = msgpack.packb(self._serialize(fv), use_bin_type=True)
                if ttl:
                    pipeline.setex(key, ttl, raw)
                else:
                    pipeline.set(key, raw)
                for entity_id in (fv.entity_id_1, fv.entity_id_2):
                    idx = self._index_key(fv.tenant_id, entity_id)
                    pipeline.sadd(idx, key)
                    if ttl:
                        pipeline.expire(idx, ttl)
                    else:
                        pipeline.persist(idx)
            results = await pipeline.execute()
        except RedisError as e:
            REDIS_CONNECTION_ERRORS.inc()
            logger.error(f"Redis batch SET error: {e}")
            return 0

        per_vector = len(results) // len(feature_vectors)
        written = sum(1 for i in range(0, len(results), per_vector) if results[i])
        logger.debug(f"Batch SET: {written}/{len(feature_vectors)} succeeded")
        return written

    async def delete(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
    ) -> bool:
        """Invalidate cached features for an entity pair."""
        key = self._make_key(entity_id_1, entity_id_2, tenant_id)
        try:
            pipeline = self._redis.pipeline(transaction=False)
            pipeline.delete(key)
            pipeline.srem(self._index_key(tenant_id, entity_id_1), key)
            pipeline.srem(self._index_key(tenant_id, entity_id_2), key)
            result = await pipeline.execute()
            return result[0] > 0
        except RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    async def invalidate_entity(self, entity_id: str, tenant_id: str) -> int:
        """
        Remove every stored vector the entity is part of, for this tenant only.
        Returns the number of vectors removed. Raises OnlineStoreError if Redis fails —
        a failed invalidation must not look like "nothing to invalidate".

        This used to scan for feature_meta:* keys that no writer ever created, so it
        always removed nothing and stale vectors kept being served.
        """
        idx = self._index_key(tenant_id, entity_id)
        try:
            keys = list(await self._redis.smembers(idx))
            deleted = 0
            if keys:
                deleted = int(await self._redis.delete(*keys))
            await self._redis.delete(idx)
        except RedisError as e:
            REDIS_CONNECTION_ERRORS.inc()
            logger.error(f"Redis error during invalidation: {e}")
            raise OnlineStoreError(f"invalidation failed: {type(e).__name__}") from e
        logger.info(f"Invalidated {deleted} cached vectors for entity {entity_id}")
        return deleted

    # ─── Stats / health ──────────────────────────────────────────────────────

    async def get_stats(self, tenant_id: str) -> Dict:
        """Hit/miss statistics OF ONE TENANT. (There used to be one global counter,
        which every tenant could read.)"""
        try:
            raw = await self._redis.hgetall(self._stats_key(tenant_id))
            stats = {k.decode(): int(v) for k, v in raw.items()}
            total = stats.get("hit", 0) + stats.get("miss", 0)
            stats["hit_rate"] = stats.get("hit", 0) / total if total > 0 else 0.0
            return stats
        except RedisError:
            return {}

    async def ping(self) -> bool:
        """Health check."""
        try:
            result = await self._redis.ping()
            return result is True
        except RedisError:
            return False

    async def close(self):
        await self._redis.aclose()

    # ─── Private Helpers ────────────────────────────────────────────────────────

    def _make_key(self, entity_id_1: str, entity_id_2: str, tenant_id: str) -> str:
        """
        Generate a deterministic, order-independent cache key.
        Sorting ensures feature(A,B) == feature(B,A).
        """
        sorted_ids = sorted([entity_id_1, entity_id_2])
        pair_hash = hashlib.sha256(
            f"{sorted_ids[0]}:{sorted_ids[1]}:{tenant_id}".encode()
        ).hexdigest()[:16]
        return f"{self.KEY_PREFIX}:{tenant_id}:{pair_hash}"

    def _index_key(self, tenant_id: str, entity_id: str) -> str:
        return f"{self.INDEX_PREFIX}:{tenant_id}:{entity_id}"

    def _stats_key(self, tenant_id: str) -> str:
        return f"{self.STATS_PREFIX}:{tenant_id}"

    def _serialize(self, fv: FeatureVector) -> dict:
        """Convert FeatureVector to msgpack-serializable dict."""
        return {
            "e1": fv.entity_id_1,
            "e2": fv.entity_id_2,
            "tid": fv.tenant_id,
            "features": fv.features,
            "version": fv.feature_version,
            "computed_at": fv.computed_at.isoformat(),
            "computation_ms": fv.computation_ms,
        }

    def _deserialize(self, data: dict) -> FeatureVector:
        """Reconstruct FeatureVector from msgpack dict."""
        return FeatureVector(
            entity_id_1=data["e1"],
            entity_id_2=data["e2"],
            tenant_id=data["tid"],
            features=data["features"],
            feature_version=data["version"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
            computation_ms=data.get("computation_ms", 0.0),
        )

    async def _record_stats(self, stat_type: str, tenant_id: str):
        """Increment the tenant's hit/miss counters."""
        try:
            await self._redis.hincrby(self._stats_key(tenant_id), stat_type, 1)
        except RedisError:
            pass  # Stats are best-effort
