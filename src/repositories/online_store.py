"""
feature-store-service/src/repositories/online_store.py

Redis-backed online feature store.
Target: <10ms P99 for single feature retrieval.
Key: feature:{tenant_id}:{pair_hash}
Value: MessagePack-encoded feature dict + metadata
TTL: 24 hours (configurable)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional

import msgpack
import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from src.core.config import settings
from src.domain.models import FeatureVector

logger = logging.getLogger(__name__)


class OnlineFeatureStore:
    """
    Redis-based online store for low-latency feature retrieval.
    Uses MessagePack for compact binary serialization (faster + smaller than JSON).
    """

    KEY_PREFIX = "feature"
    METADATA_PREFIX = "feature_meta"
    STATS_KEY = "feature_store:stats"

    def __init__(self, redis_client: Redis):
        self._redis = redis_client
        self._ttl_seconds = settings.FEATURE_TTL_HOURS * 3600

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

    async def get(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
    ) -> Optional[FeatureVector]:
        """
        Retrieve feature vector for entity pair.
        Returns None on cache miss.
        Target: <10ms P99.
        """
        start = time.perf_counter()
        key = self._make_key(entity_id_1, entity_id_2, tenant_id)

        try:
            raw = await self._redis.get(key)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if raw is None:
                await self._record_stats("miss")
                return None

            data = msgpack.unpackb(raw, raw=False)
            fv = self._deserialize(data)

            await self._record_stats("hit")
            logger.debug(f"Cache hit for {key} in {elapsed_ms:.2f}ms")
            return fv

        except RedisError as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(self, feature_vector: FeatureVector) -> bool:
        """
        Store feature vector in Redis with TTL.
        Returns True on success.
        """
        key = self._make_key(
            feature_vector.entity_id_1,
            feature_vector.entity_id_2,
            feature_vector.tenant_id,
        )

        try:
            data = self._serialize(feature_vector)
            raw = msgpack.packb(data, use_bin_type=True)

            await self._redis.setex(key, self._ttl_seconds, raw)
            return True

        except RedisError as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def get_batch(
        self,
        pairs: List[tuple],  # List of (entity_id_1, entity_id_2, tenant_id)
    ) -> Dict[str, Optional[FeatureVector]]:
        """
        Batch get using Redis pipeline (single round-trip).
        Returns dict: pair_key -> FeatureVector or None.
        """
        if not pairs:
            return {}

        keys = [self._make_key(e1, e2, tid) for e1, e2, tid in pairs]

        try:
            pipeline = self._redis.pipeline(transaction=False)
            for key in keys:
                pipeline.get(key)
            results = await pipeline.execute()

            output = {}
            hits = 0
            for (e1, e2, tid), raw in zip(pairs, results):
                pair_key = f"{e1}:{e2}:{tid}"
                if raw is not None:
                    data = msgpack.unpackb(raw, raw=False)
                    output[pair_key] = self._deserialize(data)
                    hits += 1
                else:
                    output[pair_key] = None

            hit_rate = hits / len(pairs) if pairs else 0.0
            logger.debug(f"Batch GET: {hits}/{len(pairs)} hits ({hit_rate:.1%})")
            return output

        except RedisError as e:
            logger.error(f"Redis batch GET error: {e}")
            return {f"{e1}:{e2}:{tid}": None for e1, e2, tid in pairs}

    async def set_batch(self, feature_vectors: List[FeatureVector]) -> int:
        """
        Batch set using Redis pipeline. Returns count of successful writes.
        """
        if not feature_vectors:
            return 0

        try:
            pipeline = self._redis.pipeline(transaction=False)
            for fv in feature_vectors:
                key = self._make_key(fv.entity_id_1, fv.entity_id_2, fv.tenant_id)
                data = self._serialize(fv)
                raw = msgpack.packb(data, use_bin_type=True)
                pipeline.setex(key, self._ttl_seconds, raw)

            results = await pipeline.execute()
            success_count = sum(1 for r in results if r)
            logger.debug(f"Batch SET: {success_count}/{len(feature_vectors)} succeeded")
            return success_count

        except RedisError as e:
            logger.error(f"Redis batch SET error: {e}")
            return 0

    async def delete(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
    ) -> bool:
        """Invalidate cached features for an entity pair."""
        key = self._make_key(entity_id_1, entity_id_2, tenant_id)
        try:
            result = await self._redis.delete(key)
            return result > 0
        except RedisError as e:
            logger.error(f"Redis DELETE error: {e}")
            return False

    async def invalidate_entity(self, entity_id: str, tenant_id: str) -> int:
        """
        Invalidate ALL cached feature vectors for an entity (all pairs).
        Uses SCAN to avoid blocking Redis.
        """
        pattern = f"{self.KEY_PREFIX}:{tenant_id}:*"
        deleted = 0

        try:
            async for key in self._redis.scan_iter(match=pattern, count=100):
                # Check if entity is part of this pair (stored in metadata)
                meta_key = key.decode().replace(self.KEY_PREFIX, self.METADATA_PREFIX, 1)
                meta_raw = await self._redis.get(meta_key)
                if meta_raw:
                    meta = json.loads(meta_raw)
                    if entity_id in (meta.get("e1"), meta.get("e2")):
                        await self._redis.delete(key)
                        await self._redis.delete(meta_key)
                        deleted += 1

            logger.info(f"Invalidated {deleted} cached vectors for entity {entity_id}")
            return deleted

        except RedisError as e:
            logger.error(f"Redis scan error during invalidation: {e}")
            return 0

    async def get_stats(self) -> Dict:
        """Return hit/miss statistics."""
        try:
            raw = await self._redis.hgetall(self.STATS_KEY)
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

    # ─── Private Helpers ────────────────────────────────────────────────

    def _make_key(self, entity_id_1: str, entity_id_2: str, tenant_id: str) -> str:
        """
        Generate a deterministic, order-independent cache key.
        Sorting ensures feature(A,B) == feature(B,A).
        """
        import hashlib
        sorted_ids = sorted([entity_id_1, entity_id_2])
        pair_hash = hashlib.sha256(
            f"{sorted_ids[0]}:{sorted_ids[1]}:{tenant_id}".encode()
        ).hexdigest()[:16]
        return f"{self.KEY_PREFIX}:{tenant_id}:{pair_hash}"

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
        from datetime import datetime
        return FeatureVector(
            entity_id_1=data["e1"],
            entity_id_2=data["e2"],
            tenant_id=data["tid"],
            features=data["features"],
            feature_version=data["version"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
            computation_ms=data.get("computation_ms", 0.0),
        )

    async def _record_stats(self, stat_type: str):
        """Increment hit/miss counters."""
        try:
            await self._redis.hincrby(self.STATS_KEY, stat_type, 1)
        except RedisError:
            pass  # Stats are best-effort
