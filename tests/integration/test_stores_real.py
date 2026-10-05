"""
REAL-dependency tests for the online store (Redis), the job lock (PostgreSQL) and the
offline store (S3-compatible endpoint from S3_ENDPOINT_URL).

Each group SKIPS when its dependency is unreachable. Every test uses its own random
tenant ids / S3 prefix and removes what it wrote, so the development stores are left
as they were found.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from src.core.config import settings
from src.domain.feature_catalog import FEATURE_NAMES
from src.domain.models import FeatureVector, OfflineFeatureRequest

pytestmark = pytest.mark.integration


def fv(e1, e2, tenant, version=None, value=0.5, computed_at=None):
    return FeatureVector(e1, e2, tenant, {n: value for n in FEATURE_NAMES},
                         version or settings.FEATURE_VERSION,
                         computed_at=computed_at or datetime.utcnow())


# ───────────────────────────── Redis ─────────────────────────────

@pytest.fixture
async def online():
    from src.repositories.online_store import OnlineFeatureStore
    try:
        store = await OnlineFeatureStore.create()
    except Exception:
        pytest.skip("Redis not reachable")
    tenants = [f"it-{uuid.uuid4()}", f"it-{uuid.uuid4()}"]
    yield store, tenants[0], tenants[1]
    for t in tenants:
        for pattern in (f"feature:{t}:*", f"feature_idx:{t}:*", f"feature_store:stats:{t}"):
            async for key in store._redis.scan_iter(match=pattern, count=500):
                await store._redis.delete(key)
    await store.close()


class TestOnlineStore:
    async def test_roundtrip_and_no_expiry_for_materialized_vectors(self, online):
        store, t1, _ = online
        assert await store.set_batch([fv("A", "B", t1)], ttl_hours=0) == 1
        got = await store.get("B", "A", t1)                     # order-independent
        assert got is not None and sorted(got.features) == FEATURE_NAMES
        assert await store._redis.ttl(store._make_key("A", "B", t1)) == -1   # no expiry
        assert await store._redis.ttl(store._index_key(t1, "A")) == -1

    async def test_ttl_is_applied_when_requested(self, online):
        store, t1, _ = online
        await store.set_batch([fv("A", "B", t1)], ttl_hours=1)
        assert 0 < await store._redis.ttl(store._make_key("A", "B", t1)) <= 3600

    async def test_entity_update_invalidates_all_of_its_pairs(self, online):
        """The fix: invalidate_entity used to remove nothing, ever."""
        store, t1, _ = online
        await store.set_batch([fv("A", "B", t1), fv("A", "C", t1), fv("D", "A", t1), fv("B", "C", t1)],
                              ttl_hours=0)
        removed = await store.invalidate_entity("A", t1)

        assert removed == 3
        assert await store.get("A", "B", t1) is None
        assert await store.get("A", "C", t1) is None
        assert await store.get("A", "D", t1) is None
        assert await store.get("B", "C", t1) is not None        # unrelated pair kept
        assert await store.invalidate_entity("A", t1) == 0      # nothing left

    async def test_invalidation_is_tenant_scoped(self, online):
        store, t1, t2 = online
        await store.set_batch([fv("A", "B", t1), fv("A", "B", t2)], ttl_hours=0)
        assert await store.invalidate_entity("A", t1) == 1
        assert await store.get("A", "B", t1) is None
        assert await store.get("A", "B", t2) is not None        # other tenant untouched

    async def test_same_ids_in_two_tenants_are_separate_vectors(self, online):
        store, t1, t2 = online
        await store.set_batch([fv("A", "B", t1, value=0.1), fv("A", "B", t2, value=0.9)], ttl_hours=0)
        assert (await store.get("A", "B", t1)).features["dom_geo_similarity"] == 0.1
        assert (await store.get("A", "B", t2)).features["dom_geo_similarity"] == 0.9

    async def test_vector_of_another_feature_version_is_not_served(self, online):
        store, t1, _ = online
        await store.set_batch([fv("A", "B", t1, version="v1.0.0")], ttl_hours=0)
        assert await store.get("A", "B", t1) is None
        assert (await store.get_batch([("A", "B", t1)]))[f"A:B:{t1}"] is None

    async def test_non_catalog_vector_is_refused_on_write(self, online):
        store, t1, _ = online
        bad = fv("A", "B", t1)
        bad.features.pop("dom_geo_similarity")
        bad.features["pad_49"] = 0.0
        with pytest.raises(ValueError, match="non-catalog"):
            await store.set_batch([bad], ttl_hours=0)
        assert await store.get("A", "B", t1) is None

    async def test_stats_are_per_tenant(self, online):
        store, t1, t2 = online
        await store.set_batch([fv("A", "B", t1)], ttl_hours=0)
        await store.get("A", "B", t1)            # hit for t1
        await store.get("X", "Y", t1)            # miss for t1
        s1, s2 = await store.get_stats(t1), await store.get_stats(t2)
        assert s1["hit"] == 1 and s1["miss"] == 1
        assert s2.get("hit", 0) == 0 and s2.get("miss", 0) == 0

    async def test_unreachable_redis_is_an_error_not_a_miss(self):
        import redis.asyncio as aioredis
        from src.repositories.online_store import OnlineFeatureStore, OnlineStoreError

        dead = OnlineFeatureStore(aioredis.from_url(
            "redis://127.0.0.1:1/0", socket_connect_timeout=0.3, socket_timeout=0.3))
        with pytest.raises(OnlineStoreError):
            await dead.get("A", "B", "t")
        with pytest.raises(OnlineStoreError):
            await dead.get_batch([("A", "B", "t")])
        with pytest.raises(OnlineStoreError):
            await dead.invalidate_entity("A", "t")
        await dead.close()


# ───────────────────────────── PostgreSQL lock ─────────────────────────────

@pytest.fixture
async def engine():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    eng = create_async_engine(settings.DATABASE_URL)
    try:
        async with eng.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        await eng.dispose()
        pytest.skip("PostgreSQL not reachable")
    yield eng
    await eng.dispose()


class TestPostgresJobLock:
    async def test_one_job_per_tenant_across_connections(self, engine):
        from src.workers.job_lock import PostgresJobLock
        t1, t2 = f"it-{uuid.uuid4()}", f"it-{uuid.uuid4()}"
        a, b = PostgresJobLock(engine), PostgresJobLock(engine)   # two "processes"

        h1 = await a.acquire(t1)
        assert h1 is not None
        assert await b.acquire(t1) is None           # same tenant: rejected
        h2 = await b.acquire(t2)
        assert h2 is not None                        # other tenant: allowed
        assert await b.acquire(None) is None         # all-tenant job: rejected

        await a.release(h1)
        h3 = await b.acquire(t1)
        assert h3 is not None                        # free again
        await b.release(h2)
        await b.release(h3)

    async def test_lock_dies_with_its_connection(self, engine):
        """A crashed process cannot leave a lock behind: closing the session frees it."""
        from src.workers.job_lock import PostgresJobLock
        t1 = f"it-{uuid.uuid4()}"
        lock = PostgresJobLock(engine)
        h = await lock.acquire(t1)
        await h.close()                              # simulate the process dying
        h2 = await lock.acquire(t1)
        assert h2 is not None
        await lock.release(h2)

    async def test_all_tenant_job_excludes_tenant_jobs(self, engine):
        from src.workers.job_lock import PostgresJobLock
        lock = PostgresJobLock(engine)
        everything = await lock.acquire(None)
        if everything is None:
            pytest.skip("a real materialization job is running on this database")
        assert await lock.acquire(f"it-{uuid.uuid4()}") is None
        await lock.release(everything)


class TestRegistryTenantFilter:
    async def test_jobs_are_visible_only_to_their_tenant(self, engine):
        from sqlalchemy import delete
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        from src.domain.models import MaterializationJob
        from src.repositories.feature_registry import FeatureRegistryRepository, MaterializationJobORM

        t1, t2 = f"it-{uuid.uuid4()}", f"it-{uuid.uuid4()}"
        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        j1, j2 = MaterializationJob(tenant_id=t1, triggered_by="it"), MaterializationJob(tenant_id=t2, triggered_by="it")
        async with factory() as s:
            reg = FeatureRegistryRepository(s)
            await reg.save_materialization_job(j1)
            await reg.save_materialization_job(j2)
            await reg.commit()
        try:
            async with factory() as s:
                reg = FeatureRegistryRepository(s)
                assert (await reg.get_materialization_job(j1.job_id, tenant_id=t1)).job_id == j1.job_id
                assert await reg.get_materialization_job(j1.job_id, tenant_id=t2) is None
                assert await reg.get_materialization_job(j1.job_id) is None          # no tenant: nothing
                assert (await reg.get_materialization_job(j1.job_id, all_tenants=True)) is not None

                mine = await reg.list_recent_jobs(50, tenant_id=t1)
                assert [j.job_id for j in mine] == [j1.job_id]
                assert await reg.list_recent_jobs(50) == []                          # no tenant: nothing
                assert j2.job_id not in {j.job_id for j in await reg.list_recent_jobs(50, tenant_id=t1)}

                assert (await reg.find_active_job(t1)).job_id == j1.job_id           # PENDING counts
                assert await reg.find_active_job(f"it-{uuid.uuid4()}") is None
        finally:
            async with factory() as s:
                await s.execute(delete(MaterializationJobORM).where(
                    MaterializationJobORM.tenant_id.in_([t1, t2])))
                await s.commit()


# ───────────────────────────── S3 offline store ─────────────────────────────

@pytest.fixture
def offline(monkeypatch):
    from src.repositories.offline_store import OfflineFeatureStore
    store = OfflineFeatureStore()
    if not store.ping():
        pytest.skip("S3 endpoint not reachable")
    prefix = f"it-offline-{uuid.uuid4().hex[:10]}"
    store._prefix = prefix
    yield store
    paginator = store._s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=store._bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            store._s3.delete_object(Bucket=store._bucket, Key=obj["Key"])


class TestOfflineStore:
    T = "00000000-0000-0000-0000-00000000aaaa"

    def _req(self, pairs, version=None, as_of=None):
        return OfflineFeatureRequest(entity_pairs=pairs, tenant_id=self.T,
                                     as_of_timestamp=as_of or datetime.utcnow() + timedelta(minutes=1),
                                     feature_version=version or settings.FEATURE_VERSION)

    def test_existing_vectors_finds_only_this_source_and_version(self, offline):
        offline.write_features([fv("A", "B", self.T), fv("C", "D", self.T)], source_id="ds-1")
        offline.write_features([fv("E", "F", self.T)], source_id="ds-2")
        got = offline.existing_vectors(self.T, settings.FEATURE_VERSION, "ds-1")
        assert set(got) == {"A:B", "C:D"}
        assert sorted(got["A:B"].features) == FEATURE_NAMES
        assert offline.existing_vectors(self.T, "v9.9.9", "ds-1") == {}
        assert offline.existing_vectors("other-tenant", settings.FEATURE_VERSION, "ds-1") == {}

    def test_point_in_time_read_filters_feature_version(self, offline):
        """Rows of another catalog version used to be returned alongside the requested one."""
        old = datetime.utcnow() - timedelta(hours=2)
        offline.write_features([fv("A", "B", self.T, version="v1.0.0", value=0.1, computed_at=old)])
        offline.write_features([fv("A", "B", self.T, value=0.7)])
        df = offline.read_point_in_time(self._req([("A", "B")]))
        assert len(df) == 1 and df.iloc[0]["feature_version"] == settings.FEATURE_VERSION
        df_old = offline.read_point_in_time(self._req([("A", "B")], version="v1.0.0"))
        assert len(df_old) == 1 and abs(df_old.iloc[0]["feat_dom_geo_similarity"] - 0.1) < 1e-6

    def test_reversed_pair_is_one_pair(self, offline):
        t0 = datetime.utcnow() - timedelta(minutes=5)
        offline.write_features([fv("A", "B", self.T, value=0.2, computed_at=t0)])
        offline.write_features([fv("B", "A", self.T, value=0.8)])
        df = offline.read_point_in_time(self._req([("B", "A")]))
        assert len(df) == 1 and abs(df.iloc[0]["feat_dom_geo_similarity"] - 0.8) < 1e-6

    def test_unreadable_object_fails_the_read(self, offline, monkeypatch):
        """Used to be logged and skipped: fewer rows, no sign anything was missing."""
        from src.repositories.offline_store import OfflineStoreError
        offline.write_features([fv("A", "B", self.T)])

        def broken(key):
            raise ConnectionError("connection reset")

        monkeypatch.setattr(offline, "_read_parquet_from_s3", broken)
        with pytest.raises(OfflineStoreError, match="cannot read offline object"):
            offline.read_point_in_time(self._req([("A", "B")]))

    def test_unreachable_s3_fails_the_listing(self, monkeypatch):
        import boto3
        from botocore.config import Config
        from src.repositories.offline_store import OfflineFeatureStore, OfflineStoreError

        store = OfflineFeatureStore()
        store._s3 = boto3.client(
            "s3", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="x",
            endpoint_url="http://127.0.0.1:1",
            config=Config(connect_timeout=1, read_timeout=1, retries={"max_attempts": 1}))
        with pytest.raises(OfflineStoreError, match="cannot list offline store"):
            store.read_point_in_time(self._req([("A", "B")]))
        assert store.ping() is False
