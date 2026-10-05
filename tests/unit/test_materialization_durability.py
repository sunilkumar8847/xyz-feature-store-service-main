"""
Materialization durability: idempotency, resume after interruption, one job per tenant,
and no job left RUNNING after a restart.

Dependency type: in-memory fakes for both stores and the registry (they record calls);
the real MaterializationWorker, InProcessJobLock and recover_interrupted_jobs.
"""
from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock

import pytest

from src.adapters.materialization_sources import MaterializationSource, PairBatch
from src.core.config import settings
from src.domain.feature_catalog import FEATURE_NAMES
from src.domain.models import (
    EntitySnapshot, FeatureVector, MaterializationJob, MaterializationStatus,
)
from src.workers.job_lock import InProcessJobLock
from src.workers.materialization import (
    INTERRUPTED_MESSAGE, MaterializationWorker, pair_id_of, recover_interrupted_jobs,
)

T1 = "00000000-0000-0000-0000-000000000001"
T2 = "00000000-0000-0000-0000-000000000002"


def snap(eid, tenant=T1):
    return EntitySnapshot(eid, tenant, "customer", {"name": f"Name {eid}"})


class Source(MaterializationSource):
    name = "fake"

    def __init__(self, batches, source_id="dataset-A"):
        self._batches = batches
        self.source_id = source_id

    def iter_batches(self, tenant_id, lookback_days, batch_size):
        return iter([b for b in self._batches if tenant_id in (None, b.tenant_id)])


def batches(tenant=T1, n=3, size=2):
    out, k = [], 0
    for _ in range(n):
        pairs = []
        for _ in range(size):
            pairs.append((snap(f"E{k}", tenant), snap(f"E{k + 1}", tenant)))
            k += 2
        out.append(PairBatch(tenant, pairs=pairs))
    return out


class Compute:
    def __init__(self):
        self.calls: List[str] = []

    def compute(self, e1, e2, feature_version=None):
        self.calls.append(pair_id_of(e1.entity_id, e2.entity_id))
        return FeatureVector(e1.entity_id, e2.entity_id, e1.tenant_id,
                             {n: 0.5 for n in FEATURE_NAMES}, feature_version or settings.FEATURE_VERSION)


class Offline:
    """Stores what was written, like S3: existing_vectors answers from it."""

    def __init__(self):
        self.objects: List[dict] = []
        self.fail_after_writes = None
        self.fail_scan = False

    def write_features(self, vectors, partition_dt=None, data_source=None, source_id=None):
        if self.fail_after_writes is not None and len(self.objects) >= self.fail_after_writes:
            raise ConnectionError("process killed / S3 lost")
        self.objects.append({"vectors": list(vectors), "source_id": source_id})

    def existing_vectors(self, tenant_id, feature_version, source_id):
        if self.fail_scan:
            raise ConnectionError("S3 unreachable")
        return {
            pair_id_of(fv.entity_id_1, fv.entity_id_2): fv
            for o in self.objects if o["source_id"] == source_id
            for fv in o["vectors"]
            if fv.tenant_id == tenant_id and fv.feature_version == feature_version
        }

    @property
    def rows(self):
        return [fv for o in self.objects for fv in o["vectors"]]


class Online:
    def __init__(self):
        self.store = {}
        self.ttls = []

    async def set_batch(self, vectors, ttl_hours=None):
        self.ttls.append(ttl_hours)
        for fv in vectors:
            self.store[(fv.tenant_id, pair_id_of(fv.entity_id_1, fv.entity_id_2))] = fv
        return len(vectors)


def registry():
    r = AsyncMock()
    r.saved = []

    async def save(job):
        r.saved.append((job.job_id, job.status))
        return job

    r.save_materialization_job = AsyncMock(side_effect=save)
    return r


def worker(source, offline, online, compute=None, lock=None):
    return MaterializationWorker(online, offline, compute or Compute(), registry(), source, lock=lock)


def job(tenant=None):
    return MaterializationJob(triggered_by="test", tenant_id=tenant)


# ─── 8. Idempotency ───────────────────────────────────────────────────────────

class TestIdempotency:
    async def test_second_run_writes_no_duplicate_offline_rows(self):
        offline, online, compute = Offline(), Online(), Compute()
        first = await worker(Source(batches()), offline, online, compute).run_job(job())
        assert first.status == MaterializationStatus.COMPLETED and first.processed_entities == 6
        rows_after_first, calls_after_first = len(offline.rows), len(compute.calls)

        w2 = worker(Source(batches()), offline, online, compute)
        second = await w2.run_job(job())

        assert second.status == MaterializationStatus.COMPLETED
        assert second.processed_entities == 6           # every pair is in both stores
        assert len(offline.rows) == rows_after_first == 6   # ... but nothing was re-written
        assert len(compute.calls) == calls_after_first      # ... and nothing recomputed
        assert w2.reused == 6
        assert len({pair_id_of(f.entity_id_1, f.entity_id_2) for f in offline.rows}) == 6

    async def test_rerun_restores_an_emptied_online_store_without_recomputing(self):
        offline, online, compute = Offline(), Online(), Compute()
        await worker(Source(batches()), offline, online, compute).run_job(job())
        online.store.clear()                                # Redis lost its data
        n_calls = len(compute.calls)

        await worker(Source(batches()), offline, online, compute).run_job(job())
        assert len(online.store) == 6
        assert len(compute.calls) == n_calls

    async def test_a_different_dataset_is_not_treated_as_already_done(self):
        offline, online, compute = Offline(), Online(), Compute()
        await worker(Source(batches(), "dataset-A"), offline, online, compute).run_job(job())
        w = worker(Source(batches(), "dataset-B"), offline, online, compute)
        await w.run_job(job())
        assert w.reused == 0 and len(offline.rows) == 12

    async def test_source_without_identity_always_recomputes(self):
        offline, online, compute = Offline(), Online(), Compute()
        await worker(Source(batches(), None), offline, online, compute).run_job(job())
        await worker(Source(batches(), None), offline, online, compute).run_job(job())
        assert len(compute.calls) == 12

    async def test_reuse_is_per_tenant(self):
        offline, online, compute = Offline(), Online(), Compute()
        await worker(Source(batches(T1)), offline, online, compute).run_job(job(T1))
        w = worker(Source(batches(T2)), offline, online, compute)   # same ids, other tenant
        await w.run_job(job(T2))
        assert w.reused == 0
        assert {fv.tenant_id for fv in offline.rows} == {T1, T2}

    async def test_materialized_vectors_are_written_without_expiry(self):
        offline, online = Offline(), Online()
        await worker(Source(batches()), offline, online).run_job(job())
        assert settings.MATERIALIZED_FEATURE_TTL_HOURS == 0
        assert set(online.ttls) == {0}


# ─── 9. Interruption / resume ─────────────────────────────────────────────────

class TestInterruption:
    async def test_job_restarted_after_a_crash_continues_where_it_stopped(self):
        offline, online, compute = Offline(), Online(), Compute()
        offline.fail_after_writes = 1                        # dies while writing batch 2
        failed = await worker(Source(batches()), offline, online, compute).run_job(job())
        assert failed.status == MaterializationStatus.FAILED
        assert "Offline store write failed" in failed.error_message
        assert len(offline.rows) == 2

        offline.fail_after_writes = None
        compute.calls.clear()
        w = worker(Source(batches()), offline, online, compute)
        resumed = await w.run_job(job())

        assert resumed.status == MaterializationStatus.COMPLETED
        assert w.reused == 2 and len(compute.calls) == 4     # only the missing pairs
        assert len(offline.rows) == 6 and len(online.store) == 6

    async def test_unreadable_offline_store_fails_the_job_instead_of_duplicating(self):
        offline, online = Offline(), Online()
        offline.fail_scan = True
        result = await worker(Source(batches()), offline, online).run_job(job())
        assert result.status == MaterializationStatus.FAILED
        assert "Could not read existing offline vectors" in result.error_message
        assert offline.rows == []

    async def test_stale_running_job_is_closed_when_nobody_holds_its_lock(self):
        stale = MaterializationJob(triggered_by="api:x", tenant_id=T1,
                                   status=MaterializationStatus.RUNNING)
        reg = registry()
        reg.list_unfinished_jobs = AsyncMock(return_value=[stale])
        closed = await recover_interrupted_jobs(reg, InProcessJobLock())

        assert closed == [stale]
        assert stale.status == MaterializationStatus.FAILED
        assert stale.error_message == INTERRUPTED_MESSAGE
        assert stale.completed_at is not None
        reg.commit.assert_awaited()

    async def test_job_that_is_really_running_is_left_alone(self):
        lock = InProcessJobLock()
        handle = await lock.acquire(T1)                      # a live worker holds it
        running = MaterializationJob(triggered_by="api:x", tenant_id=T1,
                                     status=MaterializationStatus.RUNNING)
        reg = registry()
        reg.list_unfinished_jobs = AsyncMock(return_value=[running])
        closed = await recover_interrupted_jobs(reg, lock)

        assert closed == [] and running.status == MaterializationStatus.RUNNING
        await lock.release(handle)

    async def test_recovery_releases_the_lock_it_took(self):
        lock = InProcessJobLock()
        stale = MaterializationJob(triggered_by="x", tenant_id=T1, status=MaterializationStatus.RUNNING)
        reg = registry()
        reg.list_unfinished_jobs = AsyncMock(return_value=[stale])
        await recover_interrupted_jobs(reg, lock)
        assert await lock.acquire(T1) is not None


# ─── 10. One job per tenant ───────────────────────────────────────────────────

class TestConcurrency:
    async def test_second_job_for_the_same_tenant_is_rejected(self):
        lock = InProcessJobLock()
        handle = await lock.acquire(T1)                      # job 1 is running
        offline, online, compute = Offline(), Online(), Compute()
        second = await worker(Source(batches()), offline, online, compute, lock=lock).run_job(job(T1))

        assert second.status == MaterializationStatus.FAILED
        assert "already running" in second.error_message
        assert compute.calls == [] and offline.rows == [] and online.store == {}

        await lock.release(handle)
        third = await worker(Source(batches()), offline, online, compute, lock=lock).run_job(job(T1))
        assert third.status == MaterializationStatus.COMPLETED

    async def test_lock_is_released_after_success_and_after_failure(self):
        lock = InProcessJobLock()
        offline = Offline()
        await worker(Source(batches()), offline, Online(), lock=lock).run_job(job(T1))
        assert (h := await lock.acquire(T1)) is not None
        await lock.release(h)

        offline.fail_after_writes = 0
        await worker(Source(batches(), "other"), offline, Online(), lock=lock).run_job(job(T1))
        assert await lock.acquire(T1) is not None

    async def test_lock_scopes(self):
        lock = InProcessJobLock()
        a = await lock.acquire(T1)
        assert a is not None
        assert await lock.acquire(T1) is None                # same tenant: excluded
        b = await lock.acquire(T2)
        assert b is not None                                 # other tenant: allowed
        assert await lock.acquire(None) is None              # all-tenant job: excluded
        await lock.release(a)
        await lock.release(b)
        everything = await lock.acquire(None)
        assert everything is not None
        assert await lock.acquire(T1) is None                # tenant job excluded by all-tenant job
        await lock.release(everything)
        assert await lock.acquire(T1) is not None
