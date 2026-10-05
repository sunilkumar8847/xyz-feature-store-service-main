"""
Unit tests for MaterializationWorker's job contract.

Dependency type:
  - Online/offline stores and the job registry: TEST DOUBLES (in-memory fakes that
    record calls and can be told to fail).
  - Feature computation: the REAL FeatureComputationService where the 50-feature
    contract is under test; a stub where only failure handling is under test.
No network, no S3, no Redis, no database.
"""
from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock

import pytest

from src.adapters.materialization_sources import (
    MaterializationSource, PairBatch, SkippedPair,
)
from src.core.metrics import MATERIALIZATION_RECORDS
from src.domain.models import (
    EntitySnapshot, FeatureVector, MaterializationJob, MaterializationStatus,
)
from src.repositories.offline_store import FEATURE_COLUMN_NAMES
from src.services.feature_computation import FeatureComputationService
from src.workers.materialization import MaterializationWorker

T1 = "00000000-0000-0000-0000-000000000001"
T2 = "00000000-0000-0000-0000-000000000002"


# ─── Test doubles ─────────────────────────────────────────────────────────────

def snap(eid: str, tenant: str, name: str) -> EntitySnapshot:
    return EntitySnapshot(eid, tenant, "customer", {
        "name": name, "email": f"{eid.lower()}@example.com", "phone": "2065551234",
    })


class FakeSource(MaterializationSource):
    name = "fake"

    def __init__(self, batches: List[PairBatch]):
        self._batches = batches
        self.calls = []

    def iter_batches(self, tenant_id, lookback_days, batch_size):
        self.calls.append((tenant_id, lookback_days, batch_size))
        return iter(self._batches)


class FakeOffline:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.writes: List[dict] = []

    def write_features(self, vectors, partition_dt=None, data_source=None, source_id=None):
        if self.fail:
            raise ConnectionError("S3 endpoint unreachable")
        self.writes.append({"vectors": list(vectors), "data_source": data_source,
                            "source_id": source_id})
        return "s3://bucket/key.parquet"

    def existing_vectors(self, tenant_id, feature_version, source_id):
        """What a real offline store answers: the latest stored vector per pair that
        was written for this tenant, version and source_id."""
        self.scans = getattr(self, "scans", 0) + 1
        if getattr(self, "fail_scan", False):
            raise ConnectionError("S3 endpoint unreachable")
        out = {}
        for w in self.writes:
            if w["source_id"] != source_id:
                continue
            for fv in w["vectors"]:
                if fv.tenant_id == tenant_id and fv.feature_version == feature_version:
                    a, b = sorted((fv.entity_id_1, fv.entity_id_2))
                    out[f"{a}:{b}"] = fv
        return out


class FakeOnline:
    def __init__(self, short_by: int = 0):
        self.short_by = short_by
        self.writes: List[List[FeatureVector]] = []

    async def set_batch(self, vectors, ttl_hours=None):
        self.writes.append(list(vectors))
        self.ttl_hours = ttl_hours
        return len(vectors) - self.short_by


class StubCompute:
    """Returns a canonical 50-feature vector; can fail for chosen entity ids."""

    def __init__(self, fail_ids=(), bad_names_ids=()):
        self.fail_ids = set(fail_ids)
        self.bad_names_ids = set(bad_names_ids)

    def compute(self, e1, e2, feature_version="v2.0.0"):
        if e1.entity_id in self.fail_ids:
            raise ValueError("boom")
        names = list(FEATURE_COLUMN_NAMES)
        if e1.entity_id in self.bad_names_ids:
            # What the group-level fallback produces: 50 values, wrong names.
            names = [n for n in names if not n.startswith("ph_")] + [f"ph_err_{i}" for i in range(5)]
        return FeatureVector(e1.entity_id, e2.entity_id, e1.tenant_id,
                             {n: 0.5 for n in names}, feature_version)


def registry():
    r = AsyncMock()
    r.save_materialization_job = AsyncMock()
    r.commit = AsyncMock()
    return r


def worker(source, compute=None, offline=None, online=None, reg=None):
    return MaterializationWorker(
        online_store=online or FakeOnline(),
        offline_store=offline or FakeOffline(),
        computation_service=compute or StubCompute(),
        registry=reg or registry(),
        source=source,
    )


def batch(tenant, *pairs, skipped=()):
    return PairBatch(
        tenant_id=tenant,
        pairs=[(snap(a, tenant, f"Name {a}"), snap(b, tenant, f"Name {b}")) for a, b in pairs],
        skipped=list(skipped),
    )


def job(**kw):
    return MaterializationJob(triggered_by="test", **kw)


# ─── Happy path ───────────────────────────────────────────────────────────────

class TestSuccessfulJob:
    async def test_writes_every_vector_to_both_stores(self):
        offline, online = FakeOffline(), FakeOnline()
        src = FakeSource([batch(T1, ("A1", "A2"), ("A3", "A4"))])
        result = await worker(src, offline=offline, online=online).run_job(job())

        assert result.status == MaterializationStatus.COMPLETED
        assert result.error_message is None
        assert result.processed_entities == 2
        assert len(offline.writes) == 1 and len(offline.writes[0]["vectors"]) == 2
        assert len(online.writes) == 1 and len(online.writes[0]) == 2

    async def test_offline_write_records_the_data_source(self):
        offline = FakeOffline()
        await worker(FakeSource([batch(T1, ("A1", "A2"))]), offline=offline).run_job(job())
        assert offline.writes[0]["data_source"] == "fake"

    async def test_source_receives_job_tenant_and_configured_batch_size(self):
        from src.core.config import settings
        src = FakeSource([batch(T2, ("B1", "B2"))])
        await worker(src).run_job(job(tenant_id=T2, lookback_days=7))
        assert src.calls == [(T2, 7, settings.MATERIALIZATION_BATCH_SIZE)]

    async def test_real_computation_yields_exactly_50_canonical_features(self):
        """REAL FeatureComputationService: vectors reaching the stores are complete."""
        offline = FakeOffline()
        src = FakeSource([PairBatch(T1, pairs=[
            (snap("A1", T1, "Patricia Cooper"), snap("A2", T1, "Patrícia Coóper")),
            (snap("A3", T1, "Richárd Nguyen"), snap("A4", T1, "Richard Nguyen")),
        ])])
        result = await worker(src, compute=FeatureComputationService(), offline=offline).run_job(job())

        assert result.status == MaterializationStatus.COMPLETED
        for fv in offline.writes[0]["vectors"]:
            assert len(fv.features) == 50
            assert sorted(fv.features) == sorted(FEATURE_COLUMN_NAMES)
            assert fv.feature_version == "v2.0.0"

    async def test_job_state_is_persisted_and_committed(self):
        reg = registry()
        await worker(FakeSource([batch(T1, ("A1", "A2"))]), reg=reg).run_job(job())
        # start, after the batch, and final
        assert reg.save_materialization_job.await_count >= 3
        assert reg.commit.await_count == reg.save_materialization_job.await_count


# ─── 11. Computation failures are counted ────────────────────────────────────

class TestComputationFailures:
    async def test_failure_is_counted_and_not_written(self):
        offline = FakeOffline()
        src = FakeSource([batch(T1, ("A1", "A2"), ("BAD", "A3"))])
        result = await worker(src, compute=StubCompute(fail_ids={"BAD"}), offline=offline).run_job(job())

        assert result.status == MaterializationStatus.COMPLETED
        assert (result.processed_entities, result.failed_entities) == (1, 1)
        assert [fv.entity_id_1 for fv in offline.writes[0]["vectors"]] == ["A1"]

    async def test_non_canonical_feature_names_never_stored(self):
        """A 50-value vector with ph_err_* names has the wrong layout; zero-filling
        or storing it would corrupt training rows."""
        offline = FakeOffline()
        src = FakeSource([batch(T1, ("ERR", "A2"), ("A3", "A4"))])
        result = await worker(src, compute=StubCompute(bad_names_ids={"ERR"}), offline=offline).run_job(job())
        assert result.failed_entities == 1
        assert [fv.entity_id_1 for fv in offline.writes[0]["vectors"]] == ["A3"]

    async def test_all_pairs_failing_is_a_failed_job(self):
        src = FakeSource([batch(T1, ("BAD", "A2"))])
        result = await worker(src, compute=StubCompute(fail_ids={"BAD"})).run_job(job())
        assert result.status == MaterializationStatus.FAILED
        assert "No pairs were materialized" in result.error_message


# ─── 12. Store failures never produce a false success ────────────────────────

class TestStoreFailures:
    async def test_offline_write_failure_fails_the_job(self):
        online = FakeOnline()
        result = await worker(
            FakeSource([batch(T1, ("A1", "A2"))]), offline=FakeOffline(fail=True), online=online,
        ).run_job(job())
        assert result.status == MaterializationStatus.FAILED
        assert "Offline store write failed" in result.error_message
        assert "S3 endpoint unreachable" in result.error_message
        assert result.processed_entities == 0
        # Offline is written first and is authoritative: Redis is not touched.
        assert online.writes == []

    async def test_offline_failure_mid_job_stops_and_reports_progress(self):
        class FailSecond(FakeOffline):
            def write_features(self, vectors, partition_dt=None, data_source=None, source_id=None):
                if self.writes:
                    raise ConnectionError("lost connection")
                return super().write_features(vectors, partition_dt, data_source, source_id)

        src = FakeSource([batch(T1, ("A1", "A2")), batch(T1, ("A3", "A4")), batch(T1, ("A5", "A6"))])
        result = await worker(src, offline=FailSecond()).run_job(job())
        assert result.status == MaterializationStatus.FAILED
        assert result.processed_entities == 1
        assert "1 written earlier" in result.error_message

    async def test_online_short_write_fails_the_job(self):
        """set_batch reports Redis errors as a short count, not an exception."""
        result = await worker(
            FakeSource([batch(T1, ("A1", "A2"), ("A3", "A4"))]), online=FakeOnline(short_by=1),
        ).run_job(job())
        assert result.status == MaterializationStatus.FAILED
        assert "Online store wrote 1/2" in result.error_message

    async def test_missing_source_is_a_failed_job_not_an_empty_success(self):
        """Previously _load_entity_pairs returned iter([]) and every job COMPLETED."""
        result = await worker(None).run_job(job())
        assert result.status == MaterializationStatus.FAILED
        assert "No materialization data source" in result.error_message

    async def test_final_state_persisted_even_on_failure(self):
        reg = registry()
        await worker(FakeSource([batch(T1, ("A1", "A2"))]),
                     offline=FakeOffline(fail=True), reg=reg).run_job(job())
        final = reg.save_materialization_job.await_args_list[-1].args[0]
        assert final.status == MaterializationStatus.FAILED
        assert reg.commit.await_count >= 1


# ─── 10. Skipped pairs are counted ────────────────────────────────────────────

class TestSkippedPairs:
    async def test_skipped_pairs_counted_separately(self):
        sk = [SkippedPair("A1", "GHOST", T1, "entity not found")]
        result = await worker(FakeSource([batch(T1, ("A1", "A2"), skipped=sk)])).run_job(job())
        assert result.status == MaterializationStatus.COMPLETED
        assert (result.processed_entities, result.failed_entities, result.skipped_entities) == (1, 0, 1)
        assert result.total_entities == 2

    async def test_only_skipped_pairs_is_a_failed_job(self):
        sk = [SkippedPair("X", "Y", T1, "entity not found")]
        result = await worker(FakeSource([PairBatch(T1, skipped=sk)])).run_job(job())
        assert result.status == MaterializationStatus.FAILED


# ─── 13. Tenant isolation ─────────────────────────────────────────────────────

class TestTenantIsolation:
    async def test_each_offline_write_is_single_tenant(self):
        offline = FakeOffline()
        src = FakeSource([batch(T1, ("A1", "A2")), batch(T2, ("B1", "B2"))])
        await worker(src, offline=offline).run_job(job())
        assert [{fv.tenant_id for fv in w["vectors"]} for w in offline.writes] == [{T1}, {T2}]

    async def test_cross_tenant_pair_in_a_batch_is_refused(self):
        """Defence in depth: even if a source misbehaves, nothing is mis-filed."""
        offline = FakeOffline()
        bad = PairBatch(T1, pairs=[
            (snap("A1", T1, "Ann"), snap("B1", T2, "Bob")),
            (snap("A2", T1, "Ann"), snap("A3", T1, "Anne")),
        ])
        result = await worker(FakeSource([bad]), offline=offline).run_job(job())
        assert result.failed_entities == 1
        assert [fv.entity_id_1 for fv in offline.writes[0]["vectors"]] == ["A2"]
        assert all(fv.tenant_id == T1 for fv in offline.writes[0]["vectors"])


# ─── Metrics ──────────────────────────────────────────────────────────────────

class TestMetrics:
    async def test_success_counter_counts_each_pair_once(self):
        """Regression: the old loop re-added running totals every batch."""
        counter = MATERIALIZATION_RECORDS.labels(status="success")
        before = counter._value.get()
        src = FakeSource([batch(T1, ("A1", "A2")), batch(T1, ("A3", "A4")), batch(T1, ("A5", "A6"))])
        await worker(src).run_job(job())
        assert counter._value.get() - before == 3


# ─── run_materialization_job: the API + scheduler entry point ────────────────

class _FakeSessionFactory:
    """Stands in for get_session_factory(): an async context manager session."""

    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def runner_env(monkeypatch):
    """Patch the runner's session + registry so it runs without a database."""
    import src.repositories.feature_registry as fr
    import src.workers.materialization as wm

    reg = registry()
    monkeypatch.setattr(fr, "get_session_factory", lambda: _FakeSessionFactory())
    monkeypatch.setattr(wm, "FeatureRegistryRepository", lambda _session: reg)
    return reg


class TestRunner:
    async def test_unconfigured_source_is_a_persisted_failed_job(self, runner_env, monkeypatch):
        from src.core.config import settings
        from src.workers.materialization import run_materialization_job

        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", None)
        result = await run_materialization_job(job(), FakeOnline(), StubCompute())

        assert result.status == MaterializationStatus.FAILED
        assert "not implemented yet" in result.error_message
        runner_env.save_materialization_job.assert_awaited()
        runner_env.commit.assert_awaited()

    async def test_invalid_synthetic_dir_is_a_failed_job(self, runner_env, monkeypatch, tmp_path):
        from src.core.config import Environment, settings
        from src.workers.materialization import run_materialization_job

        monkeypatch.setattr(settings, "ENVIRONMENT", Environment.DEVELOPMENT)
        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", "synthetic")
        monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(tmp_path / "missing"))
        result = await run_materialization_job(job(), FakeOnline(), StubCompute())
        assert result.status == MaterializationStatus.FAILED
        assert "not found" in result.error_message

    async def test_unavailable_online_store_is_a_failed_job(self, runner_env):
        from src.workers.materialization import run_materialization_job

        result = await run_materialization_job(job(), None, StubCompute())
        assert result.status == MaterializationStatus.FAILED
        assert "Online store" in result.error_message
