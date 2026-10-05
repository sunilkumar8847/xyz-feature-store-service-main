"""
API behaviour added in the "durable, traceable bootstrap lifecycle" phase:

  * job / drift / stats endpoints do not expose another tenant's information
  * features are never computed from URL parameters; a missing vector is 404
  * a second materialization for a tenant with an active job is 409
  * store outages are 503, not "0 pairs found" / "cache miss"

Dependency type: the real FastAPI app and real endpoint + service code. The job
registry is an in-memory fake that implements the SAME tenant-filter contract as
FeatureRegistryRepository (that contract is tested against real PostgreSQL in
test_stores_real.py::TestRegistryTenantFilter); the online store is an in-memory fake.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import src.core.dependencies as deps
from src.api.v1.endpoints import features as endpoints
from src.domain.feature_catalog import FEATURE_NAMES
from src.domain.models import FeatureVector, MaterializationJob, MaterializationStatus
from src.main import create_app
from src.repositories.feature_registry import FeatureRegistryRepository, get_db_session
from src.repositories.online_store import OnlineStoreError

TA = "00000000-0000-0000-0000-00000000000a"
TB = "00000000-0000-0000-0000-00000000000b"


def headers(tenant, roles="ADMIN", platform=False):
    h = {"x-verified-tenant-id": tenant, "x-verified-user-id": "u1", "x-verified-roles": roles}
    if platform:
        h["x-verified-platform"] = "true"
    return h


class FakeOnline:
    def __init__(self):
        self.vectors: Dict[tuple, FeatureVector] = {}
        self.stats_calls: List[str] = []
        self.fail = False

    async def get(self, e1, e2, tenant_id):
        if self.fail:
            raise OnlineStoreError("online store read failed: ConnectionError")
        return self.vectors.get((tenant_id, *sorted((e1, e2))))

    async def get_batch(self, pairs):
        return {f"{a}:{b}:{t}": await self.get(a, b, t) for a, b, t in pairs}

    async def get_stats(self, tenant_id):
        self.stats_calls.append(tenant_id)
        return {"hit": 1, "miss": 0, "hit_rate": 1.0, "tenant": tenant_id}

    async def invalidate_entity(self, entity_id, tenant_id):
        return 0

    async def ping(self):
        return True


JOBS: List[MaterializationJob] = []


async def _get_job(self, job_id, tenant_id=None, all_tenants=False):
    for j in JOBS:
        if j.job_id == job_id and (all_tenants or (tenant_id is not None and j.tenant_id == tenant_id)):
            return j
    return None


async def _list_jobs(self, limit=10, tenant_id=None, all_tenants=False):
    return [j for j in JOBS if all_tenants or (tenant_id is not None and j.tenant_id == tenant_id)][:limit]


async def _find_active(self, tenant_id):
    for j in JOBS:
        if j.status in (MaterializationStatus.PENDING, MaterializationStatus.RUNNING) \
                and (tenant_id is None or j.tenant_id in (tenant_id, None)):
            return j
    return None


async def _save(self, job):
    if job not in JOBS:
        JOBS.append(job)
    return job


@pytest.fixture
def client(monkeypatch):
    import xyz_security.authz as authz
    monkeypatch.setattr(authz, "_query_opa", lambda *_: True)

    JOBS.clear()
    online = FakeOnline()
    monkeypatch.setattr(deps, "online_store_instance", online)
    from src.services.feature_computation import FeatureComputationService
    monkeypatch.setattr(deps, "computation_service_instance", FeatureComputationService())
    monkeypatch.setattr(FeatureRegistryRepository, "get_materialization_job", _get_job)
    monkeypatch.setattr(FeatureRegistryRepository, "list_recent_jobs", _list_jobs)
    monkeypatch.setattr(FeatureRegistryRepository, "find_active_job", _find_active)
    monkeypatch.setattr(FeatureRegistryRepository, "save_materialization_job", _save)
    monkeypatch.setattr(endpoints, "run_materialization_job", AsyncMock())

    app = create_app()

    async def no_db():
        yield AsyncMock()

    app.dependency_overrides[get_db_session] = no_db
    c = TestClient(app, raise_server_exceptions=False)
    c.online = online
    return c


def vector(tenant, a="E1", b="E2"):
    return FeatureVector(a, b, tenant, {n: 0.5 for n in FEATURE_NAMES}, "v2.0.0")


# ─── tenant isolation on job / admin endpoints ────────────────────────────────

class TestJobEndpointsAreTenantScoped:
    def test_other_tenants_job_is_404(self, client):
        job = MaterializationJob(tenant_id=TA, triggered_by="api:u1")
        JOBS.append(job)
        assert client.get(f"/api/v1/materialize/{job.job_id}", headers=headers(TA)).status_code == 200
        r = client.get(f"/api/v1/materialize/{job.job_id}", headers=headers(TB))
        assert r.status_code == 404
        assert TA not in r.text

    def test_job_list_contains_only_the_callers_jobs(self, client):
        ja = MaterializationJob(tenant_id=TA, triggered_by="api:u1")
        jb = MaterializationJob(tenant_id=TB, triggered_by="api:u2")
        JOBS.extend([ja, jb, MaterializationJob(tenant_id=None, triggered_by="scheduler")])

        got_a = client.get("/api/v1/materialize", headers=headers(TA)).json()
        got_b = client.get("/api/v1/materialize", headers=headers(TB)).json()
        assert [j["job_id"] for j in got_a] == [str(ja.job_id)]
        assert [j["job_id"] for j in got_b] == [str(jb.job_id)]
        assert all(j["tenant_id"] == TA for j in got_a)

    def test_all_tenant_scheduler_job_is_hidden_from_tenants(self, client):
        sched = MaterializationJob(tenant_id=None, triggered_by="scheduler")
        JOBS.append(sched)
        assert client.get(f"/api/v1/materialize/{sched.job_id}", headers=headers(TA)).status_code == 404

    def test_platform_operator_sees_all_jobs(self, client):
        JOBS.extend([MaterializationJob(tenant_id=TA), MaterializationJob(tenant_id=TB)])
        for h in (headers(TA, roles="PLATFORM_ADMIN"), headers(TA, platform=True)):
            assert len(client.get("/api/v1/materialize", headers=h).json()) == 2

    def test_new_job_is_always_created_for_the_callers_tenant(self, client):
        r = client.post("/api/v1/materialize", json={"tenant_id": TB}, headers=headers(TA))
        assert r.status_code == 202
        assert r.json()["tenant_id"] == TA            # body tenant ignored; header tenant used

    def test_drift_reports_are_platform_only(self, client):
        assert client.get("/api/v1/drift", headers=headers(TA)).status_code == 403
        with patch.object(FeatureRegistryRepository, "get_latest_drift_reports",
                          new=AsyncMock(return_value=[])):
            assert client.get("/api/v1/drift", headers=headers(TA, roles="PLATFORM_ADMIN")).status_code == 200

    def test_stats_are_the_callers_own(self, client):
        ra = client.get("/api/v1/stats", headers=headers(TA)).json()
        rb = client.get("/api/v1/stats", headers=headers(TB)).json()
        assert ra["online_store"]["tenant"] == TA and rb["online_store"]["tenant"] == TB
        assert client.online.stats_calls == [TA, TB]

    def test_features_of_another_tenant_are_not_served(self, client):
        client.online.vectors[(TA, "E1", "E2")] = vector(TA)
        assert client.get("/api/v1/features/E1/E2", headers=headers(TA)).status_code == 200
        assert client.get("/api/v1/features/E1/E2", headers=headers(TB)).status_code == 404


# ─── one active job per tenant ────────────────────────────────────────────────

class TestConcurrentMaterialization:
    def test_second_request_for_the_same_tenant_is_409(self, client):
        first = client.post("/api/v1/materialize", json={}, headers=headers(TA))
        assert first.status_code == 202
        second = client.post("/api/v1/materialize", json={}, headers=headers(TA))
        assert second.status_code == 409
        assert first.json()["job_id"] in second.json()["detail"]
        assert len([j for j in JOBS if j.tenant_id == TA]) == 1

    def test_other_tenant_is_not_blocked_and_learns_nothing(self, client):
        client.post("/api/v1/materialize", json={}, headers=headers(TA))
        assert client.post("/api/v1/materialize", json={}, headers=headers(TB)).status_code == 202

    def test_all_tenant_job_blocks_without_leaking_its_id(self, client):
        sched = MaterializationJob(tenant_id=None, triggered_by="scheduler",
                                   status=MaterializationStatus.RUNNING)
        JOBS.append(sched)
        r = client.post("/api/v1/materialize", json={}, headers=headers(TA))
        assert r.status_code == 409 and str(sched.job_id) not in r.text


# ─── no compute-on-miss, no personal data in URLs ─────────────────────────────

class TestCacheMissPath:
    def test_missing_vector_is_404_not_computed(self, client):
        r = client.get("/api/v1/features/E1/E2", headers=headers(TA))
        assert r.status_code == 404
        assert "materialization" in r.json()["detail"]

    @pytest.mark.parametrize("param", ["name1", "name2", "email1", "email2", "phone1", "phone2"])
    def test_record_fields_in_the_url_are_rejected(self, client, param):
        client.online.vectors[(TA, "E1", "E2")] = vector(TA)
        r = client.get("/api/v1/features/E1/E2", params={param: "x"}, headers=headers(TA))
        assert r.status_code == 422
        assert "query string" in r.json()["detail"]

    def test_stored_vector_is_served_with_50_ordered_features(self, client):
        client.online.vectors[(TA, "E1", "E2")] = vector(TA)
        body = client.get("/api/v1/features/E1/E2", headers=headers(TA)).json()
        assert sorted(body["features"]) == FEATURE_NAMES
        assert len(body["feature_vector"]) == 50 and body["tenant_id"] == TA
        assert body["version"] == "v2.0.0"

    def test_batch_computed_vectors_are_not_stored(self, client):
        """Body-supplied records may be scored, but never become stored features."""
        fields = {"name": "Ada Lovelace", "email": "ada@example.com"}
        r = client.post("/api/v1/features/batch", headers=headers(TA), json={
            "include_entity_data": True, "tenant_id": TA,
            "pairs": [{"entity_id_1": "E1", "entity_id_2": "E2",
                       "entity1_fields": fields, "entity2_fields": fields}],
        })
        assert r.status_code == 200, r.text
        assert r.json()["pairs_found"] == 1
        assert client.online.vectors == {}            # nothing written to the online store
        assert client.get("/api/v1/features/E1/E2", headers=headers(TA)).status_code == 404


# ─── outages are visible ──────────────────────────────────────────────────────

class TestOutages:
    def test_redis_error_is_503_not_a_miss(self, client):
        client.online.fail = True
        assert client.get("/api/v1/features/E1/E2", headers=headers(TA)).status_code == 503

    def test_no_redis_at_startup_is_503_not_500(self, client, monkeypatch):
        monkeypatch.setattr(deps, "online_store_instance", None)
        assert client.get("/api/v1/features/E1/E2", headers=headers(TA)).status_code == 503
        assert client.get("/api/v1/stats", headers=headers(TA)).status_code == 503

    def test_offline_store_failure_is_503_not_zero_pairs(self, client):
        from src.repositories.offline_store import OfflineStoreError
        with patch("src.services.feature_store.FeatureStoreService.get_offline_features",
                   side_effect=OfflineStoreError("cannot list offline store: EndpointConnectionError")):
            r = client.post("/api/v1/features/offline", headers=headers(TA), json={
                "entity_pairs": [["A", "B"]], "tenant_id": TA, "as_of_timestamp": "2026-10-05T00:00:00",
                "feature_version": "v2.0.0"})
        assert r.status_code == 503 and "pairs_found" not in r.text

    def test_health_reports_embedding_and_s3_for_real(self, client, monkeypatch):
        import src.services.feature_computation as fc
        from src.repositories.offline_store import OfflineFeatureStore

        monkeypatch.setattr(fc, "_embedding_model", None)
        monkeypatch.setattr(OfflineFeatureStore, "ping", lambda self: False)

        async def ok_db():
            db = AsyncMock()
            yield db

        client.app.dependency_overrides[get_db_session] = ok_db
        body = client.get("/api/v1/health").json()
        assert body["checks"]["embedding_model"] is False
        assert body["checks"]["s3"] is False
        assert body["status"] != "healthy"
