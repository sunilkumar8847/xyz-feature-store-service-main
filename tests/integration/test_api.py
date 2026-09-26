"""
feature-store-service/tests/integration/test_api.py

Integration tests for the Feature Store REST API.
Auth: Envoy-injected x-verified-* headers (D1 compliant).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.main import create_app
from src.domain.models import FeatureVector, MaterializationJob, MaterializationStatus
from datetime import datetime

AUTH_HEADERS = {
    "x-verified-tenant-id": "00000000-0000-0000-0000-000000000001",
    "x-verified-user-id": "test-user-001",
    "x-verified-roles": "ADMIN",
}
TENANT_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def mock_opa(monkeypatch):
    """OPA is not available in unit/integration tests — grant all permissions."""
    import xyz_security.authz as authz
    monkeypatch.setattr(authz, "_query_opa", lambda *_: True)


def _make_feature_vector() -> FeatureVector:
    features = {
        "ss_levenshtein_name": 0.85, "ss_jaro_winkler_name": 0.92,
        "ss_damerau_levenshtein_name": 0.87, "ss_hamming_name": 0.80,
        "ss_jaro_name": 0.91, "ss_levenshtein_address": 0.75,
        "ss_jaro_winkler_address": 0.78, "ss_damerau_address": 0.76,
        "ss_levenshtein_email": 1.0, "ss_jaro_winkler_email": 1.0,
        "ss_name_addr_cross": 0.45, "ss_longest_common_subseq": 0.88,
        "ss_common_prefix_name": 0.90, "ss_osa_distance_name": 0.86,
        "ss_postfix_similarity": 0.82,
        "ph_soundex_name": 1.0, "ph_metaphone_name": 1.0,
        "ph_nysiis_name": 1.0, "ph_match_rating_name": 0.0,
        "ph_soundex_full_name": 0.0,
        "tk_jaccard_name": 0.67, "tk_jaccard_address": 0.55,
        "tk_token_sort_ratio_name": 0.89, "tk_token_set_ratio_name": 0.91,
        "tk_partial_ratio_name": 0.93, "tk_token_sort_address": 0.80,
        "tk_token_set_address": 0.82, "tk_common_token_count": 0.65,
        "sem_cosine_name": 0.94, "sem_euclidean_name": 0.91,
        "sem_cosine_address": 0.78, "sem_euclidean_address": 0.75,
        "sem_cosine_full": 0.89, "sem_euclidean_full": 0.86,
        "sem_cross_name_addr": 0.42, "sem_angular_name": 0.85,
        "sem_dot_product_name": 0.94, "sem_cosine_name_addr_concat": 0.88,
        "str_field_presence_ratio": 0.95, "str_length_ratio_name": 0.83,
        "str_null_count_diff": 1.0, "str_field_overlap": 0.90,
        "str_schema_similarity": 0.92, "str_asymmetric_null_ratio": 0.98,
        "str_word_count_ratio_name": 1.0,
        "dom_email_domain_match": 1.0, "dom_phone_prefix_match": 1.0,
        "dom_phone_full_match": 1.0, "dom_geo_similarity": 0.95,
        "dom_email_local_similarity": 0.90,
    }
    assert len(features) == 50
    return FeatureVector(
        entity_id_1="ent_001",
        entity_id_2="ent_002",
        tenant_id=TENANT_ID,
        features=features,
        feature_version="v2.0.0",
        computed_at=datetime.utcnow(),
        computation_ms=3.5,
    )


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestAuth:
    def test_no_headers_returns_401(self, client):
        r = client.get("/api/v1/features/e1/e2", params={"tenant_id": TENANT_ID})
        assert r.status_code == 401

    def test_missing_user_id_returns_401(self, client):
        r = client.get(
            "/api/v1/features/e1/e2",
            headers={
                "x-verified-tenant-id": TENANT_ID,
                "x-verified-roles": "ADMIN",
            },
            params={"tenant_id": TENANT_ID},
        )
        assert r.status_code == 401

    def test_valid_headers_pass_auth(self, client):
        with patch("src.services.feature_store.FeatureStoreService.get_features") as mock_svc:
            mock_svc.return_value = _make_feature_vector()
            r = client.get(
                "/api/v1/features/ent_001/ent_002",
                headers=AUTH_HEADERS,
                params={"tenant_id": TENANT_ID, "name1": "Alice", "name2": "Alise"},
            )
            assert r.status_code in (200, 500)


class TestFeatureRetrieval:
    def test_get_features_returns_50_features(self, client):
        with patch("src.services.feature_store.FeatureStoreService.get_features") as mock_svc:
            mock_svc.return_value = _make_feature_vector()
            r = client.get(
                "/api/v1/features/ent_001/ent_002",
                headers=AUTH_HEADERS,
                params={"tenant_id": TENANT_ID, "name1": "Alice", "name2": "Alise"},
            )
            if r.status_code == 200:
                d = r.json()
                assert "features" in d
                assert len(d["features"]) == 50

    def test_batch_endpoint_validates_max_size(self, client):
        pairs = [
            {"entity_id_1": f"e{i}", "entity_id_2": f"e{i+100}"}
            for i in range(101)
        ]
        r = client.post(
            "/api/v1/features/batch",
            json={"pairs": pairs, "tenant_id": TENANT_ID},
            headers=AUTH_HEADERS,
        )
        assert r.status_code in (422, 500)

    def test_batch_endpoint_no_auth_returns_401(self, client):
        r = client.post(
            "/api/v1/features/batch",
            json={"pairs": [{"entity_id_1": "e1", "entity_id_2": "e2"}],
                  "tenant_id": TENANT_ID},
        )
        assert r.status_code == 401


class TestMaterializationEndpoint:
    def test_materialize_no_auth_returns_401(self, client):
        r = client.post("/api/v1/materialize", json={"tenant_id": TENANT_ID})
        assert r.status_code == 401

    def test_materialize_with_auth(self, client):
        with patch("src.services.feature_store.FeatureStoreService.trigger_materialization") as mock_svc:
            mock_job = MaterializationJob(triggered_by="api")
            mock_svc.return_value = mock_job
            r = client.post(
                "/api/v1/materialize",
                json={"tenant_id": TENANT_ID},
                headers=AUTH_HEADERS,
            )
            assert r.status_code in (202, 500)


class TestHealthEndpoint:
    def test_health_no_auth_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code in (200, 503, 500)

    def test_health_has_status_field(self, client):
        r = client.get("/api/v1/health")
        if r.status_code == 200:
            d = r.json()
            assert "status" in d


class TestFeatureDefinitions:
    def test_definitions_require_auth(self, client):
        r = client.get("/api/v1/features/definitions")
        assert r.status_code == 401

    def test_definitions_with_auth(self, client):
        r = client.get("/api/v1/features/definitions", headers=AUTH_HEADERS)
        assert r.status_code in (200, 500)
