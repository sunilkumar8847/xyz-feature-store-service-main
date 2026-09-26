"""
feature-store-service/tests/integration/test_offline_feature_parity.py

Contract tests for POST /api/v1/features/offline — the training-time feature source.

Dependency type: TEST DOUBLE. The offline store (S3/Parquet) is patched with an
in-memory DataFrame shaped exactly like FEATURE_SCHEMA. No S3/LocalStack required.

These guard two properties the training pipeline depends on:
  1. The endpoint returns actual feature vectors, not just counts.
  2. Offline (training) ordering is identical to online (serving) ordering.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.domain.models import FeatureVector
from src.main import create_app
from src.repositories.offline_store import FEATURE_COLUMN_NAMES

AUTH_HEADERS = {
    "x-verified-tenant-id": "00000000-0000-0000-0000-000000000001",
    "x-verified-user-id": "svc-model-training",
    "x-verified-roles": "PLATFORM_ADMIN",
}
TENANT_ID = "00000000-0000-0000-0000-000000000001"
AS_OF = "2026-09-17T00:00:00Z"


@pytest.fixture(autouse=True)
def mock_opa(monkeypatch):
    """OPA is not available in tests — grant all permissions."""
    import xyz_security.authz as authz
    monkeypatch.setattr(authz, "_query_opa", lambda *_: True)


@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def _offline_dataframe() -> pd.DataFrame:
    """One point-in-time row per FEATURE_SCHEMA, with a distinct value per feature."""
    computed_at = datetime(2026, 9, 16, 12, 0, 0)
    row = {
        "entity_id_1": "CUST_001",
        "entity_id_2": "CUST_002",
        "tenant_id": TENANT_ID,
        "pair_id": "CUST_001:CUST_002",
        "feature_version": "v2.0.0",
        "computed_at": computed_at,
        "computation_ms": 3.5,
    }
    # Distinct, position-dependent values so a re-ordering bug cannot pass silently.
    for i, name in enumerate(FEATURE_COLUMN_NAMES):
        row[f"feat_{name}"] = i / 100.0
    return pd.DataFrame([row])


def _post_offline(client, pairs=None):
    return client.post(
        "/api/v1/features/offline",
        headers=AUTH_HEADERS,
        json={
            "entity_pairs": pairs or [["CUST_001", "CUST_002"]],
            "tenant_id": TENANT_ID,
            "as_of_timestamp": AS_OF,
            "feature_version": "v2.0.0",
        },
    )


class TestOfflineReturnsRealFeatures:
    """Previously this endpoint returned only counts, so training could not use it."""

    def test_returns_actual_feature_vectors(self, client):
        with patch(
            "src.services.feature_store.FeatureStoreService.get_offline_features",
            return_value=_offline_dataframe(),
        ):
            r = _post_offline(client)

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["pairs_found"] == 1
        assert len(body["results"]) == 1

        result = body["results"][0]
        assert result["entity_id_1"] == "CUST_001"
        assert result["entity_id_2"] == "CUST_002"
        assert len(result["feature_vector"]) == 50
        assert all(isinstance(v, float) for v in result["feature_vector"])

    def test_vector_values_follow_declared_feature_names(self, client):
        """feature_names must describe feature_vector position-for-position."""
        with patch(
            "src.services.feature_store.FeatureStoreService.get_offline_features",
            return_value=_offline_dataframe(),
        ):
            body = _post_offline(client).json()

        names = body["feature_names"]
        vector = body["results"][0]["feature_vector"]
        assert len(names) == 50

        # The fixture set feat_<name> = index/100 over FEATURE_COLUMN_NAMES.
        for position, name in enumerate(names):
            expected = FEATURE_COLUMN_NAMES.index(name) / 100.0
            assert vector[position] == pytest.approx(expected), (
                f"value at position {position} does not match declared name '{name}'"
            )

    def test_empty_result_is_reported_honestly(self, client):
        """No history for the pair must yield 0 found — not fabricated defaults."""
        with patch(
            "src.services.feature_store.FeatureStoreService.get_offline_features",
            return_value=pd.DataFrame(),
        ):
            body = _post_offline(client).json()

        assert body["pairs_found"] == 0
        assert body["results"] == []

    def test_point_in_time_bound_is_passed_through(self, client):
        """as_of_timestamp must reach the store — it is what prevents label leakage."""
        with patch(
            "src.services.feature_store.FeatureStoreService.get_offline_features",
            return_value=_offline_dataframe(),
        ) as mock_offline:
            _post_offline(client)

        domain_request = mock_offline.call_args.args[0]
        assert domain_request.as_of_timestamp.replace(tzinfo=None) == datetime(2026, 9, 17)
        assert domain_request.feature_version == "v2.0.0"

    def test_returned_features_predate_as_of(self, client):
        with patch(
            "src.services.feature_store.FeatureStoreService.get_offline_features",
            return_value=_offline_dataframe(),
        ):
            body = _post_offline(client).json()

        computed_at = datetime.fromisoformat(body["results"][0]["computed_at"])
        as_of = datetime.fromisoformat(AS_OF.replace("Z", "+00:00"))
        assert computed_at.replace(tzinfo=None) <= as_of.replace(tzinfo=None)


class TestTrainingServingParity:
    """
    The core Part 11 guarantee: the vector layout used to TRAIN must equal the
    layout used to SERVE. Online order comes from FeatureVector.as_list
    (sorted feature names); offline order comes from FEATURE_COLUMN_NAMES.
    """

    def test_offline_ordering_equals_online_ordering(self):
        online_vector = FeatureVector(
            entity_id_1="CUST_001",
            entity_id_2="CUST_002",
            tenant_id=TENANT_ID,
            features={name: i / 100.0 for i, name in enumerate(FEATURE_COLUMN_NAMES)},
            feature_version="v2.0.0",
        )
        # as_list sorts by feature name; FEATURE_COLUMN_NAMES is already sorted.
        assert list(FEATURE_COLUMN_NAMES) == sorted(FEATURE_COLUMN_NAMES)
        assert online_vector.as_list == [
            i / 100.0 for i, _ in enumerate(FEATURE_COLUMN_NAMES)
        ]

    def test_catalog_is_exactly_50_and_deterministic(self):
        assert len(FEATURE_COLUMN_NAMES) == 50
        assert len(set(FEATURE_COLUMN_NAMES)) == 50
        # Recomputing must give a byte-identical ordering.
        from src.repositories.offline_store import _get_feature_column_names
        assert _get_feature_column_names() == FEATURE_COLUMN_NAMES

    def test_no_placeholder_feature_names(self):
        """Guards against the `feature_0..feature_49` fabricated-vector pattern."""
        for name in FEATURE_COLUMN_NAMES:
            assert not name.startswith("feature_"), f"placeholder name leaked: {name}"
            assert "_" in name, f"feature name is not catalog-shaped: {name}"


class TestPartitionPruning:
    """
    Regression: _list_keys_in_range used to prune on the S3 object's LastModified
    (when the file was UPLOADED) instead of the partition timestamp in the key
    (when the features were COMPUTED). A backfill written today containing
    features computed last week was therefore invisible to a point-in-time read
    for last week.
    """

    def test_partition_timestamp_parsed_from_key(self):
        from src.repositories.offline_store import OfflineFeatureStore

        key = "features/tenant-a/year=2026/month=09/day=16/features_192758.parquet"
        assert OfflineFeatureStore._partition_dt_from_key(key) == datetime(2026, 9, 16, 19, 27, 58)

    def test_unparseable_key_returns_none_for_lastmodified_fallback(self):
        from src.repositories.offline_store import OfflineFeatureStore

        assert OfflineFeatureStore._partition_dt_from_key("features/tenant-a/legacy.parquet") is None
        assert OfflineFeatureStore._partition_dt_from_key("") is None

    def test_key_uploaded_late_is_still_found_by_its_partition_date(self):
        """A file uploaded AFTER as_of, but partitioned BEFORE it, must be included."""
        from src.repositories.offline_store import OfflineFeatureStore

        store = OfflineFeatureStore.__new__(OfflineFeatureStore)
        store._prefix = "features"
        store._bucket = "b"

        key = "features/tenant-a/year=2026/month=09/day=16/features_120000.parquet"

        class _Paginator:
            def paginate(self, **_):
                # LastModified is a day AFTER the as_of bound below.
                return [{"Contents": [{"Key": key, "LastModified": datetime(2026, 9, 18, 10, 0, 0)}]}]

        store._s3 = type("S3", (), {"get_paginator": lambda self, _n: _Paginator()})()

        found = store._list_keys_in_range("tenant-a", datetime(2020, 1, 1), datetime(2026, 9, 17))
        assert found == [key]
