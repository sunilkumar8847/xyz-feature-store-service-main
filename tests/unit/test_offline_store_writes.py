"""
Unit tests for OfflineFeatureStore.write_features — the guarantees materialization
relies on.

Dependency type: TEST DOUBLE for the boto3 S3 client (records put_object calls and
serves them back to get_object/list). Real pyarrow Parquet encoding and decoding.
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.domain.models import FeatureVector, OfflineFeatureRequest
from src.repositories.offline_store import FEATURE_COLUMN_NAMES, OfflineFeatureStore

T1 = "00000000-0000-0000-0000-000000000001"
T2 = "00000000-0000-0000-0000-000000000002"


class RecordingS3:
    def __init__(self):
        self.objects = {}  # key -> (body, metadata, last_modified)

    def put_object(self, Bucket, Key, Body, ContentType=None, Metadata=None):
        self.objects[Key] = (Body, Metadata or {}, datetime.utcnow())

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key][0])}

    def get_paginator(self, _name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                return [{"Contents": [
                    {"Key": k, "LastModified": v[2]} for k, v in objects.items() if k.startswith(Prefix)
                ]}]
        return Paginator()


@pytest.fixture
def store():
    s = OfflineFeatureStore.__new__(OfflineFeatureStore)
    s._s3 = RecordingS3()
    s._bucket = "test-bucket"
    s._prefix = "features"
    return s


def fv(e1, e2, tenant=T1, computed_at=None, names=None):
    names = names or FEATURE_COLUMN_NAMES
    v = FeatureVector(e1, e2, tenant, {n: i / 100 for i, n in enumerate(names)}, "v2.0.0")
    if computed_at:
        v.computed_at = computed_at
    return v


class TestNoSilentOverwrite:
    def test_two_writes_in_the_same_second_create_two_objects(self, store):
        """Keys used to be features_HHMMSS.parquet: same tenant + same second = overwrite."""
        ts = datetime(2026, 9, 28, 10, 0, 0)
        store.write_features([fv("A1", "A2", computed_at=ts)])
        store.write_features([fv("A3", "A4", computed_at=ts)])
        assert len(store._s3.objects) == 2

    def test_unique_keys_still_parse_as_partitions(self, store):
        store.write_features([fv("A1", "A2", computed_at=datetime(2026, 9, 28, 10, 11, 12))])
        (key,) = store._s3.objects
        assert OfflineFeatureStore._partition_dt_from_key(key) == datetime(2026, 9, 28, 10, 11, 12)

    def test_legacy_keys_without_suffix_still_parse(self):
        key = "features/t/year=2026/month=09/day=16/features_192758.parquet"
        assert OfflineFeatureStore._partition_dt_from_key(key) == datetime(2026, 9, 16, 19, 27, 58)


class TestTenantGuard:
    def test_mixed_tenant_batch_is_refused(self, store):
        with pytest.raises(ValueError, match="single-tenant"):
            store.write_features([fv("A1", "A2", tenant=T1), fv("B1", "B2", tenant=T2)])
        assert store._s3.objects == {}

    def test_object_is_filed_under_its_tenant(self, store):
        store.write_features([fv("B1", "B2", tenant=T2)])
        (key,) = store._s3.objects
        assert key.startswith(f"features/{T2}/")


class TestNoZeroFill:
    def test_vector_missing_canonical_names_is_refused(self, store):
        """Previously features.get(name, 0.0) wrote zeros for missing names."""
        wrong = [n for n in FEATURE_COLUMN_NAMES if not n.startswith("ph_")] + [f"ph_err_{i}" for i in range(5)]
        with pytest.raises(ValueError, match="missing 5 canonical feature"):
            store.write_features([fv("A1", "A2", names=wrong)])
        assert store._s3.objects == {}


class TestPartitionTimestamp:
    def test_partition_defaults_to_earliest_computed_at_not_write_time(self, store):
        early = datetime(2026, 9, 28, 9, 0, 0)
        store.write_features([
            fv("A1", "A2", computed_at=early + timedelta(minutes=5)),
            fv("A3", "A4", computed_at=early),
        ])
        (key,) = store._s3.objects
        assert OfflineFeatureStore._partition_dt_from_key(key) == early


class TestProvenanceMetadata:
    def test_data_source_recorded_on_the_object(self, store):
        store.write_features([fv("A1", "A2")], data_source="synthetic")
        (_, meta, _), = store._s3.objects.values()
        assert meta["data_source"] == "synthetic"
        assert meta["tenant_id"] == T1
        assert meta["feature_version"] == "v2.0.0"

    def test_unspecified_source_is_explicit(self, store):
        store.write_features([fv("A1", "A2")])
        (_, meta, _), = store._s3.objects.values()
        assert meta["data_source"] == "unspecified"


class TestPointInTimeRoundTrip:
    """write_features -> read_point_in_time, through real Parquet encoding."""

    def test_row_visible_after_computed_at_and_hidden_before(self, store):
        computed = datetime(2026, 9, 28, 12, 0, 0)
        store.write_features([fv("A1", "A2", computed_at=computed)])

        def read(as_of):
            return store.read_point_in_time(OfflineFeatureRequest(
                entity_pairs=[("A1", "A2")], tenant_id=T1, as_of_timestamp=as_of,
            ))

        after = read(computed + timedelta(seconds=1))
        assert len(after) == 1
        assert [float(after.iloc[0][f"feat_{n}"]) for n in FEATURE_COLUMN_NAMES] == pytest.approx(
            [i / 100 for i in range(50)]
        )
        assert read(computed - timedelta(seconds=1)).empty

    def test_other_tenant_cannot_read_the_row(self, store):
        computed = datetime(2026, 9, 28, 12, 0, 0)
        store.write_features([fv("A1", "A2", tenant=T1, computed_at=computed)])
        result = store.read_point_in_time(OfflineFeatureRequest(
            entity_pairs=[("A1", "A2")], tenant_id=T2, as_of_timestamp=computed + timedelta(hours=1),
        ))
        assert result.empty
