"""
REAL integration test: synthetic source -> MaterializationWorker -> real
FeatureComputationService -> real Redis + real S3-compatible store (LocalStack).

Dependency type: REAL LocalStack S3, REAL Redis, REAL feature computation (incl. the
sentence-transformer embedding model), REAL synthetic dataset from `synthetic_data`.
The job registry is a test double — job persistence is covered through the running
service in the Phase 2 smoke test.

Skipped automatically unless S3 (S3_ENDPOINT_URL) and Redis are reachable. Every run
writes under a unique S3 prefix and deletes it afterwards.

    ./.venv/Scripts/python.exe -m pytest tests/integration/test_materialization_localstack.py -m integration -v
"""
from __future__ import annotations

import dataclasses
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.core.config import settings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[4]


def _s3_reachable() -> bool:
    try:
        import boto3
        s3 = boto3.client(
            "s3", region_name=settings.S3_REGION, endpoint_url=settings.S3_ENDPOINT_URL,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        s3.head_bucket(Bucket=settings.S3_BUCKET)
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sd = pytest.importorskip("synthetic_data")
    out = tmp_path_factory.mktemp("synthetic_it")
    config = dataclasses.replace(sd.get_profile("smoke"), n_identities=6, n_tenants=2)
    identities, records, pairs = sd.SyntheticDatasetGenerator(config).generate()
    sd.write_dataset(out, config, identities, records, pairs)
    return out, pairs


@pytest.fixture
async def stores(monkeypatch):
    if not _s3_reachable():
        pytest.skip("S3/LocalStack not reachable")
    from src.repositories.offline_store import OfflineFeatureStore
    from src.repositories.online_store import OnlineFeatureStore
    try:
        online = await OnlineFeatureStore.create()
    except Exception:
        pytest.skip("Redis not reachable")

    prefix = f"it-materialization-{uuid.uuid4().hex[:10]}"
    monkeypatch.setattr(settings, "S3_PREFIX", prefix)
    offline = OfflineFeatureStore()
    yield online, offline, prefix

    # Clean up everything this run wrote.
    paginator = offline._s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            offline._s3.delete_object(Bucket=settings.S3_BUCKET, Key=obj["Key"])


async def test_synthetic_materialization_end_to_end(synthetic_dir, stores):
    from src.adapters.materialization_sources import SyntheticMaterializationSource
    from src.domain.models import MaterializationJob, MaterializationStatus, OfflineFeatureRequest
    from src.repositories.offline_store import FEATURE_COLUMN_NAMES
    from src.services.feature_computation import FeatureComputationService
    from src.workers.materialization import MaterializationWorker

    out, pairs = synthetic_dir
    online, offline, prefix = stores
    registry = AsyncMock()

    started = datetime.utcnow()
    job = await MaterializationWorker(
        online_store=online, offline_store=offline,
        computation_service=FeatureComputationService(), registry=registry,
        source=SyntheticMaterializationSource(str(out)),
    ).run_job(MaterializationJob(triggered_by="integration-test"))

    # Job contract
    assert job.status == MaterializationStatus.COMPLETED, job.error_message
    assert job.processed_entities == len(pairs)
    assert job.failed_entities == 0 and job.skipped_entities == 0

    # Offline: one object per tenant batch, correctly filed and labelled
    listed = offline._s3.list_objects_v2(Bucket=settings.S3_BUCKET, Prefix=f"{prefix}/")
    keys = [o["Key"] for o in listed.get("Contents", [])]
    tenants = sorted({p.tenant_id for p in pairs})
    assert sorted({k.split("/")[1] for k in keys}) == tenants
    for key in keys:
        meta = offline._s3.head_object(Bucket=settings.S3_BUCKET, Key=key)["Metadata"]
        assert meta["data_source"] == "synthetic"
        assert meta["feature_version"] == settings.FEATURE_VERSION
        assert key.split("/")[1] == meta["tenant_id"]

    # Offline point-in-time read, per tenant: every pair, 50 canonical columns
    for tenant in tenants:
        tenant_pairs = [(p.entity_id_1, p.entity_id_2) for p in pairs if p.tenant_id == tenant]
        df = offline.read_point_in_time(OfflineFeatureRequest(
            entity_pairs=tenant_pairs, tenant_id=tenant,
            as_of_timestamp=datetime.utcnow() + timedelta(seconds=1),
        ))
        assert len(df) == len(tenant_pairs)
        assert all(f"feat_{n}" in df.columns for n in FEATURE_COLUMN_NAMES)
        assert set(df["tenant_id"]) == {tenant}

        # Before the job ran, nothing existed yet.
        early = offline.read_point_in_time(OfflineFeatureRequest(
            entity_pairs=tenant_pairs, tenant_id=tenant,
            as_of_timestamp=started - timedelta(seconds=1),
        ))
        assert early.empty

    # Online (Redis): the same vector is served for a materialized pair
    p = pairs[0]
    cached = await online.get(p.entity_id_1, p.entity_id_2, p.tenant_id)
    assert cached is not None and len(cached.features) == 50
    other_tenant = next(t for t in tenants if t != p.tenant_id)
    assert await online.get(p.entity_id_1, p.entity_id_2, other_tenant) is None
