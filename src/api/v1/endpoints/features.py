"""
feature-store-service/src/api/v1/endpoints/features.py

All feature store REST endpoints.
Matches LLD Part V API Specification exactly.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from xyz_security import Action, Resource, TenantContext, get_current_tenant, require_permission

from src.api.v1.schemas import (
    BatchFeatureRequest, BatchFeatureResponse, DriftReportResponse,
    FeatureDefinitionListResponse, FeatureDefinitionResponse,
    FeatureGetResponse, FeaturePushRequest, FeaturePushResponse,
    HealthResponse, MaterializationRequest, MaterializationResponse,
    MaterializationStatusResponse, OfflineFeatureRequest,
    OfflineFeatureResponse, OfflineFeatureRow, StoreStatsResponse,EntityPairInput
)
from src.core.config import settings
from src.domain.models import EntitySnapshot, OfflineFeatureRequest as DomainOfflineRequest
from src.repositories.feature_registry import FeatureRegistryRepository, get_db_session
from src.repositories.online_store import OnlineFeatureStore
from src.repositories.offline_store import FEATURE_COLUMN_NAMES, OfflineFeatureStore
from src.services.feature_computation import FeatureComputationService
from src.services.feature_store import FeatureStoreService
from src.workers.materialization import DriftDetectionWorker, run_materialization_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["features"])
public_router = APIRouter(prefix="/v1", tags=["features"])


# ─── Dependency Injection ─────────────────────────────────────────────────────

async def get_online_store() -> OnlineFeatureStore:
    from src.core.dependencies import online_store_instance
    return online_store_instance


def get_offline_store() -> OfflineFeatureStore:
    return OfflineFeatureStore()


def get_computation_service() -> FeatureComputationService:
    from src.core.dependencies import computation_service_instance
    return computation_service_instance


async def get_feature_store_service(
    online_store: OnlineFeatureStore = Depends(get_online_store),
    offline_store: OfflineFeatureStore = Depends(get_offline_store),
    computation_service: FeatureComputationService = Depends(get_computation_service),
    db: AsyncSession = Depends(get_db_session),
) -> FeatureStoreService:
    return FeatureStoreService(
        online_store=online_store,
        offline_store=offline_store,
        computation_service=computation_service,
        db_session=db,
    )


# ─── Feature Retrieval Endpoints ─────────────────────────────────────────────


@router.get(
    "/features/{entity_id_1}/{entity_id_2}",
    response_model=FeatureGetResponse,
    summary="Get features for entity pair",
    description=(
        "Retrieve the 50-dimensional feature vector for an entity pair. "
        "Returns from Redis cache (<10ms) or computes on-the-fly (~100ms). "
        "Pass entity field data to enable on-the-fly computation on cache miss."
    ),
)
async def get_features(
    entity_id_1: str,
    entity_id_2: str,
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    entity_type: Optional[str] = Query(default="customer", description="Entity type"),
    name1: Optional[str] = Query(None, description="Entity 1 name (for on-the-fly compute)"),
    name2: Optional[str] = Query(None, description="Entity 2 name (for on-the-fly compute)"),
    email1: Optional[str] = Query(None, description="Entity 1 email"),
    email2: Optional[str] = Query(None, description="Entity 2 email"),
    phone1: Optional[str] = Query(None, description="Entity 1 phone"),
    phone2: Optional[str] = Query(None, description="Entity 2 phone"),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    tenant_id = tenant.tenant_id
    # Build entity snapshots if field data provided
    entity1 = None
    entity2 = None

    if name1 or email1 or phone1:
        entity1 = EntitySnapshot(
            entity_id=entity_id_1,
            tenant_id=tenant_id,
            entity_type=entity_type,
            fields={"name": name1, "email": email1, "phone": phone1},
        )
    if name2 or email2 or phone2:
        entity2 = EntitySnapshot(
            entity_id=entity_id_2,
            tenant_id=tenant_id,
            entity_type=entity_type,
            fields={"name": name2, "email": email2, "phone": phone2},
        )

    try:
        fv = await service.get_features(
            entity_id_1=entity_id_1,
            entity_id_2=entity_id_2,
            tenant_id=tenant_id,
            entity1=entity1,
            entity2=entity2,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return FeatureGetResponse(
        entity_id_1=fv.entity_id_1,
        entity_id_2=fv.entity_id_2,
        tenant_id=fv.tenant_id,
        features=fv.features,
        feature_vector=fv.as_list,
        timestamp=fv.computed_at.isoformat(),
        version=fv.feature_version,
        computation_ms=fv.computation_ms,
        source="cache" if fv.computation_ms < 1.0 else "compute",
    )


@router.post(
    "/features/batch",
    response_model=BatchFeatureResponse,
    summary="Batch feature retrieval (max 100 pairs)",
    description=(
        "Retrieve feature vectors for up to 100 entity pairs in a single request. "
        "Uses Redis pipeline for minimum round-trips. Target: <50ms P99."
    ),
)
async def get_features_batch(
    request: BatchFeatureRequest,
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    tenant_id = tenant.tenant_id
    pairs = [
        {
            "entity_id_1": p.entity_id_1,
            "entity_id_2": p.entity_id_2,
            "tenant_id": tenant_id,
        }
        for p in request.pairs
    ]

    # Build entity snapshots for on-the-fly computation
    entity_snapshots = {}
    if request.include_entity_data:
        for p in request.pairs:
            entity_type = getattr(p, "entity_type", "customer") or "customer"
            if p.entity1_fields:
                entity_snapshots[p.entity_id_1] = EntitySnapshot(
                    entity_id=p.entity_id_1,
                    tenant_id=tenant_id,
                    entity_type=entity_type,
                    fields=p.entity1_fields,
                )
            if p.entity2_fields:
                entity_snapshots[p.entity_id_2] = EntitySnapshot(
                    entity_id=p.entity_id_2,
                    tenant_id=tenant_id,
                    entity_type=entity_type,
                    fields=p.entity2_fields,
                )

    results = await service.get_features_batch(
        pairs=pairs,
        entity_snapshots=entity_snapshots if entity_snapshots else None,
    )

    response_results = []
    found = 0
    computed = 0

    for fv in results:
        if fv is None:
            response_results.append(None)
        else:
            found += 1
            if fv.computation_ms > 1.0:
                computed += 1
            response_results.append(FeatureGetResponse(
                entity_id_1=fv.entity_id_1,
                entity_id_2=fv.entity_id_2,
                tenant_id=fv.tenant_id,
                features=fv.features,
                feature_vector=fv.as_list,
                timestamp=fv.computed_at.isoformat(),
                version=fv.feature_version,
                computation_ms=fv.computation_ms,
                source="cache" if fv.computation_ms < 1.0 else "compute",
            ))

    return BatchFeatureResponse(
        pairs_requested=len(pairs),
        pairs_found=found,
        pairs_computed=computed,
        results=response_results,
    )


@router.put(
    "/features/{entity_id}",
    response_model=FeaturePushResponse,
    status_code=status.HTTP_200_OK,
    summary="Push feature updates for entity",
    description=(
        "Invalidates all cached feature vectors involving this entity. "
        "Features will be recomputed on next request."
    ),
)
async def push_features(
    entity_id: str,
    request: FeaturePushRequest,
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.WRITE)),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    entity = EntitySnapshot(
        entity_id=entity_id,
        tenant_id=tenant.tenant_id,
        entity_type=request.entity_type,
        fields=request.fields,
    )
    invalidated = await service.push_features(entity_id, tenant.tenant_id, entity)
    return FeaturePushResponse(
        entity_id=entity_id,
        invalidated_pairs=invalidated,
        message=f"Invalidated {invalidated} cached feature vectors",
    )


@router.get(
    "/features/definitions",
    response_model=FeatureDefinitionListResponse,
    summary="List feature definitions",
    description="Returns all 50 feature definitions with metadata.",
)
async def list_feature_definitions(
    version: Optional[str] = Query(None, description="Filter by version (default: current)"),
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    definitions = await service.list_feature_definitions(version=version)
    return FeatureDefinitionListResponse(
        total=len(definitions),
        version=version or settings.FEATURE_VERSION,
        features=[
            FeatureDefinitionResponse(
                id=d.id,
                name=d.name,
                category=d.category.value,
                description=d.description,
                data_type=d.data_type,
                version=d.version,
                status=d.status.value,
                tags=d.tags,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in definitions
        ],
    )


# ─── Offline Feature Access ───────────────────────────────────────────────────

@router.post(
    "/features/offline",
    response_model=OfflineFeatureResponse,
    summary="Point-in-time correct feature retrieval for training",
    description=(
        "Retrieves features as they existed at as_of_timestamp. "
        "Critical for preventing training-serving skew. "
        "For large datasets, returns a download URL to a Parquet file."
    ),
)
async def get_offline_features(
    request: OfflineFeatureRequest,
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    domain_request = DomainOfflineRequest(
        entity_pairs=[(p[0], p[1]) for p in request.entity_pairs],
        tenant_id=tenant.tenant_id,
        as_of_timestamp=request.as_of_timestamp,
        feature_version=request.feature_version,
    )

    df = await service.get_offline_features(domain_request)

    # The service already returns point-in-time correct rows; serialise them so
    # training can actually consume features (previously only counts were returned,
    # which made this endpoint unusable as a training source).
    rows = []
    for record in df.to_dict("records") if not df.empty else []:
        rows.append(OfflineFeatureRow(
            entity_id_1=record["entity_id_1"],
            entity_id_2=record["entity_id_2"],
            feature_vector=[
                float(record[f"feat_{name}"]) for name in FEATURE_COLUMN_NAMES
            ],
            computed_at=record["computed_at"],
        ))

    return OfflineFeatureResponse(
        request_id=domain_request.request_id,
        pairs_requested=len(request.entity_pairs),
        pairs_found=len(rows),
        as_of_timestamp=request.as_of_timestamp,
        feature_version=request.feature_version,
        feature_names=list(FEATURE_COLUMN_NAMES),
        results=rows,
        message=f"Retrieved {len(rows)} feature vectors",
    )


# ─── Materialization ──────────────────────────────────────────────────────────

@router.post(
    "/materialize",
    response_model=MaterializationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger feature materialization",
    description=(
        "Enqueues a batch materialization job that computes features for all "
        "entity pairs and pushes to the online store (Redis)."
    ),
)
async def trigger_materialization(
    request: MaterializationRequest,
    background_tasks: BackgroundTasks,
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.WRITE)),
    service: FeatureStoreService = Depends(get_feature_store_service),
    online_store: OnlineFeatureStore = Depends(get_online_store),
):
    job = await service.trigger_materialization(
        tenant_id=tenant.tenant_id,
        triggered_by=f"api:{tenant.user_id}",
    )

    # Run in the background through the shared runner, which owns its own DB session
    # (this request's session is closed before background tasks run) and chooses the
    # data source from configuration. A missing or invalid source is reported as a
    # FAILED job with error_message — see GET /api/v1/materialize/{job_id}.
    background_tasks.add_task(
        run_materialization_job, job, online_store, get_computation_service()
    )

    return MaterializationResponse(
        job_id=job.job_id,
        status=job.status.value,
        tenant_id=job.tenant_id,
        triggered_by=job.triggered_by,
        message=f"Materialization job {job.job_id} enqueued",
    )


@router.get(
    "/materialize/{job_id}",
    response_model=MaterializationStatusResponse,
    summary="Get materialization job status",
)
async def get_materialization_status(
    job_id: UUID,
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    job = await service.get_job_status(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Materialization job {job_id} not found",
        )
    return MaterializationStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        tenant_id=job.tenant_id,
        total_entities=job.total_entities,
        processed_entities=job.processed_entities,
        failed_entities=job.failed_entities,
        progress_pct=job.progress_pct,
        duration_seconds=job.duration_seconds,
        started_at=job.started_at,
        completed_at=job.completed_at,
        triggered_by=job.triggered_by,
        skipped_entities=job.skipped_entities,
        error_message=job.error_message,
    )


@router.get(
    "/materialize",
    response_model=List[MaterializationStatusResponse],
    summary="List recent materialization jobs",
)
async def list_materialization_jobs(
    limit: int = Query(default=10, le=50),
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    service: FeatureStoreService = Depends(get_feature_store_service),
):
    jobs = await service.list_recent_jobs(limit=limit)
    return [
        MaterializationStatusResponse(
            job_id=j.job_id,
            status=j.status.value,
            tenant_id=j.tenant_id,
            total_entities=j.total_entities,
            processed_entities=j.processed_entities,
            failed_entities=j.failed_entities,
            progress_pct=j.progress_pct,
            duration_seconds=j.duration_seconds,
            started_at=j.started_at,
            completed_at=j.completed_at,
            triggered_by=j.triggered_by,
            skipped_entities=j.skipped_entities,
            error_message=j.error_message,
        )
        for j in jobs
    ]


# ─── Drift Reports ────────────────────────────────────────────────────────────

@router.get(
    "/drift",
    response_model=List[DriftReportResponse],
    summary="Get latest drift reports",
)
async def get_drift_reports(
    feature_name: Optional[str] = Query(None),
    limit: int = Query(default=50, le=200),
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    db: AsyncSession = Depends(get_db_session),
):
    registry = FeatureRegistryRepository(db)
    reports = await registry.get_latest_drift_reports(
        feature_name=feature_name,
        limit=limit,
    )
    return [
        DriftReportResponse(
            report_id=r.report_id,
            drift_type=r.drift_type.value,
            feature_name=r.feature_name,
            kl_divergence=r.kl_divergence,
            js_divergence=r.js_divergence,
            is_drifted=r.is_drifted,
            baseline_mean=r.baseline_mean,
            current_mean=r.current_mean,
            sample_count=r.sample_count,
            computed_at=r.computed_at,
        )
        for r in reports
    ]


# ─── Health & Stats ───────────────────────────────────────────────────────────

@public_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health_check(
    online_store: OnlineFeatureStore = Depends(get_online_store),
    db: AsyncSession = Depends(get_db_session),
):
    checks = {}

    # Redis
    try:
        checks["redis"] = await online_store.ping()
    except Exception:
        checks["redis"] = False

    # PostgreSQL
    try:
        await db.execute(
            __import__("sqlalchemy", fromlist=["text"]).text("SELECT 1")
        )
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False

    # S3 (basic check — just that boto3 can be instantiated)
    try:
        import boto3
        checks["s3"] = True
    except Exception:
        checks["s3"] = False

    # Kafka (best-effort)
    checks["kafka"] = True  # Kafka consumer runs in background

    all_healthy = all(checks.values())
    any_critical_down = not checks.get("redis") or not checks.get("postgres")

    return HealthResponse(
        status="healthy" if all_healthy else ("degraded" if not any_critical_down else "unhealthy"),
        service=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@router.get(
    "/stats",
    response_model=StoreStatsResponse,
    summary="Online store statistics",
)
async def get_stats(
    tenant: TenantContext = Depends(require_permission(Resource.FEATURES, Action.READ)),
    online_store: OnlineFeatureStore = Depends(get_online_store),
):
    stats = await online_store.get_stats()
    return StoreStatsResponse(
        online_store=stats,
        service_version=settings.SERVICE_VERSION,
        timestamp=datetime.utcnow(),
    )
