"""
feature-store-service/src/api/v1/schemas.py

Pydantic v2 DTOs for all API request/response contracts.
These match the LLD Section 5.2 Response Schema exactly.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Feature Retrieval ────────────────────────────────────────────────────────

class FeatureGetResponse(BaseModel):
    """Response for GET /v1/features/{entity_id_1}/{entity_id_2}"""

    entity_id_1: str
    entity_id_2: str
    tenant_id: str
    features: Dict[str, float] = Field(
        description="Feature name → value mapping (50 features)"
    )
    feature_vector: List[float] = Field(
        description="Ordered 50-dimensional feature vector for ML inference"
    )
    timestamp: str = Field(description="When features were computed (ISO8601)")
    version: str = Field(description="Feature definition version")
    computation_ms: float = Field(description="Computation time in milliseconds")
    source: str = Field(description="cache | compute", default="cache")

    model_config = {"json_schema_extra": {
        "example": {
            "entity_id_1": "ent_123",
            "entity_id_2": "ent_456",
            "tenant_id": "tenant_abc",
            "features": {
                "ss_levenshtein_name": 0.87,
                "ph_soundex_name": 1.0,
                "sem_cosine_name": 0.92,
            },
            "feature_vector": [0.87, 1.0, 0.92],
            "timestamp": "2024-01-15T10:30:00Z",
            "version": "v2.0.0",
            "computation_ms": 3.2,
            "source": "cache",
        }
    }}

class EntityPairInput(BaseModel):
    entity_id_1: str
    entity_id_2: str
    # Optional entity data for on-the-fly computation
    entity1_fields: Optional[Dict[str, Optional[str]]] = None
    entity2_fields: Optional[Dict[str, Optional[str]]] = None

class BatchFeatureRequest(BaseModel):
    """Request for POST /v1/features/batch"""

    pairs: List[EntityPairInput] = Field(
        description="List of entity pairs (max 100)",
        max_length=100,
    )
    tenant_id: str
    include_entity_data: bool = Field(
        default=False,
        description="If true, compute features on-the-fly for cache misses",
    )




class BatchFeatureResponse(BaseModel):
    pairs_requested: int
    pairs_found: int
    pairs_computed: int
    results: List[Optional[FeatureGetResponse]]


# ─── Feature Push ─────────────────────────────────────────────────────────────

class FeaturePushRequest(BaseModel):
    """Request for PUT /v1/features/{entity_id}"""

    tenant_id: str
    entity_type: str
    fields: Dict[str, Optional[str]] = Field(
        description="Entity field values for feature computation"
    )


class FeaturePushResponse(BaseModel):
    entity_id: str
    invalidated_pairs: int
    message: str


# ─── Feature Definitions ──────────────────────────────────────────────────────

class FeatureDefinitionResponse(BaseModel):
    """Response for GET /v1/features/definitions"""

    id: UUID
    name: str
    category: str
    description: str
    data_type: str
    version: str
    status: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime


class FeatureDefinitionListResponse(BaseModel):
    total: int
    version: str
    features: List[FeatureDefinitionResponse]


# ─── Materialization ──────────────────────────────────────────────────────────

class MaterializationRequest(BaseModel):
    """Request for POST /v1/materialize"""

    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant to materialize. None = all tenants."
    )


class MaterializationResponse(BaseModel):
    job_id: UUID
    status: str
    tenant_id: Optional[str]
    triggered_by: str
    message: str


class MaterializationStatusResponse(BaseModel):
    job_id: UUID
    status: str
    tenant_id: Optional[str]
    total_entities: int
    processed_entities: int
    failed_entities: int
    progress_pct: float
    duration_seconds: Optional[float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    triggered_by: str


# ─── Offline Features ─────────────────────────────────────────────────────────

class OfflineFeatureRequest(BaseModel):
    """Request for POST /v1/features/offline"""

    entity_pairs: List[List[str]] = Field(
        description="List of [entity_id_1, entity_id_2] pairs for training",
        max_length=100000,
    )
    tenant_id: str
    as_of_timestamp: datetime = Field(
        description="Point-in-time for historical feature retrieval"
    )
    feature_version: str = "v2.0.0"

    @field_validator("entity_pairs")
    @classmethod
    def validate_pairs(cls, v: list) -> list:
        for pair in v:
            if len(pair) != 2:
                raise ValueError("Each pair must have exactly 2 entity IDs")
        return v


class OfflineFeatureRow(BaseModel):
    """One point-in-time correct feature vector for a training pair."""

    entity_id_1: str
    entity_id_2: str
    feature_vector: List[float] = Field(
        description="Ordered 50-dim vector; order matches OfflineFeatureResponse.feature_names"
    )
    computed_at: datetime = Field(
        description="When these features were computed (always <= as_of_timestamp)"
    )


class OfflineFeatureResponse(BaseModel):
    request_id: UUID
    pairs_requested: int
    pairs_found: int
    as_of_timestamp: datetime
    feature_version: str
    feature_names: List[str] = Field(
        default_factory=list,
        description=(
            "Canonical feature ordering for every feature_vector in results. "
            "Identical to the online ordering (sorted feature names), so training "
            "and serving consume the same vector layout."
        ),
    )
    results: List[OfflineFeatureRow] = Field(
        default_factory=list,
        description="Point-in-time feature vectors. Pairs with no history are omitted.",
    )
    # Reserved for large exports (Parquet/CSV handoff); unused for in-body results.
    download_url: Optional[str] = None
    message: str


# ─── Drift Reports ────────────────────────────────────────────────────────────

class DriftReportResponse(BaseModel):
    report_id: UUID
    drift_type: str
    feature_name: str
    kl_divergence: float
    js_divergence: float
    is_drifted: bool
    baseline_mean: float
    current_mean: float
    sample_count: int
    computed_at: datetime


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str  # healthy | degraded | unhealthy
    service: str
    version: str
    timestamp: datetime
    checks: Dict[str, bool]

    model_config = {"json_schema_extra": {
        "example": {
            "status": "healthy",
            "service": "feature-store",
            "version": "3.0.0",
            "timestamp": "2024-01-15T10:30:00Z",
            "checks": {
                "redis": True,
                "postgres": True,
                "kafka": True,
                "s3": True,
            }
        }
    }}


# ─── Store Stats ──────────────────────────────────────────────────────────────

class StoreStatsResponse(BaseModel):
    online_store: Dict
    service_version: str
    timestamp: datetime
