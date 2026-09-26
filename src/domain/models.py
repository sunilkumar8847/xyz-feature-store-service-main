"""
feature-store-service/src/domain/models.py

Core domain models for the Feature Store.
These are pure Python dataclasses/Pydantic models — no ORM concerns.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4


# ─── Enums ──────────────────────────────────────────────────────────────────


class FeatureStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    EXPERIMENTAL = "EXPERIMENTAL"


class FeatureCategory(str, Enum):
    STRING_SIMILARITY = "STRING_SIMILARITY"
    PHONETIC = "PHONETIC"
    TOKEN_BASED = "TOKEN_BASED"
    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"
    DOMAIN_SPECIFIC = "DOMAIN_SPECIFIC"


class MaterializationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DriftType(str, Enum):
    DATA_DRIFT = "DATA_DRIFT"
    CONCEPT_DRIFT = "CONCEPT_DRIFT"
    LABEL_DRIFT = "LABEL_DRIFT"
    PREDICTION_DRIFT = "PREDICTION_DRIFT"


# ─── Domain Models ───────────────────────────────────────────────────────────


@dataclass
class FeatureDefinition:
    """
    Metadata about a single feature: its name, type, computation logic,
    and version. Stored in the feature registry (PostgreSQL).
    """

    name: str
    category: FeatureCategory
    description: str
    data_type: str  # "float", "boolean", "integer"
    version: str = "v2.0.0"
    status: FeatureStatus = FeatureStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureVector:
    """
    A 50-dimensional feature vector computed for an entity pair.
    This is the core unit of work for the matching service.
    """

    entity_id_1: str
    entity_id_2: str
    tenant_id: str
    features: Dict[str, float]  # feature_name -> value (50 features)
    feature_version: str
    computed_at: datetime = field(default_factory=datetime.utcnow)
    computation_ms: float = 0.0
    id: UUID = field(default_factory=uuid4)

    @property
    def as_list(self) -> List[float]:
        """Ordered feature vector for ML inference."""
        return [self.features.get(k, 0.0) for k in sorted(self.features.keys())]

    @property
    def cache_key(self) -> str:
        """Redis key: deterministic for entity pair (order-independent)."""
        sorted_ids = sorted([self.entity_id_1, self.entity_id_2])
        pair_hash = hashlib.sha256(
            f"{sorted_ids[0]}:{sorted_ids[1]}:{self.tenant_id}".encode()
        ).hexdigest()[:16]
        return f"feature:{self.tenant_id}:{pair_hash}"

    def __post_init__(self):
        if len(self.features) != 50:
            raise ValueError(
                f"FeatureVector must have exactly 50 features, got {len(self.features)}"
            )


@dataclass
class EntitySnapshot:
    """
    A snapshot of an entity's field values at a specific point in time.
    Used as input for feature computation.
    """

    entity_id: str
    tenant_id: str
    entity_type: str
    fields: Dict[str, Optional[str]]  # field_name -> value
    snapshot_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def name(self) -> str:
        return self.fields.get("name") or self.fields.get("full_name") or ""

    @property
    def email(self) -> str:
        return self.fields.get("email") or ""

    @property
    def phone(self) -> str:
        return self.fields.get("phone") or self.fields.get("phone_number") or ""

    @property
    def address(self) -> str:
        parts = [
            self.fields.get("address_line1"),
            self.fields.get("address_line2"),
            self.fields.get("city"),
            self.fields.get("state"),
            self.fields.get("country"),
        ]
        return " ".join(p for p in parts if p)

    @property
    def lat_lng(self) -> Optional[tuple]:
        lat = self.fields.get("latitude")
        lng = self.fields.get("longitude")
        if lat and lng:
            try:
                return (float(lat), float(lng))
            except ValueError:
                return None
        return None


@dataclass
class MaterializationJob:
    """
    A batch job that computes features for all entities and pushes to
    the online store (Redis).
    """

    job_id: UUID = field(default_factory=uuid4)
    tenant_id: Optional[str] = None  # None = all tenants
    lookback_days: int = 30
    status: MaterializationStatus = MaterializationStatus.PENDING
    total_entities: int = 0
    processed_entities: int = 0
    failed_entities: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: str = "scheduler"  # scheduler | api | drift

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def progress_pct(self) -> float:
        if self.total_entities == 0:
            return 0.0
        return round(self.processed_entities / self.total_entities * 100, 2)


@dataclass
class DriftReport:
    """
    Statistical drift report for a set of features.
    Produced by the drift detection worker.
    """

    feature_name: str
    drift_type: str
    drift_score: float
    tenant_id: str
    run_id: UUID
    is_alert: bool
    detection_method: str
    baseline_stats: Dict = field(default_factory=dict)
    current_stats: Dict = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    detected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OfflineFeatureRequest:
    """
    Point-in-time correct feature retrieval for model training.
    Ensures zero training-serving skew.
    """

    entity_pairs: List[tuple]  # List of (entity_id_1, entity_id_2)
    tenant_id: str
    as_of_timestamp: datetime  # Point-in-time
    feature_version: str = "v2.0.0"
    request_id: UUID = field(default_factory=uuid4)
