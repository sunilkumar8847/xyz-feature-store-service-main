"""
feature-store-service/src/repositories/feature_registry.py

PostgreSQL-backed feature registry.
Stores feature definitions, versions, materialization job history,
and drift reports. The central audit trail for all features.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer,
    String, Text, Enum as SAEnum, select, update
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.core.config import settings
from src.domain.models import (
    DriftReport, DriftType, FeatureCategory, FeatureDefinition,
    FeatureStatus, MaterializationJob, MaterializationStatus
)

logger = logging.getLogger(__name__)


# ─── SQLAlchemy Base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── ORM Models ──────────────────────────────────────────────────────────────

class FeatureDefinitionORM(Base):
    __tablename__ = "feature_definitions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="v2.0.0")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="ACTIVE")
    tags: Mapped[dict] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_feature_def_name_version", "name", "version", unique=True),
        Index("ix_feature_def_category", "category"),
        Index("ix_feature_def_status", "status"),
    )


class MaterializationJobORM(Base):
    __tablename__ = "materialization_jobs"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    lookback_days: Mapped[int] = mapped_column(Integer, default=30)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="PENDING")
    total_entities: Mapped[int] = mapped_column(Integer, default=0)
    processed_entities: Mapped[int] = mapped_column(Integer, default=0)
    failed_entities: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    triggered_by: Mapped[str] = mapped_column(String(100), default="scheduler")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_mat_job_status", "status"),
        Index("ix_mat_job_tenant", "tenant_id"),
        Index("ix_mat_job_created", "created_at"),
    )


class DriftReportORM(Base):
    __tablename__ = "drift_reports"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    drift_type: Mapped[str] = mapped_column(String(50), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    kl_divergence: Mapped[float] = mapped_column(Float, nullable=False)
    js_divergence: Mapped[float] = mapped_column(Float, nullable=False)
    is_drifted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    baseline_mean: Mapped[float] = mapped_column(Float, nullable=False)
    current_mean: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_std: Mapped[float] = mapped_column(Float, nullable=False)
    current_std: Mapped[float] = mapped_column(Float, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_drift_feature", "feature_name"),
        Index("ix_drift_type", "drift_type"),
        Index("ix_drift_computed", "computed_at"),
    )


# ─── Database Engine ─────────────────────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.POSTGRES_POOL_SIZE,
            max_overflow=settings.POSTGRES_MAX_OVERFLOW,
            pool_timeout=settings.POSTGRES_POOL_TIMEOUT,
            echo=settings.POSTGRES_ECHO_SQL,
            future=True,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncSession:
    """FastAPI dependency for DB session injection."""
    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ─── Feature Registry Repository ─────────────────────────────────────────────

class FeatureRegistryRepository:
    """CRUD operations on the PostgreSQL feature registry."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_feature(self, name: str, version: str) -> Optional[FeatureDefinition]:
        result = await self._session.execute(
            select(FeatureDefinitionORM).where(
                FeatureDefinitionORM.name == name,
                FeatureDefinitionORM.version == version,
            )
        )
        row = result.scalar_one_or_none()
        return self._orm_to_domain(row) if row else None

    async def list_features(
        self,
        version: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[FeatureDefinition]:
        stmt = select(FeatureDefinitionORM)
        if version:
            stmt = stmt.where(FeatureDefinitionORM.version == version)
        if status:
            stmt = stmt.where(FeatureDefinitionORM.status == status)
        result = await self._session.execute(stmt)
        return [self._orm_to_domain(r) for r in result.scalars().all()]

    async def upsert_feature(self, feature: FeatureDefinition) -> FeatureDefinition:
        existing = await self.get_feature(feature.name, feature.version)
        if existing:
            await self._session.execute(
                update(FeatureDefinitionORM)
                .where(
                    FeatureDefinitionORM.name == feature.name,
                    FeatureDefinitionORM.version == feature.version,
                )
                .values(
                    description=feature.description,
                    status=feature.status.value,
                    tags=feature.tags,
                    updated_at=datetime.utcnow(),
                )
            )
            return feature
        else:
            orm = FeatureDefinitionORM(
                id=feature.id,
                name=feature.name,
                category=feature.category.value,
                description=feature.description,
                data_type=feature.data_type,
                version=feature.version,
                status=feature.status.value,
                tags=feature.tags,
            )
            self._session.add(orm)
            return feature

    async def save_materialization_job(self, job: MaterializationJob) -> MaterializationJob:
        existing = await self._session.get(MaterializationJobORM, job.job_id)
        if existing:
            existing.status = job.status.value
            existing.total_entities = job.total_entities
            existing.processed_entities = job.processed_entities
            existing.failed_entities = job.failed_entities
            existing.started_at = job.started_at
            existing.completed_at = job.completed_at
        else:
            orm = MaterializationJobORM(
                id=job.job_id,
                tenant_id=job.tenant_id,
                lookback_days=job.lookback_days,
                status=job.status.value,
                triggered_by=job.triggered_by,
            )
            self._session.add(orm)
        return job

    async def get_materialization_job(self, job_id: UUID) -> Optional[MaterializationJob]:
        orm = await self._session.get(MaterializationJobORM, job_id)
        return self._mat_orm_to_domain(orm) if orm else None

    async def list_recent_jobs(self, limit: int = 10) -> List[MaterializationJob]:
        result = await self._session.execute(
            select(MaterializationJobORM)
            .order_by(MaterializationJobORM.created_at.desc())
            .limit(limit)
        )
        return [self._mat_orm_to_domain(r) for r in result.scalars().all()]

    async def save_drift_report(self, report: DriftReport) -> None:
        orm = DriftReportORM(
            id=report.report_id,
            drift_type=report.drift_type.value,
            feature_name=report.feature_name,
            kl_divergence=report.kl_divergence,
            js_divergence=report.js_divergence,
            is_drifted=report.is_drifted,
            baseline_mean=report.baseline_mean,
            current_mean=report.current_mean,
            baseline_std=report.baseline_std,
            current_std=report.current_std,
            sample_count=report.sample_count,
            metadata=report.metadata,
            computed_at=report.computed_at,
        )
        self._session.add(orm)

    async def get_latest_drift_reports(
        self,
        feature_name: Optional[str] = None,
        limit: int = 50,
    ) -> List[DriftReport]:
        stmt = select(DriftReportORM).order_by(DriftReportORM.computed_at.desc()).limit(limit)
        if feature_name:
            stmt = stmt.where(DriftReportORM.feature_name == feature_name)
        result = await self._session.execute(stmt)
        return [self._drift_orm_to_domain(r) for r in result.scalars().all()]

    # ─── Private Mapping Helpers ─────────────────────────────────────────────

    @staticmethod
    def _orm_to_domain(orm: FeatureDefinitionORM) -> FeatureDefinition:
        return FeatureDefinition(
            id=orm.id,
            name=orm.name,
            category=FeatureCategory(orm.category),
            description=orm.description,
            data_type=orm.data_type,
            version=orm.version,
            status=FeatureStatus(orm.status),
            tags=orm.tags or [],
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    @staticmethod
    def _mat_orm_to_domain(orm: MaterializationJobORM) -> MaterializationJob:
        return MaterializationJob(
            job_id=orm.id,
            tenant_id=orm.tenant_id,
            lookback_days=orm.lookback_days,
            status=MaterializationStatus(orm.status),
            total_entities=orm.total_entities,
            processed_entities=orm.processed_entities,
            failed_entities=orm.failed_entities,
            started_at=orm.started_at,
            completed_at=orm.completed_at,
            triggered_by=orm.triggered_by,
        )

    @staticmethod
    def _drift_orm_to_domain(orm: DriftReportORM) -> DriftReport:
        return DriftReport(
            report_id=orm.id,
            drift_type=DriftType(orm.drift_type),
            feature_name=orm.feature_name,
            kl_divergence=orm.kl_divergence,
            js_divergence=orm.js_divergence,
            is_drifted=orm.is_drifted,
            baseline_mean=orm.baseline_mean,
            current_mean=orm.current_mean,
            baseline_std=orm.baseline_std,
            current_std=orm.current_std,
            sample_count=orm.sample_count,
            metadata=orm.metadata or {},
            computed_at=orm.computed_at,
        )
