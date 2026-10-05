"""
feature-store-service/src/workers/materialization.py

Batch materialization worker: takes entity pairs from a configured
MaterializationSource, computes the 50 features, and writes them to the offline store
(S3/Parquet) and the online store (Redis). Triggered by POST /api/v1/materialize or the
daily scheduler, both through run_materialization_job().
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from src.core.config import settings
from src.core.metrics import (
    ACTIVE_MATERIALIZATION_JOBS, MATERIALIZATION_DURATION,
    MATERIALIZATION_RECORDS, FEATURE_DRIFT_SCORE, DRIFT_ALERTS,
)
from src.adapters.materialization_sources import (
    MaterializationSource, MaterializationSourceError, MaterializationSourceNotConfigured,
    PairBatch, build_materialization_source,
)
from src.domain.models import (
    DriftReport, DriftType, EntitySnapshot, FeatureVector, MaterializationJob,
    MaterializationStatus, OfflineFeatureRequest,
)
from src.repositories.online_store import OnlineFeatureStore
from src.repositories.offline_store import FEATURE_COLUMN_NAMES, OfflineFeatureStore
from src.repositories.feature_registry import FeatureRegistryRepository
from src.domain.feature_catalog import is_canonical, non_canonical_names
from src.services.feature_computation import EmbeddingUnavailableError, FeatureComputationService
from src.workers.job_lock import InProcessJobLock, JobLock, PostgresJobLock

logger = logging.getLogger(__name__)


class MaterializationFailed(RuntimeError):
    """A store write failed; the job must not report COMPLETED."""


class MaterializationAlreadyRunning(RuntimeError):
    """Another job holds the lock for this tenant (or for all tenants)."""


INTERRUPTED_MESSAGE = (
    "Interrupted: the service stopped while this job was running. Start a new job — "
    "vectors already written are reused, not recomputed."
)


def pair_id_of(entity_id_1: str, entity_id_2: str) -> str:
    """Order-independent pair id, identical to the offline store's pair_id column."""
    a, b = sorted((entity_id_1, entity_id_2))
    return f"{a}:{b}"


class MaterializationWorker:
    """
    Batch materialization: pulls entity pairs from a MaterializationSource, computes
    the 50-dim feature vector for each with the real FeatureComputationService, and
    writes every vector to BOTH the offline store (S3/Parquet — authoritative for
    training) and the online store (Redis).

    Job contract:
      COMPLETED  every pair the source resolved has a vector in both stores. Pairs
                 whose computation failed, or that the source could not resolve, are
                 counted (failed_entities / skipped_entities) and logged.
      FAILED     no source configured, another job is running for the tenant, a store
                 read/write failed, or not a single pair could be materialized.
                 error_message says which.

    Idempotent and resumable: when the source declares a `source_id` (the exact input,
    e.g. a dataset id), a pair that already has an offline vector computed from that
    same input under the current feature version is NOT recomputed and NOT written to
    the offline store again — the stored vector is copied to the online store. So
    running a job twice leaves one offline row per pair, a job restarted after a crash
    continues where the first one stopped, and an emptied online store is restored
    without recomputation.

    One job per tenant: the JobLock is held for the whole run.

    The worker knows nothing about where pairs come from — see
    src/adapters/materialization_sources.py.
    """

    def __init__(
        self,
        online_store: OnlineFeatureStore,
        offline_store: OfflineFeatureStore,
        computation_service: FeatureComputationService,
        registry: FeatureRegistryRepository,
        source: Optional[MaterializationSource] = None,
        lock: Optional[JobLock] = None,
    ):
        self._online = online_store
        self._offline = offline_store
        self._compute = computation_service
        self._registry = registry
        self._source = source
        self._lock = lock or InProcessJobLock()
        self.reused = 0   # vectors taken from the offline store instead of recomputed

    async def run_job(self, job: MaterializationJob) -> MaterializationJob:
        start_time = datetime.utcnow()
        job.error_message = None
        ACTIVE_MATERIALIZATION_JOBS.inc()

        processed = failed = skipped = 0
        self.reused = 0
        handle = None
        try:
            handle = await self._lock.acquire(job.tenant_id)
            if handle is None:
                raise MaterializationAlreadyRunning(
                    f"Another materialization job is already running for "
                    f"{'tenant ' + job.tenant_id if job.tenant_id else 'all tenants'}; "
                    f"this job was not started."
                )
            job.status = MaterializationStatus.RUNNING
            job.started_at = start_time
            await self._persist(job)
            if self._source is None:
                raise MaterializationSourceNotConfigured(
                    "No materialization data source is configured."
                )
            source_id = getattr(self._source, "source_id", None)
            logger.info(
                f"Materialization job {job.job_id} started "
                f"(source={self._source.name}, source_id={source_id or 'none'}, "
                f"tenant={job.tenant_id or 'all'})"
            )

            existing_by_tenant: dict = {}
            batches = self._source.iter_batches(
                job.tenant_id, job.lookback_days, settings.MATERIALIZATION_BATCH_SIZE
            )
            for batch in batches:
                skipped += len(batch.skipped)
                for sk in batch.skipped[:5]:
                    logger.warning(
                        f"Job {job.job_id}: skipped {sk.entity_id_1}:{sk.entity_id_2} "
                        f"(tenant {sk.tenant_id}) — {sk.reason}"
                    )

                # Vectors this exact input already produced (idempotency / resume).
                if source_id and batch.tenant_id not in existing_by_tenant:
                    existing_by_tenant[batch.tenant_id] = await asyncio.to_thread(
                        self._load_existing, batch.tenant_id, source_id
                    )
                existing = existing_by_tenant.get(batch.tenant_id, {})
                reuse = [
                    existing[pid] for e1, e2 in batch.pairs
                    if (pid := pair_id_of(e1.entity_id, e2.entity_id)) in existing
                    and e1.tenant_id == batch.tenant_id and e2.tenant_id == batch.tenant_id
                ]
                todo = PairBatch(tenant_id=batch.tenant_id, pairs=[
                    (e1, e2) for e1, e2 in batch.pairs
                    if pair_id_of(e1.entity_id, e2.entity_id) not in existing
                    or e1.tenant_id != batch.tenant_id or e2.tenant_id != batch.tenant_id
                ])

                # CPU-bound (~75 ms/pair with embeddings): run off the event loop so
                # the service keeps answering requests while a job runs.
                vectors, batch_failed = await asyncio.to_thread(self._compute_batch, todo)
                failed += batch_failed

                if vectors:
                    self._write_offline(vectors, batch.tenant_id, processed, source_id)
                if vectors or reuse:
                    await self._write_online(vectors + reuse, batch.tenant_id, processed)
                processed += len(vectors) + len(reuse)
                self.reused += len(reuse)

                # Per-batch deltas. (Previously the running totals were re-added every
                # batch, inflating these counters.)
                MATERIALIZATION_RECORDS.labels(status="success").inc(len(vectors))
                MATERIALIZATION_RECORDS.labels(status="reused").inc(len(reuse))
                MATERIALIZATION_RECORDS.labels(status="failed").inc(batch_failed)
                MATERIALIZATION_RECORDS.labels(status="skipped").inc(len(batch.skipped))

                self._set_counts(job, processed, failed, skipped)
                await self._persist(job)
                logger.info(
                    f"Job {job.job_id}: tenant {batch.tenant_id} — {processed} in both stores "
                    f"({self.reused} reused), {failed} failed, {skipped} skipped so far"
                )

            if processed == 0 and (failed or skipped):
                raise MaterializationFailed(
                    f"No pairs were materialized: {failed} failed computation, "
                    f"{skipped} skipped by the source."
                )
            if processed == 0:
                logger.warning(
                    f"Job {job.job_id}: the source returned no pairs "
                    f"(tenant={job.tenant_id or 'all'})."
                )
            if failed or skipped:
                logger.warning(
                    f"Job {job.job_id} completed with {failed} computation failure(s) "
                    f"and {skipped} skipped pair(s); see warnings above."
                )
            job.status = MaterializationStatus.COMPLETED

        except (MaterializationFailed, MaterializationSourceNotConfigured,
                MaterializationSourceError, MaterializationAlreadyRunning) as e:
            job.status = MaterializationStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Materialization job {job.job_id} FAILED: {e}")
        except Exception as e:
            job.status = MaterializationStatus.FAILED
            job.error_message = f"{type(e).__name__}: {e}"
            logger.error(f"Materialization job {job.job_id} FAILED: {e}", exc_info=True)
        finally:
            self._set_counts(job, processed, failed, skipped)
            job.started_at = job.started_at or start_time
            job.completed_at = datetime.utcnow()
            duration = (job.completed_at - start_time).total_seconds()
            MATERIALIZATION_DURATION.labels(job_name="entity_features").observe(duration)
            ACTIVE_MATERIALIZATION_JOBS.dec()
            try:
                await self._persist(job)
            finally:
                await self._lock.release(handle)
            logger.info(
                f"Materialization job {job.job_id} {job.status.value}: {processed} in both "
                f"stores ({self.reused} reused), {failed} failed, {skipped} skipped in {duration:.1f}s"
            )

        return job

    # ── helpers ───────────────────────────────────────────────────────────

    def _load_existing(self, tenant_id: str, source_id: str) -> dict:
        try:
            existing = self._offline.existing_vectors(tenant_id, settings.FEATURE_VERSION, source_id)
        except Exception as ex:
            # Not knowing what is already stored must stop the job: carrying on would
            # write every vector a second time.
            raise MaterializationFailed(
                f"Could not read existing offline vectors for tenant {tenant_id}: "
                f"{type(ex).__name__}: {ex}"
            ) from ex
        if existing:
            logger.info(
                f"Tenant {tenant_id}: {len(existing)} vectors already stored for source "
                f"{source_id}; they will be reused"
            )
        return existing

    def _compute_batch(self, batch: PairBatch) -> Tuple[List[FeatureVector], int]:
        """Compute vectors for one single-tenant batch. Returns (vectors, n_failed).
        Runs in a worker thread."""
        vectors: List[FeatureVector] = []
        failed = 0
        for e1, e2 in batch.pairs:
            if e1.tenant_id != batch.tenant_id or e2.tenant_id != batch.tenant_id:
                # Sources must never produce this; refuse rather than mis-file it.
                logger.error(
                    f"Refusing cross-tenant pair {e1.entity_id}:{e2.entity_id} in a "
                    f"batch for tenant {batch.tenant_id}"
                )
                failed += 1
                continue
            try:
                fv = self._compute.compute(e1, e2, settings.FEATURE_VERSION)
            except EmbeddingUnavailableError as ex:
                # Not a property of this pair: every pair would fail the same way.
                raise MaterializationFailed(
                    f"Embedding model unavailable — no semantic features can be computed: {ex}"
                ) from ex
            except Exception as ex:
                logger.warning(
                    f"Feature computation failed for {e1.entity_id}:{e2.entity_id}: "
                    f"{type(ex).__name__}: {ex}"
                )
                failed += 1
                continue
            # compute() guarantees the catalog; checked again because a vector with
            # other names must never reach a store.
            if not is_canonical(fv.features.keys()):
                logger.warning(
                    f"Non-canonical feature names for {e1.entity_id}:{e2.entity_id} "
                    f"(e.g. {non_canonical_names(fv.features.keys())[:3]}); not stored"
                )
                failed += 1
                continue
            vectors.append(fv)
        return vectors, failed

    def _write_offline(self, vectors: List[FeatureVector], tenant_id: str, already: int,
                       source_id: Optional[str] = None) -> None:
        try:
            self._offline.write_features(
                vectors, data_source=self._source.name, source_id=source_id)
        except Exception as ex:
            raise MaterializationFailed(
                f"Offline store write failed for tenant {tenant_id} "
                f"({len(vectors)} vectors; {already} written earlier in this job): "
                f"{type(ex).__name__}: {ex}"
            ) from ex

    async def _write_online(self, vectors: List[FeatureVector], tenant_id: str, already: int) -> None:
        written = await self._online.set_batch(
            vectors, ttl_hours=settings.MATERIALIZED_FEATURE_TTL_HOURS)
        if written != len(vectors):
            # set_batch reports Redis errors as a short count rather than raising.
            raise MaterializationFailed(
                f"Online store wrote {written}/{len(vectors)} vectors for tenant "
                f"{tenant_id} ({already} written earlier in this job; this batch IS in "
                f"the offline store)."
            )

    @staticmethod
    def _set_counts(job: MaterializationJob, processed: int, failed: int, skipped: int) -> None:
        job.processed_entities = processed
        job.failed_entities = failed
        job.total_entities = processed + failed + skipped

    async def _persist(self, job: MaterializationJob) -> None:
        await self._registry.save_materialization_job(job)
        await self._registry.commit()


def default_job_lock() -> JobLock:
    """The cross-process lock (PostgreSQL advisory locks on the registry database)."""
    from src.repositories.feature_registry import get_engine

    return PostgresJobLock(get_engine())


async def run_materialization_job(
    job: MaterializationJob,
    online_store: Optional[OnlineFeatureStore],
    computation_service: FeatureComputationService,
    lock: Optional[JobLock] = None,
) -> MaterializationJob:
    """
    The single entry point for running a job, used by BOTH the API and the scheduler.

    Owns its own DB session. The API used to hand the worker the request-scoped
    session, but FastAPI (>= 0.106) closes yield-dependencies BEFORE background tasks
    run, so job status updates were never committed; the scheduler path opened a
    session but never committed at all.
    """
    from src.repositories.feature_registry import get_session_factory

    async with get_session_factory()() as session:
        registry = FeatureRegistryRepository(session)

        problem: Optional[str] = None
        source: Optional[MaterializationSource] = None
        if online_store is None:
            problem = "Online store (Redis) is unavailable."
        else:
            try:
                source = build_materialization_source(settings)
            except (MaterializationSourceNotConfigured, ValueError) as exc:
                problem = str(exc)

        if problem:
            now = datetime.utcnow()
            job.status = MaterializationStatus.FAILED
            job.error_message = problem
            job.started_at = job.started_at or now
            job.completed_at = now
            await registry.save_materialization_job(job)
            await registry.commit()
            logger.error(f"Materialization job {job.job_id} FAILED before start: {problem}")
            return job

        worker = MaterializationWorker(
            online_store=online_store,
            offline_store=OfflineFeatureStore(),
            computation_service=computation_service,
            registry=registry,
            source=source,
            lock=lock or default_job_lock(),
        )
        return await worker.run_job(job)


async def recover_interrupted_jobs(
    registry: FeatureRegistryRepository,
    lock: JobLock,
    pending_grace_seconds: int = 120,
) -> List[MaterializationJob]:
    """
    Mark jobs that are recorded as RUNNING/PENDING but that nobody is running as FAILED.

    A job that is really running holds its tenant lock, so "the lock is free" proves the
    process that owned the job is gone. Without this a job interrupted by a restart
    stayed RUNNING forever. PENDING jobs get a grace period: one created a moment ago
    may simply not have taken its lock yet.
    Returns the jobs it closed.
    """
    closed: List[MaterializationJob] = []
    for job in await registry.list_unfinished_jobs(pending_grace_seconds):
        handle = await lock.acquire(job.tenant_id)
        if handle is None:
            continue          # somebody holds the lock: the job is alive
        try:
            job.status = MaterializationStatus.FAILED
            job.error_message = INTERRUPTED_MESSAGE
            job.completed_at = datetime.utcnow()
            await registry.save_materialization_job(job)
            await registry.commit()
            closed.append(job)
            logger.warning(f"Materialization job {job.job_id} marked FAILED: interrupted by a restart")
        finally:
            await lock.release(handle)
    return closed


# ─── Drift Detection Worker ───────────────────────────────────────────────────

class DriftDetectionWorker:
    """
    Monitors feature distributions for statistical drift.
    Runs on configurable interval (default: 1 hour).

    Drift Types Monitored:
    - DATA_DRIFT: KL divergence on feature distributions > 0.1
    - PREDICTION_DRIFT: Jensen-Shannon divergence > 0.05
    """

    def __init__(
        self,
        offline_store: OfflineFeatureStore,
        registry: FeatureRegistryRepository,
    ):
        self._offline = offline_store
        self._registry = registry

    async def run_drift_check(self, tenant_id: str) -> List[DriftReport]:
        """
        Run drift detection for all features across a tenant.
        Compares baseline (14-7 days ago) vs current (last 7 days).
        """
        logger.info(f"Running drift check for tenant {tenant_id}")

        try:
            baseline_df, current_df = self._offline.read_for_drift_analysis(
                tenant_id=tenant_id,
                baseline_days=14,
                current_days=7,
            )

            if baseline_df.empty or current_df.empty:
                logger.warning(f"Insufficient data for drift check (tenant {tenant_id})")
                return []

        except Exception as e:
            logger.error(f"Failed to load data for drift check: {e}")
            return []

        feature_cols = [c for c in baseline_df.columns if c.startswith("feat_")]
        reports = []

        for col in feature_cols:
            feature_name = col.replace("feat_", "")
            try:
                report = self._compute_drift(
                    feature_name=feature_name,
                    baseline=baseline_df[col].dropna().values,
                    current=current_df[col].dropna().values,
                )
                if report:
                    reports.append(report)
                    FEATURE_DRIFT_SCORE.labels(feature_name=feature_name).set(
                        report.kl_divergence
                    )
                    if report.is_drifted:
                        DRIFT_ALERTS.labels(
                            drift_type=report.drift_type.value,
                            feature_name=feature_name,
                        ).inc()
                        logger.warning(
                            f"DRIFT DETECTED: {feature_name} "
                            f"KL={report.kl_divergence:.4f} "
                            f"JS={report.js_divergence:.4f}"
                        )

                    await self._registry.save_drift_report(report)

            except Exception as e:
                logger.error(f"Drift check failed for feature {feature_name}: {e}")

        drifted = [r for r in reports if r.is_drifted]
        logger.info(
            f"Drift check complete: {len(drifted)}/{len(reports)} features drifted"
        )
        return reports

    def _compute_drift(
        self,
        feature_name: str,
        baseline: "np.ndarray",
        current: "np.ndarray",
    ) -> Optional[DriftReport]:
        """
        Compute KL divergence and Jensen-Shannon divergence between
        baseline and current feature distributions.
        """
        import numpy as np
        from scipy.stats import entropy
        from scipy.spatial.distance import jensenshannon

        if len(baseline) < 100 or len(current) < 100:
            return None  # Not enough samples

        # Build histograms over shared range
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins = 50

        baseline_hist, _ = np.histogram(baseline, bins=bins, range=(min_val, max_val), density=True)
        current_hist, _ = np.histogram(current, bins=bins, range=(min_val, max_val), density=True)

        # Smooth to avoid log(0)
        eps = 1e-10
        baseline_hist = baseline_hist + eps
        current_hist = current_hist + eps
        baseline_hist /= baseline_hist.sum()
        current_hist /= current_hist.sum()

        kl_div = float(entropy(current_hist, baseline_hist))
        js_div = float(jensenshannon(baseline_hist, current_hist))

        is_drifted = (
            kl_div > settings.DRIFT_KL_THRESHOLD
            or js_div > settings.DRIFT_JS_THRESHOLD
        )

        return DriftReport(
            drift_type=DriftType.DATA_DRIFT,
            feature_name=feature_name,
            kl_divergence=kl_div,
            js_divergence=js_div,
            is_drifted=is_drifted,
            baseline_mean=float(np.mean(baseline)),
            current_mean=float(np.mean(current)),
            baseline_std=float(np.std(baseline)),
            current_std=float(np.std(current)),
            sample_count=len(current),
        )
