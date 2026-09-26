"""
feature-store-service/src/workers/materialization.py

Batch materialization worker: computes features for ALL entity pairs
and pushes to online store (Redis).
Scheduled: Daily at 2 AM UTC (configurable).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from src.core.config import settings
from src.core.metrics import (
    ACTIVE_MATERIALIZATION_JOBS, MATERIALIZATION_DURATION,
    MATERIALIZATION_RECORDS, FEATURE_DRIFT_SCORE, DRIFT_ALERTS,
)
from src.domain.models import (
    DriftReport, DriftType, EntitySnapshot, MaterializationJob,
    MaterializationStatus, OfflineFeatureRequest,
)
from src.repositories.online_store import OnlineFeatureStore
from src.repositories.offline_store import OfflineFeatureStore
from src.repositories.feature_registry import FeatureRegistryRepository
from src.services.feature_computation import FeatureComputationService

logger = logging.getLogger(__name__)


class MaterializationWorker:
    """
    Batch materialization: reads entity data, computes 50-dim feature vectors,
    and writes to Redis online store + S3 offline store.
    """

    def __init__(
        self,
        online_store: OnlineFeatureStore,
        offline_store: OfflineFeatureStore,
        computation_service: FeatureComputationService,
        registry: FeatureRegistryRepository,
    ):
        self._online = online_store
        self._offline = offline_store
        self._compute = computation_service
        self._registry = registry

    async def run_job(self, job: MaterializationJob) -> MaterializationJob:
        """
        Execute a materialization job.
        In production, entity pairs come from the MDM entity service.
        Here we demonstrate the full pipeline with the data loading abstracted.
        """
        start_time = datetime.utcnow()
        job.status = MaterializationStatus.RUNNING
        job.started_at = start_time
        ACTIVE_MATERIALIZATION_JOBS.inc()

        try:
            await self._registry.save_materialization_job(job)
            logger.info(f"Materialization job {job.job_id} started for tenant {job.tenant_id}")

            # In production: stream entity pairs from MDM data plane
            # For now: demonstrate pipeline with simulated data
            batch_size = 1000
            total_processed = 0
            total_failed = 0

            # Simulate processing in batches
            entity_pair_batches = await self._load_entity_pairs(
                job.tenant_id,
                job.lookback_days,
                batch_size,
            )

            for batch in entity_pair_batches:
                batch_fvs = []
                for e1, e2 in batch:
                    try:
                        fv = self._compute.compute(e1, e2, settings.FEATURE_VERSION)
                        batch_fvs.append(fv)
                        total_processed += 1
                    except Exception as ex:
                        logger.warning(f"Feature computation failed for pair: {ex}")
                        total_failed += 1

                # Write batch to online store
                await self._online.set_batch(batch_fvs)

                # Write batch to offline store
                try:
                    self._offline.write_features(batch_fvs)
                except Exception as ex:
                    logger.error(f"Offline store write failed: {ex}")

                MATERIALIZATION_RECORDS.labels(status="success").inc(total_processed)
                MATERIALIZATION_RECORDS.labels(status="failed").inc(total_failed)

                logger.info(
                    f"Job {job.job_id}: processed {total_processed} pairs "
                    f"({total_failed} failed)"
                )
                await asyncio.sleep(0)  # Yield to event loop

            # Finalize job
            job.status = MaterializationStatus.COMPLETED
            job.processed_entities = total_processed
            job.failed_entities = total_failed
            job.total_entities = total_processed + total_failed
            job.completed_at = datetime.utcnow()

            duration = (job.completed_at - start_time).total_seconds()
            MATERIALIZATION_DURATION.labels(job_name="entity_features").observe(duration)
            logger.info(
                f"Materialization job {job.job_id} COMPLETED: "
                f"{total_processed} processed, {total_failed} failed in {duration:.1f}s"
            )

        except Exception as e:
            job.status = MaterializationStatus.FAILED
            job.completed_at = datetime.utcnow()
            logger.error(f"Materialization job {job.job_id} FAILED: {e}", exc_info=True)
        finally:
            ACTIVE_MATERIALIZATION_JOBS.dec()
            await self._registry.save_materialization_job(job)

        return job

    async def _load_entity_pairs(
        self,
        tenant_id: Optional[str],
        lookback_days: int,
        batch_size: int,
    ):
        """
        Load entity pairs to materialize features for.
        Production: queries MDM entity service for all active entities.
        Returns batches of (EntitySnapshot, EntitySnapshot) tuples.
        """
        # In production, this would call the MDM entity service API
        # and retrieve pairs of entities that need feature computation.
        # For now, returns empty iterator.
        return iter([])


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
