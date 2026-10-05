"""
feature-store-service/src/repositories/offline_store.py

S3 + Parquet offline feature store.
Used for: Model training (point-in-time correct features), batch analysis,
          feature drift computation, and materialization jobs.

Partitioning: s3://{bucket}/features/{tenant_id}/year={Y}/month={M}/day={D}/features.parquet
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional
from uuid import uuid4

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import BotoCoreError, ClientError

from src.core.config import settings
from src.domain.feature_catalog import FEATURE_NAMES
from src.domain.models import FeatureVector, OfflineFeatureRequest

logger = logging.getLogger(__name__)


def _to_naive_utc(dt: datetime) -> datetime:
    """Normalize to naive UTC — this store's convention (utcnow(), stripped S3 LastModified).
    Comparing a tz-aware client timestamp with naive values raises TypeError."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


class OfflineStoreError(RuntimeError):
    """S3 could not be listed or read. Never reported as "no features found"."""


# Canonical feature ordering — defined once in src/domain/feature_catalog.py and shared
# by the Parquet schema below and the /features/offline response. Matches
# FeatureVector.as_list, which orders by sorted(features.keys()), so offline and online
# vectors align.
FEATURE_COLUMN_NAMES: List[str] = list(FEATURE_NAMES)


# ─── Arrow Schema: Fixed schema for all feature parquet files ─────────────────

FEATURE_SCHEMA = pa.schema([
    pa.field("entity_id_1", pa.string()),
    pa.field("entity_id_2", pa.string()),
    pa.field("tenant_id", pa.string()),
    pa.field("pair_id", pa.string()),
    pa.field("feature_version", pa.string()),
    pa.field("computed_at", pa.timestamp("ms", tz="UTC")),
    pa.field("computation_ms", pa.float32()),
    # 50 feature columns (all float32 for storage efficiency)
    *[pa.field(f"feat_{name}", pa.float32()) for name in FEATURE_COLUMN_NAMES],
])




class OfflineFeatureStore:
    """
    S3-backed Parquet offline store for batch training data retrieval.
    Supports point-in-time correct feature retrieval to prevent training-serving skew.
    """

    def __init__(self):
        self._s3 = boto3.client(
            "s3",
            region_name=settings.S3_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            endpoint_url=settings.S3_ENDPOINT_URL,  # LocalStack support
        )
        self._bucket = settings.S3_BUCKET
        self._prefix = settings.S3_PREFIX

    def write_features(
        self,
        feature_vectors: List[FeatureVector],
        partition_dt: Optional[datetime] = None,
        data_source: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> str:
        """
        Write a batch of feature vectors to S3 as Parquet. Returns the S3 path written.

        * All vectors must belong to ONE tenant — the object is filed under that
          tenant's prefix, so a mixed batch would expose one tenant's rows in another's
          partition. Mixed input raises instead of being written.
        * The partition timestamp defaults to the EARLIEST computed_at in the batch, not
          the write time. Point-in-time reads prune files by partition timestamp; a
          partition later than some of its rows would hide those rows from an as_of
          that falls between computed_at and the write.
        * The object key carries a random suffix so two batches in the same second
          never overwrite each other (S3 PUT replaces an existing key silently).
        * `data_source` and `source_id` (the exact input it was computed from, e.g. a
          dataset id) are recorded as object metadata for provenance; source_id is what
          lets a repeated materialization recognise vectors it has already written.
        """
        if not feature_vectors:
            return ""

        tenants = {fv.tenant_id for fv in feature_vectors}
        if len(tenants) != 1:
            raise ValueError(
                f"write_features received vectors for {len(tenants)} tenants "
                f"{sorted(tenants)}; a batch must be single-tenant."
            )

        dt = partition_dt or min(_to_naive_utc(fv.computed_at) for fv in feature_vectors)
        s3_key = self._make_s3_key(feature_vectors[0].tenant_id, dt, unique=True)

        rows = [self._fv_to_row(fv) for fv in feature_vectors]
        df = pd.DataFrame(rows)

        # Write to Parquet in memory then upload
        buffer = io.BytesIO()
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table,
            buffer,
            compression="snappy",
            row_group_size=100000,
        )
        buffer.seek(0)

        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType="application/octet-stream",
                Metadata={
                    "feature_version": feature_vectors[0].feature_version,
                    "row_count": str(len(feature_vectors)),
                    "tenant_id": feature_vectors[0].tenant_id,
                    "data_source": data_source or "unspecified",
                    "source_id": source_id or "unspecified",
                },
            )
            logger.info(f"Wrote {len(feature_vectors)} features to s3://{self._bucket}/{s3_key}")
            return f"s3://{self._bucket}/{s3_key}"

        except (BotoCoreError, ClientError) as e:
            logger.error(f"S3 write error: {e}")
            raise

    def read_point_in_time(
        self,
        request: OfflineFeatureRequest,
    ) -> pd.DataFrame:
        """
        Point-in-time correct feature retrieval.
        Returns features as they existed at `as_of_timestamp`.
        This is critical for training-serving skew prevention.
        """
        # Clients may send tz-aware timestamps ("...Z"); S3 keys are compared as naive UTC.
        as_of = _to_naive_utc(request.as_of_timestamp)

        # Find all Parquet files up to as_of_timestamp
        keys = self._list_keys_before(
            tenant_id=request.tenant_id,
            before_dt=as_of,
        )

        if not keys:
            logger.warning(f"No features found for tenant {request.tenant_id} before {request.as_of_timestamp}")
            return pd.DataFrame()

        # Requested pairs, order-independent (pair_id = "min:max").
        wanted = {":".join(sorted((e1, e2))) for e1, e2 in request.entity_pairs}

        # Read and filter Parquet files. A file that cannot be read FAILS the request:
        # skipping it used to return fewer rows with no sign that data was missing.
        frames = []
        for key in keys:
            try:
                df = self._read_parquet_from_s3(key)
            except Exception as e:
                logger.error(f"Error reading {key}: {e}")
                raise OfflineStoreError(f"cannot read offline object {key}: {type(e).__name__}") from e
            filtered = df[df["pair_id"].isin(wanted)]
            if not filtered.empty:
                frames.append(filtered)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Point-in-time: for each pair, keep the LATEST feature vector
        # that was computed BEFORE as_of_timestamp
        # computed_at may be read back tz-aware (schema declares tz="UTC") or naive;
        # compare both sides as UTC-aware so either form works.
        computed_at_utc = pd.to_datetime(combined["computed_at"], utc=True)
        combined = combined[computed_at_utc <= pd.Timestamp(as_of, tz="UTC")]
        # Only the requested feature catalog version: rows of another version are a
        # different feature definition and must not be mixed into one training set.
        if request.feature_version:
            combined = combined[combined["feature_version"] == request.feature_version]
        combined = combined.sort_values("computed_at", ascending=False)
        # pair_id is order-independent, so A:B and B:A are one pair.
        combined = combined.drop_duplicates(subset=["pair_id"], keep="first")

        logger.info(
            f"Retrieved {len(combined)} point-in-time feature vectors "
            f"for {len(request.entity_pairs)} requested pairs"
        )
        return combined

    def read_for_drift_analysis(
        self,
        tenant_id: str,
        baseline_days: int = 14,
        current_days: int = 7,
    ) -> tuple:
        """
        Read two time windows of features for drift analysis.
        Returns (baseline_df, current_df).
        """
        now = datetime.utcnow()
        baseline_start = now - timedelta(days=baseline_days)
        baseline_end = now - timedelta(days=current_days)
        current_start = baseline_end

        baseline_keys = self._list_keys_in_range(tenant_id, baseline_start, baseline_end)
        current_keys = self._list_keys_in_range(tenant_id, current_start, now)

        baseline_df = self._read_multiple(baseline_keys)
        current_df = self._read_multiple(current_keys)

        return baseline_df, current_df

    def stream_batches(
        self,
        tenant_id: str,
        start_dt: datetime,
        end_dt: datetime,
        batch_size: int = 10000,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream feature vectors in batches. Used for materialization jobs.
        """
        keys = self._list_keys_in_range(tenant_id, start_dt, end_dt)

        for key in keys:
            try:
                df = self._read_parquet_from_s3(key)
                # Yield in chunks
                for i in range(0, len(df), batch_size):
                    yield df.iloc[i:i + batch_size]
            except Exception as e:
                logger.error(f"Error streaming {key}: {e}")
                continue

    # ─── Private Helpers ─────────────────────────────────────────────────────

    def _make_s3_key(self, tenant_id: str, dt: datetime, unique: bool = False) -> str:
        suffix = f"_{uuid4().hex[:12]}" if unique else ""
        return (
            f"{self._prefix}/{tenant_id}/"
            f"year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/"
            f"features_{dt.strftime('%H%M%S')}{suffix}.parquet"
        )

    def _list_keys_before(
        self,
        tenant_id: str,
        before_dt: datetime,
    ) -> List[str]:
        prefix = f"{self._prefix}/{tenant_id}/"
        return self._list_keys_in_range(
            tenant_id,
            datetime(2020, 1, 1),
            before_dt,
        )

    @staticmethod
    def _partition_dt_from_key(key: str) -> Optional[datetime]:
        """
        Parse the partition timestamp encoded in the object key:
            features/{tenant}/year=YYYY/month=MM/day=DD/features_HHMMSS[_<hex>].parquet
        The optional hex suffix makes keys unique per write. Returns None if the key
        does not follow that layout.
        """
        match = re.search(
            r"year=(\d{4})/month=(\d{2})/day=(\d{2})/features_(\d{2})(\d{2})(\d{2})(?:_[0-9a-f]+)?\.parquet$",
            key,
        )
        if not match:
            return None
        year, month, day, hour, minute, second = (int(g) for g in match.groups())
        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError:
            return None

    def _list_keys_in_range(
        self,
        tenant_id: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[str]:
        """
        List S3 keys whose PARTITION timestamp falls in [start_dt, end_dt].

        The partition timestamp comes from the key path, which encodes when the
        features were computed. S3's LastModified is when the file was uploaded,
        which is a different thing: a backfill written today can contain features
        computed weeks ago, and pruning on upload time would hide it from a
        point-in-time read (and, conversely, admit late-written rows into a window
        they do not belong to). Row-level `computed_at` filtering in
        read_point_in_time() remains the authoritative correctness check; this is
        the partition-pruning optimisation in front of it.

        Keys that predate this layout fall back to LastModified so older data is
        still readable rather than silently dropped.
        """
        prefix = f"{self._prefix}/{tenant_id}/"
        keys = []

        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    partition_dt = self._partition_dt_from_key(key)
                    if partition_dt is None:
                        partition_dt = obj["LastModified"].replace(tzinfo=None)
                        logger.debug(
                            "Key %s has no partition timestamp; falling back to LastModified",
                            key,
                        )
                    if start_dt <= partition_dt <= end_dt:
                        keys.append(key)
        except (BotoCoreError, ClientError) as e:
            # Used to be logged and answered with an empty list — "no features".
            logger.error(f"S3 LIST error: {e}")
            raise OfflineStoreError(f"cannot list offline store: {type(e).__name__}") from e

        return sorted(keys)

    def ping(self) -> bool:
        """True if the offline bucket is reachable (real check for the health endpoint)."""
        try:
            self._s3.head_bucket(Bucket=self._bucket)
            return True
        except (BotoCoreError, ClientError) as e:
            logger.warning(f"Offline store not reachable: {type(e).__name__}")
            return False

    def existing_vectors(
        self,
        tenant_id: str,
        feature_version: str,
        source_id: str,
    ) -> dict:
        """
        Vectors already stored for this tenant that were computed from EXACTLY this
        input (`source_id`) under this feature catalog version, as
        {pair_id: FeatureVector} (latest per pair).

        This is what makes materialization idempotent and resumable: a repeated or
        restarted job reuses these instead of writing duplicates, and can restore the
        online store from them without recomputing. Raises OfflineStoreError on any
        list/read failure — a partial answer would cause silent recomputation.
        """
        prefix = f"{self._prefix}/{tenant_id}/"
        latest: dict = {}
        try:
            paginator = self._s3.get_paginator("list_objects_v2")
            keys = [
                obj["Key"]
                for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix)
                for obj in page.get("Contents", [])
            ]
            for key in sorted(keys):
                meta = self._s3.head_object(Bucket=self._bucket, Key=key).get("Metadata", {})
                if meta.get("source_id") != source_id or meta.get("feature_version") != feature_version:
                    continue
                df = self._read_parquet_from_s3(key)
                df = df[(df["tenant_id"] == tenant_id) & (df["feature_version"] == feature_version)]
                for row in df.to_dict("records"):
                    computed_at = pd.Timestamp(row["computed_at"])
                    if computed_at.tzinfo is not None:
                        computed_at = computed_at.tz_convert("UTC").tz_localize(None)
                    computed_at = computed_at.to_pydatetime()
                    current = latest.get(row["pair_id"])
                    if current is not None and current.computed_at >= computed_at:
                        continue
                    latest[row["pair_id"]] = FeatureVector(
                        entity_id_1=row["entity_id_1"],
                        entity_id_2=row["entity_id_2"],
                        tenant_id=row["tenant_id"],
                        features={n: float(row[f"feat_{n}"]) for n in FEATURE_COLUMN_NAMES},
                        feature_version=row["feature_version"],
                        computed_at=computed_at,
                        computation_ms=float(row.get("computation_ms") or 0.0),
                    )
        except (BotoCoreError, ClientError) as e:
            raise OfflineStoreError(f"cannot scan offline store: {type(e).__name__}") from e
        return latest

    def _read_parquet_from_s3(self, key: str) -> pd.DataFrame:
        """Read a single Parquet file from S3."""
        response = self._s3.get_object(Bucket=self._bucket, Key=key)
        buffer = io.BytesIO(response["Body"].read())
        return pd.read_parquet(buffer, engine="pyarrow")

    def _read_multiple(self, keys: List[str]) -> pd.DataFrame:
        frames = []
        for key in keys:
            try:
                frames.append(self._read_parquet_from_s3(key))
            except Exception as e:
                logger.error(f"Error reading {key}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _fv_to_row(self, fv: FeatureVector) -> dict:
        """Convert FeatureVector to a flat dict for Parquet."""
        row = {
            "entity_id_1": fv.entity_id_1,
            "entity_id_2": fv.entity_id_2,
            "tenant_id": fv.tenant_id,
            "pair_id": f"{sorted([fv.entity_id_1, fv.entity_id_2])[0]}:{sorted([fv.entity_id_1, fv.entity_id_2])[1]}",
            "feature_version": fv.feature_version,
            "computed_at": fv.computed_at,
            "computation_ms": fv.computation_ms,
        }
        # Every canonical feature must be present. A missing name (for example a vector
        # carrying the group-level `ph_err_*` fallback names) is an error, never a
        # silently zero-filled column — zeros would be indistinguishable from real values.
        missing = [n for n in FEATURE_COLUMN_NAMES if n not in fv.features]
        if missing:
            raise ValueError(
                f"FeatureVector {fv.entity_id_1}:{fv.entity_id_2} is missing "
                f"{len(missing)} canonical feature(s), e.g. {missing[:3]}"
            )
        for name in FEATURE_COLUMN_NAMES:
            row[f"feat_{name}"] = float(fv.features[name])
        return row
