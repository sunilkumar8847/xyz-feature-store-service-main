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

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import BotoCoreError, ClientError

from src.core.config import settings
from src.domain.models import FeatureVector, OfflineFeatureRequest

logger = logging.getLogger(__name__)


def _to_naive_utc(dt: datetime) -> datetime:
    """Normalize to naive UTC — this store's convention (utcnow(), stripped S3 LastModified).
    Comparing a tz-aware client timestamp with naive values raises TypeError."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _get_feature_column_names() -> List[str]:
    """Return sorted list of all 50 feature names (without 'feat_' prefix)."""
    return sorted([
        # String Similarity (15)
        "ss_levenshtein_name", "ss_jaro_winkler_name", "ss_damerau_levenshtein_name",
        "ss_hamming_name", "ss_jaro_name", "ss_levenshtein_address",
        "ss_jaro_winkler_address", "ss_damerau_address", "ss_levenshtein_email",
        "ss_jaro_winkler_email", "ss_name_addr_cross", "ss_longest_common_subseq",
        "ss_common_prefix_name", "ss_osa_distance_name", "ss_postfix_similarity",
        # Phonetic (5)
        "ph_soundex_name", "ph_metaphone_name", "ph_nysiis_name",
        "ph_match_rating_name", "ph_soundex_full_name",
        # Token-based (8)
        "tk_jaccard_name", "tk_jaccard_address", "tk_token_sort_ratio_name",
        "tk_token_set_ratio_name", "tk_partial_ratio_name", "tk_token_sort_address",
        "tk_token_set_address", "tk_common_token_count",
        # Semantic (10)
        "sem_cosine_name", "sem_euclidean_name", "sem_cosine_address",
        "sem_euclidean_address", "sem_cosine_full", "sem_euclidean_full",
        "sem_cross_name_addr", "sem_angular_name", "sem_dot_product_name",
        "sem_cosine_name_addr_concat",
        # Structural (7)
        "str_field_presence_ratio", "str_length_ratio_name", "str_null_count_diff",
        "str_field_overlap", "str_schema_similarity", "str_asymmetric_null_ratio",
        "str_word_count_ratio_name",
        # Domain-specific (5)
        "dom_email_domain_match", "dom_phone_prefix_match", "dom_phone_full_match",
        "dom_geo_similarity", "dom_email_local_similarity",
    ])

# Canonical feature ordering — the single source of truth shared by the Parquet
# schema below and the /features/offline response. Matches FeatureVector.as_list,
# which orders by sorted(features.keys()), so offline and online vectors align.
FEATURE_COLUMN_NAMES: List[str] = _get_feature_column_names()


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
    ) -> str:
        """
        Write a batch of feature vectors to S3 as Parquet.
        Returns the S3 path written.
        """
        if not feature_vectors:
            return ""

        dt = partition_dt or datetime.utcnow()
        s3_key = self._make_s3_key(feature_vectors[0].tenant_id, dt)

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

        # Build pair lookup set
        pair_set = {
            (e1, e2) for e1, e2 in request.entity_pairs
        }

        # Read and filter Parquet files
        frames = []
        for key in keys:
            try:
                df = self._read_parquet_from_s3(key)
                # Filter to requested pairs
                mask = df.apply(
                    lambda row: (row["entity_id_1"], row["entity_id_2"]) in pair_set
                    or (row["entity_id_2"], row["entity_id_1"]) in pair_set,
                    axis=1,
                )
                filtered = df[mask]
                if not filtered.empty:
                    frames.append(filtered)
            except Exception as e:
                logger.error(f"Error reading {key}: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Point-in-time: for each pair, keep the LATEST feature vector
        # that was computed BEFORE as_of_timestamp
        # computed_at may be read back tz-aware (schema declares tz="UTC") or naive;
        # compare both sides as UTC-aware so either form works.
        computed_at_utc = pd.to_datetime(combined["computed_at"], utc=True)
        combined = combined[computed_at_utc <= pd.Timestamp(as_of, tz="UTC")]
        combined = combined.sort_values("computed_at", ascending=False)
        combined = combined.drop_duplicates(subset=["entity_id_1", "entity_id_2"], keep="first")

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

    def _make_s3_key(self, tenant_id: str, dt: datetime) -> str:
        return (
            f"{self._prefix}/{tenant_id}/"
            f"year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/"
            f"features_{dt.strftime('%H%M%S')}.parquet"
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
            features/{tenant}/year=YYYY/month=MM/day=DD/features_HHMMSS.parquet
        Returns None if the key does not follow that layout.
        """
        match = re.search(
            r"year=(\d{4})/month=(\d{2})/day=(\d{2})/features_(\d{2})(\d{2})(\d{2})\.parquet$",
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
            logger.error(f"S3 LIST error: {e}")

        return sorted(keys)

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
        # Add feature columns
        for name in _get_feature_column_names():
            row[f"feat_{name}"] = float(fv.features.get(name, 0.0))
        return row
