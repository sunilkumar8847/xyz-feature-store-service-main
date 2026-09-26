"""
feature-store-service/src/core/metrics.py

Prometheus metrics for Feature Store observability.
Matches the LLD Section 7.1 metric specifications exactly.
"""
from prometheus_client import Counter, Gauge, Histogram

# ─── Feature Retrieval ────────────────────────────────────────────────────────

FEATURE_GET_LATENCY = Histogram(
    "feature_get_latency_ms",
    "Feature retrieval latency in milliseconds",
    ["store_type"],  # cache | compute | offline
    buckets=[1, 2, 5, 10, 15, 25, 50, 100, 200, 500, 1000],
)

FEATURE_GET_COUNTER = Counter(
    "feature_get_total",
    "Total feature GET requests",
    ["status", "store"],  # status: hit|miss, store: online|offline
)

FEATURE_CACHE_HIT_RATE = Gauge(
    "feature_cache_hit_rate",
    "Online store cache hit rate (rolling)",
)

FEATURE_PUSH_LATENCY = Histogram(
    "feature_push_latency_ms",
    "Feature push latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100],
)

# ─── Materialization ──────────────────────────────────────────────────────────

MATERIALIZATION_DURATION = Histogram(
    "materialization_duration_seconds",
    "Materialization job duration in seconds",
    ["job_name"],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
)

MATERIALIZATION_RECORDS = Counter(
    "materialization_records_total",
    "Total records processed in materialization",
    ["status"],  # success | failed
)

# ─── Streaming ────────────────────────────────────────────────────────────────

STREAMING_LAG = Gauge(
    "streaming_lag_seconds",
    "Kafka consumer lag in seconds",
)

STREAMING_EVENTS = Counter(
    "streaming_events_total",
    "Total streaming events processed",
    ["event_type", "status"],
)

# ─── Drift Detection ──────────────────────────────────────────────────────────

FEATURE_DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Feature drift KL divergence score",
    ["feature_name"],
)

DRIFT_ALERTS = Counter(
    "feature_drift_alerts_total",
    "Total drift alerts triggered",
    ["drift_type", "feature_name"],
)

# ─── Service Health ───────────────────────────────────────────────────────────

ACTIVE_MATERIALIZATION_JOBS = Gauge(
    "active_materialization_jobs",
    "Number of currently running materialization jobs",
)

REDIS_CONNECTION_ERRORS = Counter(
    "redis_connection_errors_total",
    "Total Redis connection errors",
)

S3_OPERATION_ERRORS = Counter(
    "s3_operation_errors_total",
    "Total S3 operation errors",
    ["operation"],
)
