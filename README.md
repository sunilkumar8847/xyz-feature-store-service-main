# XYZ MDM 3.0 — Feature Store Service

> **"Features are the language of ML. One Truth. One Computation. Every Time."**

Centralized feature management for the XYZ MDM 3.0 AI plane. Computes and serves a **50-dimensional feature vector** for every entity pair — identically in both model training and real-time inference, guaranteeing zero training-serving skew.

---

## Overview

| Attribute | Value |
|---|---|
| **Service Name** | feature-store |
| **Bounded Context** | AI Intelligence Context (AI Plane) |
| **Language** | Python 3.11 + FastAPI |
| **Criticality** | P2 |
| **Port** | 8006 |
| **Online Store** | Redis (< 10ms P99) |
| **Offline Store** | S3 + Parquet (point-in-time) |
| **Registry** | PostgreSQL |

---

## Service SLOs

| Operation | P50 | P95 | P99 | Timeout |
|---|---|---|---|---|
| Online Get (single pair) | 2ms | 5ms | 10ms | 50ms |
| Online Get (batch 100) | 10ms | 25ms | 50ms | 200ms |
| Feature Push | 5ms | 15ms | 30ms | 100ms |
| Streaming Lag | 500ms | 2s | 5s | Alert @ 30s |
| Materialization Job | — | — | — | 4 hours |

---

## Architecture

The service uses a **Hexagonal (Ports & Adapters)** architecture. The 50-feature computation engine is the single source of truth — the exact same Python code runs during both batch training materialization and real-time serving, making training-serving skew structurally impossible.

```
                    ┌─────────────────────────────────┐
                    │         FEATURE STORE            │
                    ├─────────────────────────────────┤
    REST API  ──▶   │   FeatureComputationService      │
    Kafka     ──▶   │   (50 features, single truth)    │
    Scheduler ──▶   │                                  │
                    ├──────────────┬──────────────────┤
                    │  ONLINE      │  OFFLINE          │
                    │  Redis       │  S3 + Parquet     │
                    │  < 10ms P99  │  Point-in-time    │
                    │  Latest only │  Full history     │
                    └──────────────┴──────────────────┘
                           │                │
                    AI Inference      Model Training
                    (Real-time)         (Batch)
```

---

## 50-Feature Vector

| Category | Count | Features |
|---|---|---|
| **String Similarity** | 15 | Levenshtein, Jaro-Winkler, Damerau, Hamming, OSA, LCS, prefix, postfix |
| **Phonetic** | 5 | Soundex, Metaphone, NYSIIS, Match Rating, full-name Soundex |
| **Token-based** | 8 | Jaccard, token sort ratio, token set ratio, partial ratio |
| **Semantic** | 10 | Cosine similarity, Euclidean distance (sentence-transformers embeddings) |
| **Structural** | 7 | Field presence ratio, length ratio, null count diff, schema similarity |
| **Domain-specific** | 5 | Email domain match, phone prefix/full match, geo similarity |

---

## Project Structure

```
xyz-feature-store-service/
├── pyproject.toml
├── alembic.ini
├── pytest.ini
├── Dockerfile
├── .env.example
├── migrations/
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py
└── src/
    ├── main.py                         # FastAPI app + lifespan
    ├── core/
    │   ├── config.py                   # Pydantic Settings
    │   ├── dependencies.py             # DI container
    │   └── metrics.py                  # Prometheus metrics
    ├── domain/
    │   └── models.py                   # FeatureVector, EntitySnapshot,
    │                                   # MaterializationJob, DriftReport
    ├── services/
    │   ├── feature_computation.py      # 50-feature engine (single truth)
    │   └── feature_store.py            # Orchestration service
    ├── repositories/
    │   ├── online_store.py             # Redis + MessagePack
    │   ├── offline_store.py            # S3 + Parquet + point-in-time
    │   └── feature_registry.py         # PostgreSQL ORM + repository
    ├── api/v1/
    │   ├── schemas.py                  # Pydantic v2 DTOs
    │   └── endpoints/
    │       └── features.py             # All REST endpoints
    └── workers/
        ├── streaming_ingest.py         # Kafka consumer
        └── materialization.py          # Batch worker + drift detection
```

---

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for shared infrastructure)
- Shared infra running: Redis, Kafka, PostgreSQL (feature-store-postgres), LocalStack (S3)

---

## Local Setup

### 1. Start shared infrastructure

```bash
cd infra
docker compose up -d
```

Wait for all services to be healthy:

```bash
docker compose ps
# feature-store-postgres → healthy
# localstack            → healthy
# redis                 → healthy
# kafka                 → healthy
```

### 2. Install dependencies

```bash
cd xyz-feature-store-service
pip install -e ".[dev]"
```

> **Note:** If you get a `setuptools.backends` error, ensure `pyproject.toml` has `build-backend = "setuptools.build_meta"` or install without editable mode using the individual packages listed in pyproject.toml.

### 3. Configure environment

```bash
cp .env.example .env
```

Key variables to verify:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_DB=feature_store
POSTGRES_USER=feature_user
POSTGRES_PASSWORD=feature_pass

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=2

KAFKA_BROKERS=localhost:9092

S3_BUCKET=xyz-mdm-feature-store
S3_ENDPOINT_URL=https://localhost:4566
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
```

### 4. Run database migrations

```bash
alembic upgrade head
```

Verify tables were created:

```bash
docker exec -it xyz-feature-store-postgres psql -U feature_user -d feature_store -c "\dt"
# Should show: feature_definitions, materialization_jobs, drift_reports
```

### 5. Start the service

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8006 --reload
```

Expected startup output:
```
Connected to Redis at localhost:6379
Redis online store: CONNECTED
FeatureComputationService initialized with all 50 feature computers
Feature computation service: INITIALIZED
Database schema: READY
Feature definitions: SEEDED
Kafka consumer: STARTED
Materialization scheduler: STARTED
feature-store startup complete. Docs at /docs
```

---

## API Reference

**Swagger UI:** http://localhost:8006/docs  
**Metrics:** http://localhost:8006/metrics

> **Windows users:** Use Swagger UI or `curl.exe` (not PowerShell's `curl` alias) for API testing.

### Get features for a single entity pair

```
GET /api/v1/features/{entity_id_1}/{entity_id_2}
```

Query parameters:

| Parameter | Required | Description |
|---|---|---|
| `tenant_id` | ✅ | Tenant identifier |
| `name1` / `name2` | Optional | Entity names (for on-the-fly compute) |
| `email1` / `email2` | Optional | Entity emails |
| `phone1` / `phone2` | Optional | Entity phone numbers |

Example (Windows):
```powershell
curl.exe -s "http://localhost:8006/api/v1/features/entity-001/entity-002?tenant_id=tenant-123&name1=John+Smith&name2=Jon+Smith&email1=john@example.com&email2=jon@example.com&phone1=%2B14155551234&phone2=%2B14155551234"
```

### Batch feature retrieval (up to 100 pairs)

```
POST /api/v1/features/batch
```

```json
{
  "tenant_id": "tenant-123",
  "pairs": [
    {"entity_id_1": "entity-001", "entity_id_2": "entity-002"},
    {"entity_id_1": "entity-003", "entity_id_2": "entity-004"}
  ]
}
```

### Push / invalidate features for an entity

```
PUT /api/v1/features/{entity_id}
```

```json
{
  "tenant_id": "tenant-123"
}
```

### Get feature definitions catalog

```
GET /api/v1/features/definitions
```

Returns all 50 feature definitions from the registry.

### Trigger materialization job

```
POST /api/v1/materialize
```

```json
{
  "tenant_id": "tenant-123",
  "full_refresh": false
}
```

### Get materialization job status

```
GET /api/v1/materialize/{job_id}
```

### Get drift reports

```
GET /api/v1/drift?tenant_id=tenant-123
```

### Health check

```
GET /api/v1/health
```

```json
{
  "status": "healthy",
  "checks": {
    "postgres": true,
    "redis": true,
    "s3": true,
    "kafka": true
  }
}
```

---

## Observability

### Prometheus Metrics

| Metric | Type | Labels | Alert |
|---|---|---|---|
| `feature_get_latency_ms` | Histogram | `store_type` | P99 > 15ms |
| `feature_get_total` | Counter | `status`, `store` | — |
| `feature_cache_hit_rate` | Gauge | — | < 90% |
| `materialization_duration_seconds` | Histogram | `job_name` | > 6 hours |
| `streaming_lag_seconds` | Gauge | — | > 30s |
| `feature_drift_score` | Gauge | `feature_name` | > 0.1 |

All metrics available at: `GET /metrics`

---

## Infrastructure Dependencies

| Service | Purpose | Host Port | Container |
|---|---|---|---|
| `feature-store-postgres` | Feature registry (metadata, jobs, drift) | 5434 | `xyz-feature-store-postgres` |
| `redis` | Online store (shared) | 6379 | `xyz-redis` |
| `kafka` | Streaming entity events (shared) | 9092 | `xyz-kafka` |
| `localstack` | S3 offline store emulation | 4566 | `xyz-localstack` |

Redis DB allocation: `DB 2` (isolated from other services using the shared Redis instance).

S3 bucket: `xyz-mdm-feature-store`  
S3 path pattern: `features/{tenant_id}/year={Y}/month={M}/day={D}/features.parquet`

---

## Key Design Decisions

**Zero training-serving skew** — The `FeatureComputationService` class is the single source of truth for all 50 feature computations. The exact same code path runs during real-time serving (via the online store) and batch materialization (via the offline store).

**MessagePack serialization** — Redis values are serialized with MessagePack (binary) rather than JSON, giving ~40% smaller payload and faster serialization for the 50-float feature vector.

**Order-independent Redis keys** — Keys are `feature:{tenant_id}:{sha256(sorted_ids)[:16]}` ensuring `(A, B)` and `(B, A)` always resolve to the same cache entry.

**Redis pipeline for batch** — Batch GET/SET uses a single Redis pipeline (one round-trip) regardless of batch size, hitting the <50ms P99 target for 100-pair batches.

**Drift detection** — KL divergence (threshold 0.1) and Jensen-Shannon divergence (threshold 0.05) computed on 50-bin histograms, checked hourly via APScheduler.

---

## Running Tests

```bash
pytest tests/unit/ -v
```

---

## Downstream Consumers

| Service | How it uses Feature Store |
|---|---|
| **model-inference-service** | `GET /api/v1/features/{e1}/{e2}` — real-time serving |
| **matching-service** | Feature vectors for ensemble scoring |
| **model-training-pipeline** | `POST /api/v1/features/offline` — point-in-time training data |