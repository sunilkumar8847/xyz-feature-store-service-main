"""
feature-store-service/src/main.py

FastAPI application factory with:
- Lifespan management (startup/shutdown)
- Middleware (CORS, logging, tracing)
- Router registration
- Prometheus metrics endpoint
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from xyz_security import get_current_tenant


def _setup_otel(service_name: str, endpoint: str) -> None:
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as e:
        logging.getLogger(__name__).warning(f"OTel setup failed (non-fatal): {e}")


def _instrument_fastapi(app: "FastAPI") -> None:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except Exception as e:
        logging.getLogger(__name__).warning(f"OTel FastAPI instrumentation failed: {e}")

from src.api.v1.endpoints.features import public_router, router as features_router
from src.core.config import settings
from src.core.dependencies import (
    computation_service_instance,
    online_store_instance,
)
from src.repositories.feature_registry import Base, get_engine
from src.repositories.online_store import OnlineFeatureStore
from src.services.feature_computation import FeatureComputationService
from src.workers.streaming_ingest import EntityEventConsumer

# ─── Structured Logging Setup ─────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer()
        if settings.ENVIRONMENT.value == "development"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.LOG_LEVEL.value)
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logging.basicConfig(
    level=logging.getLevelName(settings.LOG_LEVEL.value),
    stream=sys.stdout,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


# ─── Application Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Manage startup and shutdown of all connections and workers.
    All heavy initialization happens here — not at module import time.
    """
    import src.core.dependencies as deps

    logger.info(
        f"Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION} "
        f"[{settings.ENVIRONMENT.value}]"
    )

    # 1. Initialize Online Store (Redis)
    try:
        deps.online_store_instance = await OnlineFeatureStore.create()
        logger.info("Redis online store: CONNECTED")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        # Don't crash — allow degraded operation
        deps.online_store_instance = None

    # 2. Initialize Feature Computation Service
    deps.computation_service_instance = FeatureComputationService()
    logger.info("Feature computation service: INITIALIZED")

    # 2b. Load the embedding model now, so a missing model is visible at startup (and
    #     in /health) instead of silently zeroing the semantic features later.
    from src.services.feature_computation import embedding_model_ready
    if await asyncio.to_thread(embedding_model_ready):
        logger.info("Embedding model: LOADED")
    else:
        logger.error(
            "Embedding model: NOT AVAILABLE — feature computation and materialization "
            "will fail until it can be loaded"
        )

    # 3. Create database tables (idempotent)
    try:
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema: READY")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # 4. Seed feature definitions (if not present)
    try:
        await _seed_feature_definitions()
        logger.info("Feature definitions: SEEDED")
    except Exception as e:
        logger.warning(f"Feature definition seeding failed (non-critical): {e}")

    # 4b. Close jobs a previous process left RUNNING (see recover_interrupted_jobs).
    try:
        from src.repositories.feature_registry import (
            FeatureRegistryRepository, get_session_factory,
        )
        from src.workers.materialization import default_job_lock, recover_interrupted_jobs
        async with get_session_factory()() as session:
            closed = await recover_interrupted_jobs(
                FeatureRegistryRepository(session), default_job_lock())
        if closed:
            logger.warning(f"Marked {len(closed)} interrupted materialization job(s) FAILED")
    except Exception as e:
        logger.error(f"Interrupted-job recovery failed: {e}")

    # 5. Start Kafka consumer (background task)
    kafka_task = None
    if deps.online_store_instance:
        consumer = EntityEventConsumer(online_store=deps.online_store_instance)
        kafka_task = asyncio.create_task(_run_kafka_consumer(consumer))
        logger.info("Kafka consumer: STARTED")

    # 6. Start scheduled materialization
    scheduler_task = asyncio.create_task(_run_scheduler())
    logger.info("Materialization scheduler: STARTED")

    logger.info(f"{settings.SERVICE_NAME} startup complete. Docs at /docs")

    yield  # Application runs here

    # ─── Shutdown ────────────────────────────────────────────────────────────

    logger.info("Shutting down feature-store-service...")

    if kafka_task:
        kafka_task.cancel()
        try:
            await kafka_task
        except asyncio.CancelledError:
            pass

    scheduler_task.cancel()

    if deps.online_store_instance:
        await deps.online_store_instance.close()

    await get_engine().dispose()
    logger.info("Shutdown complete")


async def _run_kafka_consumer(consumer: EntityEventConsumer):
    """Run Kafka consumer with restart-on-failure."""
    while True:
        try:
            await consumer.run()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Kafka consumer crashed: {e}. Restarting in 10s...")
            await asyncio.sleep(10)


async def _run_scheduler():
    """APScheduler-based cron for daily materialization."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = AsyncIOScheduler()

        # Daily materialization at 2 AM UTC
        scheduler.add_job(
            _trigger_scheduled_materialization,
            CronTrigger.from_crontab(settings.MATERIALIZATION_CRON),
            id="daily_materialization",
            max_instances=1,
            coalesce=True,
        )

        # Hourly drift check
        scheduler.add_job(
            _trigger_drift_check,
            "interval",
            seconds=settings.DRIFT_CHECK_INTERVAL_SECONDS,
            id="drift_check",
            max_instances=1,
        )

        scheduler.start()
        logger.info(f"Scheduler started. Materialization cron: {settings.MATERIALIZATION_CRON}")

        # Keep running until cancelled
        while True:
            await asyncio.sleep(3600)

    except asyncio.CancelledError:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


async def _trigger_scheduled_materialization():
    """Called by scheduler — materializes every tenant the configured source knows."""
    import src.core.dependencies as deps
    from src.core.config import settings
    from src.domain.models import MaterializationJob
    from src.workers.materialization import run_materialization_job

    logger.info("Scheduled materialization triggered")

    if not deps.online_store_instance:
        logger.warning("Online store not available — skipping scheduled materialization")
        return
    if settings.MATERIALIZATION_SOURCE is None:
        # No source exists in production yet (the MDM entity source is not
        # implemented). Skip rather than record a failed job every night.
        logger.warning(
            "No MATERIALIZATION_SOURCE configured — skipping scheduled materialization"
        )
        return

    job = MaterializationJob(triggered_by="scheduler")
    await run_materialization_job(
        job, deps.online_store_instance, deps.computation_service_instance
    )


async def _trigger_drift_check():
    """Called by scheduler — runs drift detection."""
    logger.info("Scheduled drift check triggered (placeholder)")


async def _seed_feature_definitions():
    """Seed the 50 feature definitions into PostgreSQL registry."""
    from src.core.config import settings
    from src.domain.models import (
        FeatureCategory, FeatureDefinition, FeatureStatus
    )
    from src.repositories.feature_registry import get_session_factory, FeatureRegistryRepository

    definitions = [
        # String Similarity (15)
        FeatureDefinition("ss_levenshtein_name", FeatureCategory.STRING_SIMILARITY, "Levenshtein edit distance between names (normalized)", "float"),
        FeatureDefinition("ss_jaro_winkler_name", FeatureCategory.STRING_SIMILARITY, "Jaro-Winkler similarity for names (prefix-weighted)", "float"),
        FeatureDefinition("ss_damerau_levenshtein_name", FeatureCategory.STRING_SIMILARITY, "Damerau-Levenshtein (allows transpositions)", "float"),
        FeatureDefinition("ss_hamming_name", FeatureCategory.STRING_SIMILARITY, "Hamming distance for same-length strings", "float"),
        FeatureDefinition("ss_jaro_name", FeatureCategory.STRING_SIMILARITY, "Jaro similarity for names", "float"),
        FeatureDefinition("ss_levenshtein_address", FeatureCategory.STRING_SIMILARITY, "Levenshtein distance for addresses", "float"),
        FeatureDefinition("ss_jaro_winkler_address", FeatureCategory.STRING_SIMILARITY, "Jaro-Winkler similarity for addresses", "float"),
        FeatureDefinition("ss_damerau_address", FeatureCategory.STRING_SIMILARITY, "Damerau-Levenshtein for addresses", "float"),
        FeatureDefinition("ss_levenshtein_email", FeatureCategory.STRING_SIMILARITY, "Levenshtein distance for email strings", "float"),
        FeatureDefinition("ss_jaro_winkler_email", FeatureCategory.STRING_SIMILARITY, "Jaro-Winkler for email strings", "float"),
        FeatureDefinition("ss_name_addr_cross", FeatureCategory.STRING_SIMILARITY, "Cross-field: name prefix vs address prefix", "float"),
        FeatureDefinition("ss_longest_common_subseq", FeatureCategory.STRING_SIMILARITY, "Longest common subsequence (normalized)", "float"),
        FeatureDefinition("ss_common_prefix_name", FeatureCategory.STRING_SIMILARITY, "Common prefix length between names", "float"),
        FeatureDefinition("ss_osa_distance_name", FeatureCategory.STRING_SIMILARITY, "Optimal String Alignment distance for names", "float"),
        FeatureDefinition("ss_postfix_similarity", FeatureCategory.STRING_SIMILARITY, "Last 10 characters similarity (suffixes)", "float"),
        # Phonetic (5)
        FeatureDefinition("ph_soundex_name", FeatureCategory.PHONETIC, "Soundex code match for first name token", "boolean"),
        FeatureDefinition("ph_metaphone_name", FeatureCategory.PHONETIC, "Metaphone code match for first name token", "boolean"),
        FeatureDefinition("ph_nysiis_name", FeatureCategory.PHONETIC, "NYSIIS code match for first name token", "boolean"),
        FeatureDefinition("ph_match_rating_name", FeatureCategory.PHONETIC, "Match Rating Codex comparison for names", "boolean"),
        FeatureDefinition("ph_soundex_full_name", FeatureCategory.PHONETIC, "Soundex on full name string", "boolean"),
        # Token-based (8)
        FeatureDefinition("tk_jaccard_name", FeatureCategory.TOKEN_BASED, "Jaccard index on name tokens", "float"),
        FeatureDefinition("tk_jaccard_address", FeatureCategory.TOKEN_BASED, "Jaccard index on address tokens", "float"),
        FeatureDefinition("tk_token_sort_ratio_name", FeatureCategory.TOKEN_BASED, "RapidFuzz token sort ratio for names", "float"),
        FeatureDefinition("tk_token_set_ratio_name", FeatureCategory.TOKEN_BASED, "RapidFuzz token set ratio for names", "float"),
        FeatureDefinition("tk_partial_ratio_name", FeatureCategory.TOKEN_BASED, "RapidFuzz partial ratio for names", "float"),
        FeatureDefinition("tk_token_sort_address", FeatureCategory.TOKEN_BASED, "RapidFuzz token sort ratio for addresses", "float"),
        FeatureDefinition("tk_token_set_address", FeatureCategory.TOKEN_BASED, "RapidFuzz token set ratio for addresses", "float"),
        FeatureDefinition("tk_common_token_count", FeatureCategory.TOKEN_BASED, "Fraction of tokens shared between names", "float"),
        # Semantic (10)
        FeatureDefinition("sem_cosine_name", FeatureCategory.SEMANTIC, "Cosine similarity of name embeddings", "float"),
        FeatureDefinition("sem_euclidean_name", FeatureCategory.SEMANTIC, "Euclidean similarity of name embeddings (normalized)", "float"),
        FeatureDefinition("sem_cosine_address", FeatureCategory.SEMANTIC, "Cosine similarity of address embeddings", "float"),
        FeatureDefinition("sem_euclidean_address", FeatureCategory.SEMANTIC, "Euclidean similarity of address embeddings", "float"),
        FeatureDefinition("sem_cosine_full", FeatureCategory.SEMANTIC, "Cosine similarity of full entity embeddings", "float"),
        FeatureDefinition("sem_euclidean_full", FeatureCategory.SEMANTIC, "Euclidean similarity of full entity embeddings", "float"),
        FeatureDefinition("sem_cross_name_addr", FeatureCategory.SEMANTIC, "Cross-field cosine: name1 vs address2", "float"),
        FeatureDefinition("sem_angular_name", FeatureCategory.SEMANTIC, "Angular distance between name embeddings", "float"),
        FeatureDefinition("sem_dot_product_name", FeatureCategory.SEMANTIC, "Dot product of normalized name embeddings", "float"),
        FeatureDefinition("sem_cosine_name_addr_concat", FeatureCategory.SEMANTIC, "Cosine similarity of concatenated name+address embeddings", "float"),
        # Structural (7)
        FeatureDefinition("str_field_presence_ratio", FeatureCategory.STRUCTURAL, "Fraction of fields populated in both entities", "float"),
        FeatureDefinition("str_length_ratio_name", FeatureCategory.STRUCTURAL, "Min/max ratio of name lengths", "float"),
        FeatureDefinition("str_null_count_diff", FeatureCategory.STRUCTURAL, "Normalized difference in null field counts", "float"),
        FeatureDefinition("str_field_overlap", FeatureCategory.STRUCTURAL, "Fraction of fields present in both entities", "float"),
        FeatureDefinition("str_schema_similarity", FeatureCategory.STRUCTURAL, "Jaccard similarity of field key sets", "float"),
        FeatureDefinition("str_asymmetric_null_ratio", FeatureCategory.STRUCTURAL, "Fraction of fields not asymmetrically null", "float"),
        FeatureDefinition("str_word_count_ratio_name", FeatureCategory.STRUCTURAL, "Min/max ratio of word counts in names", "float"),
        # Domain-specific (5)
        FeatureDefinition("dom_email_domain_match", FeatureCategory.DOMAIN_SPECIFIC, "Boolean: email domains are identical", "boolean"),
        FeatureDefinition("dom_phone_prefix_match", FeatureCategory.DOMAIN_SPECIFIC, "Boolean: phone area codes match (first 3 digits)", "boolean"),
        FeatureDefinition("dom_phone_full_match", FeatureCategory.DOMAIN_SPECIFIC, "Boolean: full 10-digit phone numbers match", "boolean"),
        FeatureDefinition("dom_geo_similarity", FeatureCategory.DOMAIN_SPECIFIC, "Geographic proximity score (1=same, 0=1000km+)", "float"),
        FeatureDefinition("dom_email_local_similarity", FeatureCategory.DOMAIN_SPECIFIC, "Jaro-Winkler similarity of email local parts", "float"),
    ]

    async with get_session_factory()() as session:
        registry = FeatureRegistryRepository(session)
        for defn in definitions:
            await registry.upsert_feature(defn)
        await session.commit()


# ─── FastAPI Application Factory ─────────────────────────────────────────────

def create_app() -> FastAPI:
    _setup_otel(
        service_name=settings.SERVICE_NAME,
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317"),
    )

    is_prod = settings.ENVIRONMENT.value in ("production", "staging", "prod")

    app = FastAPI(
        title="XYZ MDM — Feature Store Service",
        description=(
            "Centralized ML feature management: real-time feature retrieval (<10ms P99), "
            "training-serving skew prevention, 50 matching features, Feast-compatible API."
        ),
        version="1.0.0",
        contact={"name": "XYZ MDM Platform", "email": "engineering@xyzmdm.com"},
        servers=[{"url": "http://localhost:8034", "description": "Integration"}],
        docs_url=None if is_prod else "/swagger-ui.html",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
        lifespan=lifespan,
    )

    _instrument_fastapi(app)

    # ─── Middleware ──────────────────────────────────────────────────────────
    _cors_origins = [o.strip() for o in os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Verified-Tenant-ID",
                       "X-Verified-User-ID", "X-Verified-Roles", "X-Verified-Platform",
                       "X-Request-Timestamp", "X-Gateway-Signature"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        import time
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} [{elapsed_ms:.1f}ms]"
        )
        return response

    # ─── Routes ─────────────────────────────────────────────────────────────
    app.include_router(public_router, prefix=settings.API_PREFIX)
    app.include_router(features_router, prefix=settings.API_PREFIX, dependencies=[Depends(get_current_tenant)])

    # ─── Prometheus Metrics ──────────────────────────────────────────────────
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ─── Root ────────────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.SERVICE_NAME,
            "version": settings.SERVICE_VERSION,
            "docs": settings.DOCS_URL,
            "health": f"{settings.API_PREFIX}/v1/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.value.lower(),
    )
