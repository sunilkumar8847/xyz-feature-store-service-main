"""
feature-store-service/src/core/config.py

Pydantic Settings: All configuration from environment variables.
Production-grade with validation and documentation.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),   # .env.local overrides .env
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Service Identity ─────────────────────────────────────────────
    SERVICE_NAME: str = "feature-store"
    SERVICE_VERSION: str = "3.0.0"
    ENVIRONMENT: Environment = Environment.TEST
    LOG_LEVEL: LogLevel = LogLevel.INFO


    # ─── API Server ───────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8115
    API_WORKERS: int = 4
    API_RELOAD: bool = False
    API_PREFIX: str = "/api"
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"

    # ─── Security ─────────────────────────────────────────────────────
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    API_KEY_HEADER: str = "X-API-Key"
    INTERNAL_API_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_SECONDS: int = 3600

    # ─── PostgreSQL (Feature Registry) ────────────────────────────────
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "xyzmdm"
    POSTGRES_USER: str = "xyzmdm"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10
    POSTGRES_POOL_TIMEOUT: int = 30
    POSTGRES_ECHO_SQL: bool = False

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def SYNC_DATABASE_URL(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ─── Redis (Online Store) ─────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_MAX_CONNECTIONS: int = 100
    REDIS_SOCKET_TIMEOUT: float = 0.5
    REDIS_CONNECT_TIMEOUT: float = 2.0

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ─── S3 (Offline Store) ───────────────────────────────────────────
    S3_BUCKET: str = "xyz-mdm-feature-store"
    S3_REGION: str = "us-east-1"
    S3_PREFIX: str = "features"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None  # For LocalStack

    # ─── Kafka (Streaming Ingest) ─────────────────────────────────────
    KAFKA_BROKERS: str = "localhost:9092"
    KAFKA_ENTITY_EVENTS_TOPIC: str = "mdm.entity.events"
    KAFKA_FEATURE_UPDATES_TOPIC: str = "mdm.feature.updates"
    KAFKA_CONSUMER_GROUP: str = "feature-store-consumer"
    KAFKA_AUTO_OFFSET_RESET: str = "latest"
    KAFKA_MAX_POLL_RECORDS: int = 500
    KAFKA_SESSION_TIMEOUT_MS: int = 30000
    KAFKA_HEARTBEAT_INTERVAL_MS: int = 3000
    KAFKA_ENABLE_AUTO_COMMIT: bool = False

    @computed_field
    @property
    def KAFKA_BROKERS_LIST(self) -> List[str]:
        return self.KAFKA_BROKERS.split(",")

    # ─── Feature Store Configuration ─────────────────────────────────
    FEATURE_VERSION: str = "v2.0.0"
    FEATURE_TTL_HOURS: int = 24
    FEATURE_BATCH_SIZE: int = 100
    FEATURE_VECTOR_DIM: int = 50
    MATERIALIZATION_CRON: str = "0 2 * * *"  # Daily at 2 AM UTC
    MATERIALIZATION_LOOKBACK_DAYS: int = 30
    CACHE_HIT_WARN_THRESHOLD: float = 0.9  # Alert if < 90% cache hit

    # ─── Embedding Model ─────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CACHE_SIZE: int = 10000  # LRU cache entries

    # ─── Drift Detection ─────────────────────────────────────────────
    DRIFT_CHECK_INTERVAL_SECONDS: int = 3600
    DRIFT_KL_THRESHOLD: float = 0.1
    DRIFT_JS_THRESHOLD: float = 0.05
    DRIFT_LOOKBACK_DAYS: int = 7

    # ─── Observability ────────────────────────────────────────────────
    PROMETHEUS_PORT: int = 9090
    OTLP_ENDPOINT: Optional[str] = None
    TRACE_SAMPLE_RATE: float = 0.1

    # ─── External Services ────────────────────────────────────────────
    MODEL_INFERENCE_SERVICE_URL: str = "http://localhost:8090"
    LINEAGE_SERVICE_URL: str = "http://localhost:8095"

    @field_validator("API_WORKERS")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        if v < 1 or v > 32:
            raise ValueError("API_WORKERS must be between 1 and 32")
        return v


# Singleton instance
settings = Settings()
