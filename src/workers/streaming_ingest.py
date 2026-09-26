"""
feature-store-service/src/workers/streaming_ingest.py

Kafka consumer that processes entity change events in real-time
and invalidates/recomputes affected feature vectors.
Target streaming lag: <5s P99.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from confluent_kafka import Consumer, KafkaError

from src.core.config import settings
from src.core.metrics import STREAMING_LAG, STREAMING_EVENTS
from src.repositories.online_store import OnlineFeatureStore

logger = logging.getLogger(__name__)


class EntityEventConsumer:
    """
    Kafka consumer for real-time entity change events.
    When an entity is updated, invalidates all its cached feature pairs
    so they will be recomputed on next request.
    """

    def __init__(self, online_store: OnlineFeatureStore):
        self._online_store = online_store
        self._consumer: Optional[Consumer] = None
        self._running = False
        self._processed = 0
        self._errors = 0

    def _create_consumer(self) -> Consumer:
        return Consumer({
            "bootstrap.servers": ",".join(settings.KAFKA_BROKERS_LIST),
            "group.id": settings.KAFKA_CONSUMER_GROUP,
            "auto.offset.reset": settings.KAFKA_AUTO_OFFSET_RESET,
            "enable.auto.commit": False,
        })

    async def start(self):
        loop = asyncio.get_event_loop()
        self._consumer = await loop.run_in_executor(None, self._create_consumer)
        self._consumer.subscribe([settings.KAFKA_ENTITY_EVENTS_TOPIC])
        self._running = True
        logger.info(
            f"Started Kafka consumer on topic {settings.KAFKA_ENTITY_EVENTS_TOPIC} "
            f"group {settings.KAFKA_CONSUMER_GROUP}"
        )

    async def stop(self):
        self._running = False
        if self._consumer:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._consumer.close)
        logger.info("Kafka consumer stopped")

    async def run(self):
        """Main consume loop — uses run_in_executor so confluent-kafka poll is non-blocking."""
        if not self._consumer:
            await self.start()

        logger.info("Entity event consumer running...")
        loop = asyncio.get_event_loop()

        try:
            while self._running:
                msg = await loop.run_in_executor(
                    None, lambda: self._consumer.poll(timeout=1.0)
                )

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Kafka consumer error: {msg.error()}")
                    continue

                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    await self._process_message(msg, payload)

                    # Track streaming lag
                    lag_seconds = time.time() - (msg.timestamp()[1] / 1000.0)
                    STREAMING_LAG.set(lag_seconds)

                    # Manual commit after successful processing
                    await loop.run_in_executor(
                        None, lambda: self._consumer.commit(message=msg, asynchronous=False)
                    )

                except Exception as e:
                    self._errors += 1
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    STREAMING_EVENTS.labels(event_type="unknown", status="error").inc()

        finally:
            await self.stop()

    async def _process_message(self, msg, event: dict):
        event_type = event.get("event_type", "unknown")
        entity_id = event.get("entity_id")
        tenant_id = event.get("tenant_id")

        if not entity_id or not tenant_id:
            logger.warning(f"Malformed event (missing entity_id or tenant_id): {event}")
            STREAMING_EVENTS.labels(event_type=event_type, status="skipped").inc()
            return

        logger.debug(f"Processing {event_type} for entity {entity_id} tenant {tenant_id}")

        if event_type in ("entity.updated", "entity.deleted", "entity.merged"):
            invalidated = await self._online_store.invalidate_entity(entity_id, tenant_id)
            logger.info(
                f"Invalidated {invalidated} feature vectors for {event_type} "
                f"entity {entity_id}"
            )
            STREAMING_EVENTS.labels(event_type=event_type, status="processed").inc()

        elif event_type == "entity.created":
            STREAMING_EVENTS.labels(event_type=event_type, status="skipped").inc()

        else:
            logger.debug(f"Unhandled event type: {event_type}")
            STREAMING_EVENTS.labels(event_type=event_type, status="unhandled").inc()

        self._processed += 1

    @property
    def stats(self) -> dict:
        return {
            "processed": self._processed,
            "errors": self._errors,
            "running": self._running,
        }
