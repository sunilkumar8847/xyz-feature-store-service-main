"""
feature-store-service/src/core/dependencies.py

Singleton application-level dependencies.
Initialized once at startup, shared across all requests.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.repositories.online_store import OnlineFeatureStore
from src.services.feature_computation import FeatureComputationService

logger = logging.getLogger(__name__)

# Singletons set during application startup
online_store_instance: Optional[OnlineFeatureStore] = None
computation_service_instance: Optional[FeatureComputationService] = None
