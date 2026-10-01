"""
feature-store-service/src/adapters/materialization_sources.py

Where MaterializationWorker gets the entity pairs it computes features for.

    MaterializationSource            (abstract — the worker depends only on this)
        ├── SyntheticMaterializationSource   development/test: synthetic_data output
        └── (future) MDM entity source        NOT IMPLEMENTED — no MDM API contract yet

A source yields PairBatch objects. Every batch is SINGLE-TENANT: the worker writes
each batch to the offline store as one tenant partition, so a mixed batch would file
one tenant's features under another tenant's prefix.

Adding the production source means implementing `iter_batches()` against the real MDM
entity service and registering it in `build_materialization_source()`. The worker, the
feature computation and both stores stay unchanged.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from src.core.config import Settings, check_materialization_source
from src.domain.models import EntitySnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkippedPair:
    """A requested pair the source could not turn into two entity snapshots."""

    entity_id_1: str
    entity_id_2: str
    tenant_id: str
    reason: str


@dataclass
class PairBatch:
    """Up to batch_size resolvable pairs for ONE tenant, plus what was skipped."""

    tenant_id: str
    pairs: List[Tuple[EntitySnapshot, EntitySnapshot]] = field(default_factory=list)
    skipped: List[SkippedPair] = field(default_factory=list)


class MaterializationSourceError(ValueError):
    """The source's input is missing, malformed or has the wrong provenance."""


class MaterializationSource(ABC):
    """Supplies entity pairs, as EntitySnapshot pairs, to MaterializationWorker."""

    #: Recorded as object metadata on every offline file this source produces, so the
    #: provenance of stored features is never ambiguous.
    name: str = "unknown"

    @abstractmethod
    def iter_batches(
        self,
        tenant_id: Optional[str],
        lookback_days: int,
        batch_size: int,
    ) -> Iterator[PairBatch]:
        """
        Yield single-tenant batches.

        tenant_id=None means every tenant the source knows about. lookback_days bounds
        how far back to look for changed entities; a source for which that has no
        meaning must document that it ignores it.
        """


# ─── Synthetic development source ─────────────────────────────────────────────

class SyntheticMaterializationSource(MaterializationSource):
    """
    Reads a dataset written by `python -m synthetic_data`:

      manifest.json     provenance; must declare source=synthetic, is_production_data=false
      entities.parquet  one row per entity record; fields stored as JSON so an OMITTED
                        key stays distinct from a null value (str_schema_similarity
                        depends on that difference)
      pairs.parquet     the pairs to materialize

    All pairs are materialized, train AND holdout. Features carry no label, so
    materializing holdout pairs leaks nothing; the training reader separately
    restricts LABELS to the train split, and holdout features are needed later for
    evaluation.

    Memory: entities are loaded one tenant at a time (a dict keyed by entity_id);
    pairs are streamed in batches with a pyarrow dataset scan, never all at once.

    lookback_days is ignored: the synthetic dataset has no change history.
    """

    name = "synthetic"

    MANIFEST_FILE = "manifest.json"
    ENTITIES_FILE = "entities.parquet"
    PAIRS_FILE = "pairs.parquet"

    def __init__(self, data_dir: str):
        self._dir = Path(data_dir)
        self.manifest = self._load_manifest()
        self._snapshot_at = self._parse_generated_at(self.manifest)

    # ── provenance ────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict:
        path = self._dir / self.MANIFEST_FILE
        if not path.is_file():
            raise MaterializationSourceError(
                f"{path} not found. SYNTHETIC_DATA_DIR must be a directory written by "
                f"`python -m synthetic_data`."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("source") != "synthetic":
            raise MaterializationSourceError(
                f"{path}: source={manifest.get('source')!r}; expected 'synthetic'. "
                f"The old demo dataset (demo/data) is not a materialization source."
            )
        if manifest.get("is_production_data") is not False:
            raise MaterializationSourceError(
                f"{path}: is_production_data must be explicitly false, got "
                f"{manifest.get('is_production_data')!r}."
            )
        for required in (self.ENTITIES_FILE, self.PAIRS_FILE):
            if not (self._dir / required).is_file():
                raise MaterializationSourceError(f"{self._dir / required} not found.")
        return manifest

    @staticmethod
    def _parse_generated_at(manifest: dict) -> datetime:
        """Entity records carry no timestamp; the dataset's generation time is the
        moment they existed as-of. Naive UTC, matching the rest of this service."""
        raw = manifest.get("generated_at")
        if not raw:
            raise MaterializationSourceError("manifest.json has no generated_at timestamp.")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    # ── reading ───────────────────────────────────────────────────────────

    def tenants(self) -> List[str]:
        import pyarrow.parquet as pq

        table = pq.read_table(self._dir / self.PAIRS_FILE, columns=["tenant_id"])
        return sorted(set(table.column("tenant_id").to_pylist()))

    def _load_tenant_entities(self, tenant_id: str) -> Dict[str, EntitySnapshot]:
        import pyarrow.dataset as ds

        dataset = ds.dataset(self._dir / self.ENTITIES_FILE, format="parquet")
        table = dataset.to_table(
            columns=["entity_id", "tenant_id", "entity_type", "fields_json"],
            filter=ds.field("tenant_id") == tenant_id,
        )
        snapshots: Dict[str, EntitySnapshot] = {}
        for row in table.to_pylist():
            snapshots[row["entity_id"]] = EntitySnapshot(
                entity_id=row["entity_id"],
                tenant_id=row["tenant_id"],
                entity_type=row["entity_type"],
                fields=json.loads(row["fields_json"]),
                snapshot_at=self._snapshot_at,
            )
        return snapshots

    def _resolve(
        self, entities: Dict[str, EntitySnapshot], tenant_id: str, e1: str, e2: str,
    ) -> Tuple[Optional[Tuple[EntitySnapshot, EntitySnapshot]], Optional[SkippedPair]]:
        """Both snapshots, or an explicit reason why not. Never a guess."""
        if e1 == e2:
            return None, SkippedPair(e1, e2, tenant_id, "self-pair")
        missing = [e for e in (e1, e2) if e not in entities]
        if missing:
            # Entities are loaded per tenant, so an id belonging to ANOTHER tenant is
            # also reported here — a cross-tenant pair is never materialized.
            return None, SkippedPair(
                e1, e2, tenant_id, f"entity not found in tenant {tenant_id}: {missing}"
            )
        return (entities[e1], entities[e2]), None

    def iter_batches(
        self,
        tenant_id: Optional[str],
        lookback_days: int,
        batch_size: int,
    ) -> Iterator[PairBatch]:
        import pyarrow.dataset as ds

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        tenants = [tenant_id] if tenant_id else self.tenants()
        pairs_ds = ds.dataset(self._dir / self.PAIRS_FILE, format="parquet")

        for tenant in tenants:
            entities = self._load_tenant_entities(tenant)
            logger.info(
                f"Synthetic source: tenant {tenant} — {len(entities)} entity records"
            )
            batch = PairBatch(tenant_id=tenant)

            scanner = pairs_ds.scanner(
                columns=["entity_id_1", "entity_id_2", "tenant_id"],
                filter=ds.field("tenant_id") == tenant,
                batch_size=batch_size,
            )
            for record_batch in scanner.to_batches():
                for row in record_batch.to_pylist():
                    resolved, skipped = self._resolve(
                        entities, tenant, row["entity_id_1"], row["entity_id_2"]
                    )
                    if skipped:
                        batch.skipped.append(skipped)
                    else:
                        batch.pairs.append(resolved)
                    if len(batch.pairs) >= batch_size:
                        yield batch
                        batch = PairBatch(tenant_id=tenant)

            if batch.pairs or batch.skipped:
                yield batch


# ─── Construction from configuration ──────────────────────────────────────────

class MaterializationSourceNotConfigured(RuntimeError):
    """No MATERIALIZATION_SOURCE is configured."""


def build_materialization_source(s: Settings) -> MaterializationSource:
    """
    The one place a source is chosen, from configuration only. Raises
    MaterializationSourceNotConfigured when none is set, rather than returning an
    empty source that would let a job report COMPLETED having done nothing.
    """
    check_materialization_source(s)
    if s.MATERIALIZATION_SOURCE is None:
        raise MaterializationSourceNotConfigured(
            "No materialization data source is configured (MATERIALIZATION_SOURCE is "
            "unset). The production MDM entity source is not implemented yet; for local "
            "development set MATERIALIZATION_SOURCE=synthetic and SYNTHETIC_DATA_DIR."
        )
    if s.MATERIALIZATION_SOURCE == "synthetic":
        return SyntheticMaterializationSource(s.SYNTHETIC_DATA_DIR)
    # Unreachable: check_materialization_source rejects unknown values.
    raise MaterializationSourceError(f"Unsupported source {s.MATERIALIZATION_SOURCE!r}")
