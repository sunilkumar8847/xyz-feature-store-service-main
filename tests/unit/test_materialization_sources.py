"""
Unit tests for materialization data sources (src/adapters/materialization_sources.py).

Dependency type:
  - REAL synthetic dataset, written to tmp_path by the repository-root generator
    (`synthetic_data`). Skipped if that package is not importable.
  - Hand-built minimal datasets (pyarrow, generator schema) for edge cases.
No network, no S3, no Redis.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.adapters.materialization_sources import (
    MaterializationSourceError,
    MaterializationSourceNotConfigured,
    SyntheticMaterializationSource,
    build_materialization_source,
)
from src.core.config import Environment, check_materialization_source, settings

REPO_ROOT = Path(__file__).resolve().parents[4]
T1 = "00000000-0000-0000-0000-000000000001"
T2 = "00000000-0000-0000-0000-000000000002"
GENERATED_AT = "2026-09-20T08:30:00+00:00"


def _synthetic_package():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return pytest.importorskip("synthetic_data")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """A real smoke dataset produced by `synthetic_data`, plus its in-memory objects."""
    sd = _synthetic_package()
    out = tmp_path_factory.mktemp("synthetic_smoke")
    config = dataclasses.replace(sd.get_profile("smoke"), n_identities=25)
    identities, records, pairs = sd.SyntheticDatasetGenerator(config).generate()
    sd.write_dataset(out, config, identities, records, pairs)
    return out, records, pairs


def _write_minimal(path: Path, entities, pairs, manifest=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest if manifest is not None else {
        "source": "synthetic", "is_production_data": False, "generated_at": GENERATED_AT,
    }), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([
        {"entity_id": e, "tenant_id": t, "entity_type": "customer", "fields_json": json.dumps(f)}
        for e, t, f in entities
    ]), path / "entities.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"entity_id_1": a, "entity_id_2": b, "tenant_id": t} for a, b, t in pairs
    ]), path / "pairs.parquet")
    return path


def _all(source, tenant_id=None, batch_size=1000):
    return list(source.iter_batches(tenant_id, lookback_days=30, batch_size=batch_size))


# ─── 1. Loads the actual generated entities ───────────────────────────────────

class TestLoadsGeneratedDataset:
    def test_every_generated_entity_is_loaded(self, generated):
        out, records, _ = generated
        source = SyntheticMaterializationSource(str(out))
        loaded = {}
        for tenant in source.tenants():
            loaded.update(source._load_tenant_entities(tenant))
        assert set(loaded) == {r.entity_id for r in records}

    def test_entity_fields_match_generator_exactly(self, generated):
        """Including OMITTED keys — they drive str_schema_similarity."""
        out, records, _ = generated
        source = SyntheticMaterializationSource(str(out))
        loaded = {}
        for tenant in source.tenants():
            loaded.update(source._load_tenant_entities(tenant))
        for r in records:
            assert loaded[r.entity_id].fields == r.fields
            assert set(loaded[r.entity_id].fields) == set(r.fields)

    # ── 2. Loads the generated pairs ──

    def test_every_generated_pair_is_yielded(self, generated):
        """All splits: features carry no label, so holdout pairs leak nothing."""
        out, _, pairs = generated
        batches = _all(SyntheticMaterializationSource(str(out)))
        yielded = {(a.entity_id, b.entity_id) for batch in batches for a, b in batch.pairs}
        assert yielded == {(p.entity_id_1, p.entity_id_2) for p in pairs}
        assert sum(len(b.skipped) for b in batches) == 0

    def test_tenants_come_from_the_dataset(self, generated):
        out, records, _ = generated
        assert SyntheticMaterializationSource(str(out)).tenants() == sorted(
            {r.tenant_id for r in records}
        )


# ─── 3. EntitySnapshot construction ───────────────────────────────────────────

class TestSnapshotConstruction:
    def test_snapshot_carries_id_tenant_type_and_fields(self, tmp_path):
        d = _write_minimal(tmp_path, [
            ("E1", T1, {"name": "Ada Lovelace", "email": "ada@x.com"}),
            ("E2", T1, {"name": "Ada Lovelase", "email": None}),
        ], [("E1", "E2", T1)])
        (batch,) = _all(SyntheticMaterializationSource(str(d)))
        ((s1, s2),) = batch.pairs
        assert (s1.entity_id, s1.tenant_id, s1.entity_type) == ("E1", T1, "customer")
        assert s1.fields == {"name": "Ada Lovelace", "email": "ada@x.com"}
        assert s2.fields == {"name": "Ada Lovelase", "email": None}
        assert "phone" not in s2.fields

    def test_snapshot_at_is_dataset_generation_time_naive_utc(self, tmp_path):
        d = _write_minimal(tmp_path, [("E1", T1, {"name": "A"}), ("E2", T1, {"name": "B"})],
                           [("E1", "E2", T1)])
        (batch,) = _all(SyntheticMaterializationSource(str(d)))
        s1, _ = batch.pairs[0]
        assert s1.snapshot_at == datetime(2026, 9, 20, 8, 30)
        assert s1.snapshot_at.tzinfo is None


# ─── 4 / 13. Tenants preserved and isolated ───────────────────────────────────

class TestTenantIsolation:
    @pytest.fixture
    def two_tenants(self, tmp_path):
        return _write_minimal(tmp_path, [
            ("A1", T1, {"name": "Ann"}), ("A2", T1, {"name": "Anne"}),
            ("B1", T2, {"name": "Bob"}), ("B2", T2, {"name": "Rob"}),
        ], [("A1", "A2", T1), ("B1", "B2", T2)])

    def test_every_batch_is_single_tenant(self, two_tenants):
        for batch in _all(SyntheticMaterializationSource(str(two_tenants))):
            for s1, s2 in batch.pairs:
                assert s1.tenant_id == s2.tenant_id == batch.tenant_id

    def test_tenants_are_yielded_separately(self, two_tenants):
        batches = _all(SyntheticMaterializationSource(str(two_tenants)))
        assert sorted(b.tenant_id for b in batches) == [T1, T2]

    def test_tenant_filter_returns_only_that_tenant(self, two_tenants):
        batches = _all(SyntheticMaterializationSource(str(two_tenants)), tenant_id=T2)
        assert [b.tenant_id for b in batches] == [T2]
        assert {s.entity_id for b in batches for p in b.pairs for s in p} == {"B1", "B2"}

    def test_cross_tenant_pair_is_skipped_not_materialized(self, tmp_path):
        """A pair filed under T1 that references a T2 entity must never resolve."""
        d = _write_minimal(tmp_path, [
            ("A1", T1, {"name": "Ann"}), ("B1", T2, {"name": "Bob"}),
        ], [("A1", "B1", T1)])
        (batch,) = _all(SyntheticMaterializationSource(str(d)))
        assert batch.pairs == []
        (sk,) = batch.skipped
        assert sk.tenant_id == T1 and "not found in tenant" in sk.reason and "B1" in sk.reason


# ─── 10. Missing / invalid entity ids are explicit ────────────────────────────

class TestUnresolvablePairs:
    def test_unknown_entity_is_skipped_with_reason(self, tmp_path):
        d = _write_minimal(tmp_path, [("E1", T1, {"name": "A"}), ("E2", T1, {"name": "B"})],
                           [("E1", "E2", T1), ("E1", "GHOST", T1)])
        (batch,) = _all(SyntheticMaterializationSource(str(d)))
        assert len(batch.pairs) == 1
        (sk,) = batch.skipped
        assert (sk.entity_id_1, sk.entity_id_2) == ("E1", "GHOST")
        assert "GHOST" in sk.reason

    def test_self_pair_is_skipped(self, tmp_path):
        d = _write_minimal(tmp_path, [("E1", T1, {"name": "A"})], [("E1", "E1", T1)])
        (batch,) = _all(SyntheticMaterializationSource(str(d)))
        assert batch.pairs == [] and batch.skipped[0].reason == "self-pair"

    def test_invalid_batch_size_rejected(self, tmp_path):
        d = _write_minimal(tmp_path, [("E1", T1, {"name": "A"})], [])
        with pytest.raises(ValueError, match="batch_size"):
            _all(SyntheticMaterializationSource(str(d)), batch_size=0)


# ─── Batching / streaming ─────────────────────────────────────────────────────

class TestBatching:
    def test_batches_respect_batch_size(self, generated):
        out, _, pairs = generated
        batches = _all(SyntheticMaterializationSource(str(out)), batch_size=7)
        assert all(len(b.pairs) <= 7 for b in batches)
        assert sum(len(b.pairs) for b in batches) == len(pairs)


# ─── Provenance ───────────────────────────────────────────────────────────────

class TestProvenance:
    def test_old_demo_directory_is_refused(self, tmp_path):
        (tmp_path / "customers.json").write_text("[]", encoding="utf-8")
        (tmp_path / "training_pairs.json").write_text("[]", encoding="utf-8")
        (tmp_path / "manifest.json").write_text(json.dumps({"seed": 42}), encoding="utf-8")
        with pytest.raises(MaterializationSourceError, match="expected 'synthetic'"):
            SyntheticMaterializationSource(str(tmp_path))

    def test_must_declare_not_production(self, tmp_path):
        d = _write_minimal(tmp_path, [], [], manifest={"source": "synthetic", "generated_at": GENERATED_AT})
        with pytest.raises(MaterializationSourceError, match="is_production_data"):
            SyntheticMaterializationSource(str(d))

    def test_missing_directory_is_refused(self, tmp_path):
        with pytest.raises(MaterializationSourceError, match="not found"):
            SyntheticMaterializationSource(str(tmp_path / "nope"))

    def test_source_name_is_recorded(self, tmp_path):
        d = _write_minimal(tmp_path, [], [])
        assert SyntheticMaterializationSource(str(d)).name == "synthetic"


# ─── Configuration ────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_unset_source_raises_not_configured(self, monkeypatch):
        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", None)
        with pytest.raises(MaterializationSourceNotConfigured, match="not implemented yet"):
            build_materialization_source(settings)

    def test_synthetic_source_built_from_config(self, monkeypatch, tmp_path):
        d = _write_minimal(tmp_path, [], [])
        monkeypatch.setattr(settings, "ENVIRONMENT", Environment.DEVELOPMENT)
        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", "synthetic")
        monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(d))
        assert isinstance(build_materialization_source(settings), SyntheticMaterializationSource)

    @pytest.mark.parametrize("env", [Environment.STAGING, Environment.PRODUCTION])
    def test_synthetic_refused_outside_development(self, monkeypatch, tmp_path, env):
        monkeypatch.setattr(settings, "ENVIRONMENT", env)
        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", "synthetic")
        monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="development/test only"):
            check_materialization_source(settings)

    def test_synthetic_requires_a_data_dir(self, monkeypatch):
        monkeypatch.setattr(settings, "ENVIRONMENT", Environment.DEVELOPMENT)
        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", "synthetic")
        monkeypatch.setattr(settings, "SYNTHETIC_DATA_DIR", None)
        with pytest.raises(ValueError, match="requires SYNTHETIC_DATA_DIR"):
            check_materialization_source(settings)

    def test_unknown_source_rejected_mdm_not_faked(self, monkeypatch):
        """There is no MDM implementation; asking for one must fail, not fake it."""
        monkeypatch.setattr(settings, "MATERIALIZATION_SOURCE", "mdm")
        with pytest.raises(ValueError, match="not supported"):
            check_materialization_source(settings)
