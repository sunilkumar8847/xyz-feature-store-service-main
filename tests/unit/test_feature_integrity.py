"""
Feature integrity: a failed computation must never become a valid-looking vector.

Dependency type: REAL FeatureComputationService (incl. the real embedding model for the
happy-path checks); failures are INJECTED by replacing one collaborator. No network
services.
"""
from __future__ import annotations

import math
from unittest.mock import AsyncMock

import pytest

import src.services.feature_computation as fc
from src.adapters.materialization_sources import MaterializationSource, PairBatch
from src.core.config import settings
from src.domain.feature_catalog import (
    FEATURE_COUNT, FEATURE_NAMES, feature_names_sha256, is_canonical,
)
from src.domain.models import EntitySnapshot, MaterializationJob, MaterializationStatus
from src.services.feature_computation import (
    EmbeddingUnavailableError, FeatureComputationError, FeatureComputationService,
)
from src.workers.materialization import MaterializationWorker

T1 = "00000000-0000-0000-0000-000000000001"


def snap(eid, **fields):
    base = {"name": "Patricia Cooper", "email": "pcooper@example.com", "phone": "2065551234",
            "address_line1": "12 Maple Street", "city": "Seattle", "state": "WA"}
    base.update(fields)
    return EntitySnapshot(eid, T1, "customer", base)


@pytest.fixture(scope="module")
def service():
    return FeatureComputationService()


class TestCatalog:
    def test_catalog_is_50_sorted_unique_names(self):
        assert len(FEATURE_NAMES) == FEATURE_COUNT == 50
        assert FEATURE_NAMES == sorted(set(FEATURE_NAMES))

    def test_real_computation_produces_exactly_the_catalog(self, service):
        fv = service.compute(snap("A"), snap("B", name="Patricia Coper"))
        assert is_canonical(fv.features.keys())
        assert sorted(fv.features) == FEATURE_NAMES
        assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in fv.features.values())
        assert fv.feature_version == settings.FEATURE_VERSION

    def test_name_fingerprint_matches_training_and_inference_algorithm(self):
        import hashlib
        assert feature_names_sha256() == hashlib.sha256("\n".join(FEATURE_NAMES).encode()).hexdigest()


class TestGroupFailureIsNotZeroFilled:
    @pytest.mark.parametrize("attr,group", [
        ("_string_computer", "string_similarity"), ("_phonetic_computer", "phonetic"),
        ("_token_computer", "token"), ("_structural_computer", "structural"),
        ("_domain_computer", "domain"),
    ])
    def test_failing_group_raises_instead_of_err_zeros(self, monkeypatch, attr, group):
        """Used to return 50 values with ss_err_0 = 0.0 ... placeholder names."""
        svc = FeatureComputationService()

        class Boom:
            @staticmethod
            def compute(e1, e2):
                raise ValueError("injected failure")

        monkeypatch.setattr(svc, attr, Boom())
        with pytest.raises(FeatureComputationError) as exc:
            svc.compute(snap("A"), snap("B"))
        assert exc.value.group == group
        assert "injected failure" in str(exc.value)

    def test_missing_feature_is_not_padded(self, monkeypatch):
        """Used to be padded with pad_N = 0.0 up to 50 values."""
        svc = FeatureComputationService()
        real = svc._domain_computer

        class OneShort:
            @staticmethod
            def compute(e1, e2):
                out = dict(real.compute(e1, e2))
                out.pop("dom_geo_similarity")
                return out

        monkeypatch.setattr(svc, "_domain_computer", OneShort())
        with pytest.raises(FeatureComputationError, match="catalog"):
            svc.compute(snap("A"), snap("B"))

    def test_non_finite_value_is_rejected(self, monkeypatch):
        svc = FeatureComputationService()
        real = svc._domain_computer

        class Nan:
            @staticmethod
            def compute(e1, e2):
                out = dict(real.compute(e1, e2))
                out["dom_geo_similarity"] = float("nan")
                return out

        monkeypatch.setattr(svc, "_domain_computer", Nan())
        with pytest.raises(FeatureComputationError, match="non-finite"):
            svc.compute(snap("A"), snap("B"))


class TestEmbeddingModel:
    def test_unloadable_model_raises(self, monkeypatch):
        """Used to log a warning and return None -> all sem_* served as 0.0."""
        import sentence_transformers

        def explode(*a, **k):
            raise OSError("model files not found")

        monkeypatch.setattr(fc, "_embedding_model", None)
        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", explode)
        with pytest.raises(EmbeddingUnavailableError, match="cannot load embedding model"):
            fc.get_embedding_model()
        assert fc.embedding_model_ready() is False
        assert fc.embedding_model_loaded() is False

    def test_compute_fails_when_embedding_unavailable(self, monkeypatch, service):
        def unavailable():
            raise EmbeddingUnavailableError("injected: model missing")

        monkeypatch.setattr(fc, "get_embedding_model", unavailable)
        with pytest.raises(EmbeddingUnavailableError):
            service.compute(snap("A"), snap("B"))

    def test_encode_failure_raises(self, monkeypatch, service):
        class BrokenModel:
            def encode(self, *a, **k):
                raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(fc, "get_embedding_model", lambda: BrokenModel())
        with pytest.raises(EmbeddingUnavailableError, match="encode failed"):
            service.compute(snap("A"), snap("B"))

    def test_empty_text_is_a_defined_zero_not_a_failure(self, service):
        """Feature DEFINITION unchanged: a record with no address scores 0.0 on address
        semantics. That is evidence ("nothing to compare"), not a failed computation."""
        a = snap("A", address_line1=None, city=None, state=None)
        b = snap("B", address_line1=None, city=None, state=None)
        fv = service.compute(a, b)
        assert fv.features["sem_cosine_address"] == 0.0
        assert fv.features["sem_cosine_name"] > 0.9
        assert is_canonical(fv.features.keys())

    def test_pinned_revision_is_passed_to_the_loader(self, monkeypatch):
        import sentence_transformers
        seen = {}

        class Fake:
            def __init__(self, name, **kwargs):
                seen.update(name=name, **kwargs)

        monkeypatch.setattr(fc, "_embedding_model", None)
        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Fake)
        monkeypatch.setattr(settings, "EMBEDDING_MODEL_REVISION", "abc123")
        fc.get_embedding_model()
        assert seen == {"name": settings.EMBEDDING_MODEL, "revision": "abc123"}


class TestFeatureVersion:
    def test_other_version_label_is_refused(self, service):
        """The code computes ONE catalog version; it must not label output as another."""
        with pytest.raises(FeatureComputationError, match="feature_version"):
            service.compute(snap("A"), snap("B"), "v9.9.9")

    def test_default_is_the_configured_version(self, service):
        assert service.compute(snap("A"), snap("B")).feature_version == settings.FEATURE_VERSION


class _OneBatch(MaterializationSource):
    name = "fake"

    def iter_batches(self, tenant_id, lookback_days, batch_size):
        return iter([PairBatch(T1, pairs=[(snap("A"), snap("B")), (snap("C"), snap("D"))])])


class _Offline:
    def __init__(self):
        self.writes = []

    def write_features(self, vectors, partition_dt=None, data_source=None, source_id=None):
        self.writes.append(list(vectors))


class _Online:
    def __init__(self):
        self.writes = []

    async def set_batch(self, vectors, ttl_hours=None):
        self.writes.append(list(vectors))
        return len(vectors)


class TestMaterializationWithFailedComputation:
    async def test_embedding_outage_fails_the_job_and_stores_nothing(self, monkeypatch):
        def unavailable():
            raise EmbeddingUnavailableError("injected: model missing")

        monkeypatch.setattr(fc, "get_embedding_model", unavailable)
        offline, online = _Offline(), _Online()
        reg = AsyncMock()
        worker = MaterializationWorker(online, offline, FeatureComputationService(), reg, _OneBatch())
        job = await worker.run_job(MaterializationJob(triggered_by="test"))

        assert job.status == MaterializationStatus.FAILED
        assert "Embedding model unavailable" in job.error_message
        assert offline.writes == [] and online.writes == []

    async def test_one_failing_pair_is_counted_not_zero_filled(self, monkeypatch):
        svc = FeatureComputationService()
        real = svc._string_computer

        class FailForA:
            @staticmethod
            def compute(e1, e2):
                if e1.entity_id == "A":
                    raise ValueError("bad record")
                return real.compute(e1, e2)

        monkeypatch.setattr(svc, "_string_computer", FailForA())
        offline, online = _Offline(), _Online()
        worker = MaterializationWorker(online, offline, svc, AsyncMock(), _OneBatch())
        job = await worker.run_job(MaterializationJob(triggered_by="test"))

        assert job.status == MaterializationStatus.COMPLETED
        assert job.failed_entities == 1 and job.processed_entities == 1
        stored = [fv for w in offline.writes for fv in w]
        assert [fv.entity_id_1 for fv in stored] == ["C"]
        assert all(is_canonical(fv.features.keys()) for fv in stored)
