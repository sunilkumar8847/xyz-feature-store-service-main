"""
Bootstrap mode: the company-owned synthetic SEED corpus may be materialized outside
development/test only when it is explicitly enabled AND pinned by its manifest hash.

Dependency type: a REAL dataset written by the repository-root generator; no services.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from src.adapters.materialization_sources import (
    MaterializationSourceError, SyntheticMaterializationSource, build_materialization_source,
)
from src.core.config import Environment, Settings, check_materialization_source

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sd = pytest.importorskip("synthetic_data")
    out = tmp_path_factory.mktemp("seed_corpus")
    config = dataclasses.replace(sd.get_profile("smoke"), n_identities=10)
    identities, records, pairs = sd.SyntheticDatasetGenerator(config).generate()
    manifest = sd.write_dataset(out, config, identities, records, pairs)
    return out, manifest, hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()


def cfg(env, data_dir, **kw) -> Settings:
    return Settings(ENVIRONMENT=env, MATERIALIZATION_SOURCE="synthetic",
                    SYNTHETIC_DATA_DIR=str(data_dir), **kw)


class TestDevelopment:
    def test_dev_accepts_a_verified_dataset_without_bootstrap_mode(self, corpus):
        out, manifest, _ = corpus
        src = SyntheticMaterializationSource(str(out), settings=cfg(Environment.DEVELOPMENT, out))
        assert src.source_id == manifest["dataset_id"]

    def test_source_id_is_the_dataset_identity(self, corpus):
        out, manifest, digest = corpus
        src = build_materialization_source(cfg(Environment.TEST, out))
        assert src.source_id == manifest["dataset_id"] and src.source_id.startswith("synds-")
        assert src.manifest_sha256 == digest


class TestProductionIsProtected:
    @pytest.mark.parametrize("env", [Environment.STAGING, Environment.PRODUCTION])
    def test_synthetic_refused_without_bootstrap_mode(self, corpus, env):
        out, _, _ = corpus
        with pytest.raises(ValueError, match="BOOTSTRAP_MODE"):
            check_materialization_source(cfg(env, out))

    def test_bootstrap_mode_without_a_pin_is_refused(self, corpus):
        out, _, _ = corpus
        with pytest.raises(ValueError, match="BOOTSTRAP_DATASET_MANIFEST_SHA256"):
            check_materialization_source(cfg(Environment.PRODUCTION, out, BOOTSTRAP_MODE=True))

    def test_pinned_corpus_is_accepted(self, corpus):
        out, manifest, digest = corpus
        s = cfg(Environment.PRODUCTION, out, BOOTSTRAP_MODE=True,
                BOOTSTRAP_DATASET_MANIFEST_SHA256=digest.upper())   # case-insensitive
        src = build_materialization_source(s)
        assert src.source_id == manifest["dataset_id"]

    def test_arbitrary_synthetic_directory_is_refused_even_in_bootstrap_mode(self, corpus):
        out, _, _ = corpus
        s = cfg(Environment.PRODUCTION, out, BOOTSTRAP_MODE=True,
                BOOTSTRAP_DATASET_MANIFEST_SHA256="0" * 64)
        with pytest.raises(MaterializationSourceError, match="not an approved bootstrap corpus"):
            build_materialization_source(s)

    def test_pin_without_bootstrap_mode_is_not_enough(self, corpus):
        out, _, digest = corpus
        with pytest.raises(ValueError, match="BOOTSTRAP_MODE"):
            check_materialization_source(
                cfg(Environment.STAGING, out, BOOTSTRAP_DATASET_MANIFEST_SHA256=digest))


class TestIntegrity:
    def _copy(self, corpus, tmp_path):
        d = tmp_path / "copy"
        shutil.copytree(corpus[0], d)
        return d

    def test_modified_data_file_is_refused_in_every_environment(self, corpus, tmp_path):
        d = self._copy(corpus, tmp_path)
        with open(d / "pairs.parquet", "ab") as f:
            f.write(b"tampered")
        with pytest.raises(MaterializationSourceError, match="does not match the manifest"):
            SyntheticMaterializationSource(str(d), settings=cfg(Environment.DEVELOPMENT, d))

    def test_edited_identity_is_refused(self, corpus, tmp_path):
        d = self._copy(corpus, tmp_path)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        m["identity"]["seed"] = 7
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(MaterializationSourceError, match="dataset_id"):
            SyntheticMaterializationSource(str(d), settings=cfg(Environment.DEVELOPMENT, d))

    def test_dataset_without_identity_is_refused(self, corpus, tmp_path):
        d = self._copy(corpus, tmp_path)
        m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        for k in ("identity", "dataset_id", "file_sha256"):
            m.pop(k)
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        with pytest.raises(MaterializationSourceError, match="no dataset identity"):
            SyntheticMaterializationSource(str(d), settings=cfg(Environment.DEVELOPMENT, d))

    def test_pinned_corpus_that_was_modified_is_refused_in_production(self, corpus, tmp_path):
        """The pin is the manifest hash; the manifest holds the file hashes."""
        d = self._copy(corpus, tmp_path)
        with open(d / "entities.parquet", "ab") as f:
            f.write(b"x")
        s = cfg(Environment.PRODUCTION, d, BOOTSTRAP_MODE=True,
                BOOTSTRAP_DATASET_MANIFEST_SHA256=corpus[2])
        with pytest.raises(MaterializationSourceError, match="does not match the manifest"):
            build_materialization_source(s)
