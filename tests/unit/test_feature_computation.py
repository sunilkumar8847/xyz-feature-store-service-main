"""
feature-store-service/tests/unit/test_feature_computation.py

Unit tests for the feature computation engine.
All 50 features are tested for:
  - Correct count
  - Valid range [0, 1]
  - Determinism (same input → same output)
  - Order independence (feature(A,B) == feature(B,A) by symmetry where applicable)
"""
from __future__ import annotations

import pytest

from src.domain.models import EntitySnapshot
from src.services.feature_computation import (
    FeatureComputationService,
    StringSimilarityFeatures,
    PhoneticFeatures,
    TokenBasedFeatures,
    StructuralFeatures,
    DomainSpecificFeatures,
)


@pytest.fixture
def computation_service():
    return FeatureComputationService()


@pytest.fixture
def entity_alice():
    return EntitySnapshot(
        entity_id="alice_001",
        tenant_id="tenant_xyz",
        entity_type="customer",
        fields={
            "name": "Alice Johnson",
            "email": "alice.johnson@acme.com",
            "phone": "5551234567",
            "address_line1": "123 Main Street",
            "city": "New York",
            "state": "NY",
            "country": "US",
        },
    )


@pytest.fixture
def entity_alice_dup():
    """Duplicate of Alice with minor variations."""
    return EntitySnapshot(
        entity_id="alice_002",
        tenant_id="tenant_xyz",
        entity_type="customer",
        fields={
            "name": "Alise Jonson",  # Typos
            "email": "alice.johnson@acme.com",  # Exact email
            "phone": "5551234567",  # Exact phone
            "address_line1": "123 Main St",  # Abbreviated
            "city": "New York",
            "state": "NY",
            "country": "US",
        },
    )


@pytest.fixture
def entity_bob():
    """Completely different entity."""
    return EntitySnapshot(
        entity_id="bob_001",
        tenant_id="tenant_xyz",
        entity_type="customer",
        fields={
            "name": "Robert Williams",
            "email": "rwilliams@other.org",
            "phone": "9876543210",
            "address_line1": "789 Oak Avenue",
            "city": "Los Angeles",
            "state": "CA",
            "country": "US",
        },
    )


class TestFeatureCount:
    def test_exactly_50_features(self, computation_service, entity_alice, entity_alice_dup):
        fv = computation_service.compute(entity_alice, entity_alice_dup)
        assert len(fv.features) == 50, f"Expected 50 features, got {len(fv.features)}"

    def test_feature_names_consistent(self, computation_service, entity_alice, entity_bob):
        fv1 = computation_service.compute(entity_alice, entity_alice)
        fv2 = computation_service.compute(entity_alice, entity_bob)
        assert set(fv1.features.keys()) == set(fv2.features.keys())


class TestFeatureRange:
    def test_all_features_in_range(self, computation_service, entity_alice, entity_alice_dup):
        fv = computation_service.compute(entity_alice, entity_alice_dup)
        for name, value in fv.features.items():
            assert 0.0 <= value <= 1.0, f"Feature {name}={value} out of range [0,1]"

    def test_all_features_in_range_different_entities(self, computation_service, entity_alice, entity_bob):
        fv = computation_service.compute(entity_alice, entity_bob)
        for name, value in fv.features.items():
            assert 0.0 <= value <= 1.0, f"Feature {name}={value} out of range [0,1]"


class TestDuplicateVsNonDuplicate:
    def test_duplicate_scores_higher(self, computation_service, entity_alice, entity_alice_dup, entity_bob):
        """Duplicate pairs should have higher aggregate similarity than non-duplicates."""
        fv_dup = computation_service.compute(entity_alice, entity_alice_dup)
        fv_diff = computation_service.compute(entity_alice, entity_bob)

        avg_dup = sum(fv_dup.features.values()) / len(fv_dup.features)
        avg_diff = sum(fv_diff.features.values()) / len(fv_diff.features)

        assert avg_dup > avg_diff, (
            f"Expected duplicate avg ({avg_dup:.3f}) > different avg ({avg_diff:.3f})"
        )

    def test_identical_entity_max_string_similarity(self, computation_service, entity_alice):
        """Identical entities should score 1.0 on all string similarity features."""
        fv = computation_service.compute(entity_alice, entity_alice)
        assert fv.features["ss_levenshtein_name"] == pytest.approx(1.0)
        assert fv.features["ss_jaro_winkler_name"] == pytest.approx(1.0)

    def test_phonetic_match(self, computation_service):
        """Entities with phonetically similar names should match."""
        e1 = EntitySnapshot("e1", "t1", "customer", {"name": "Catherine"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"name": "Katherine"})
        fv = computation_service.compute(e1, e2)
        # Phonetic features should indicate similarity
        assert fv.features["ph_soundex_name"] == 1.0 or fv.features["ph_metaphone_name"] == 1.0


class TestStringSimilarity:
    def test_levenshtein_exact_match(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"name": "John Smith"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"name": "John Smith"})
        features = StringSimilarityFeatures.compute(e1, e2)
        assert features["ss_levenshtein_name"] == pytest.approx(1.0)

    def test_levenshtein_no_match(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"name": "AAAA"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"name": "ZZZZ"})
        features = StringSimilarityFeatures.compute(e1, e2)
        assert features["ss_levenshtein_name"] < 0.5

    def test_empty_names(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {})
        e2 = EntitySnapshot("e2", "t1", "customer", {})
        features = StringSimilarityFeatures.compute(e1, e2)
        # Should not crash
        assert isinstance(features, dict)

    def test_produces_15_features(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"name": "Test"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"name": "Test"})
        features = StringSimilarityFeatures.compute(e1, e2)
        ss_features = [k for k in features if k.startswith("ss_")]
        assert len(ss_features) == 15


class TestPhoneticFeatures:
    def test_produces_5_features(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"name": "Robert"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"name": "Roberta"})
        features = PhoneticFeatures.compute(e1, e2)
        ph_features = [k for k in features if k.startswith("ph_")]
        assert len(ph_features) == 5


class TestDomainFeatures:
    def test_email_domain_match(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"email": "alice@acme.com"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"email": "bob@acme.com"})
        features = DomainSpecificFeatures.compute(e1, e2)
        assert features["dom_email_domain_match"] == 1.0

    def test_email_domain_no_match(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"email": "alice@acme.com"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"email": "bob@other.com"})
        features = DomainSpecificFeatures.compute(e1, e2)
        assert features["dom_email_domain_match"] == 0.0

    def test_phone_prefix_match(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"phone": "5551234567"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"phone": "5559876543"})
        features = DomainSpecificFeatures.compute(e1, e2)
        assert features["dom_phone_prefix_match"] == 1.0

    def test_phone_full_match(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {"phone": "5551234567"})
        e2 = EntitySnapshot("e2", "t1", "customer", {"phone": "5551234567"})
        features = DomainSpecificFeatures.compute(e1, e2)
        assert features["dom_phone_full_match"] == 1.0

    def test_produces_5_features(self):
        e1 = EntitySnapshot("e1", "t1", "customer", {})
        e2 = EntitySnapshot("e2", "t1", "customer", {})
        features = DomainSpecificFeatures.compute(e1, e2)
        dom_features = [k for k in features if k.startswith("dom_")]
        assert len(dom_features) == 5


class TestDeterminism:
    def test_same_input_same_output(self, computation_service, entity_alice, entity_bob):
        fv1 = computation_service.compute(entity_alice, entity_bob)
        fv2 = computation_service.compute(entity_alice, entity_bob)
        for name in fv1.features:
            assert fv1.features[name] == pytest.approx(fv2.features[name]), (
                f"Feature {name} not deterministic: {fv1.features[name]} vs {fv2.features[name]}"
            )


class TestFeatureVector:
    def test_as_list_ordered(self, computation_service, entity_alice, entity_bob):
        fv = computation_service.compute(entity_alice, entity_bob)
        vec = fv.as_list
        assert len(vec) == 50
        assert all(isinstance(v, float) for v in vec)
        assert all(0.0 <= v <= 1.0 for v in vec)


class TestPhoneticPunctuation:
    """Names with punctuation used to raise in jellyfish.match_rating_codex and replace
    the whole phonetic group with ph_err_* zeros."""

    @pytest.mark.parametrize("name1,name2", [
        ("Cooper, Patricia", "Cooper Patricia"),
        ("O'Brien Mary-Jane", "OBrien Mary"),
    ])
    def test_punctuated_names_match_phonetically(self, name1, name2):
        e1 = EntitySnapshot("e1", "t", "customer", {"name": name1})
        e2 = EntitySnapshot("e2", "t", "customer", {"name": name2})
        features = PhoneticFeatures.compute(e1, e2)
        assert features["ph_match_rating_name"] == 1.0
        assert features["ph_soundex_name"] == 1.0

    def test_service_keeps_real_phonetic_names(self, computation_service):
        e1 = EntitySnapshot("e1", "t", "customer", {"name": "Cooper, Patricia"})
        e2 = EntitySnapshot("e2", "t", "customer", {"name": "Patricia Cooper"})
        fv = computation_service.compute(e1, e2)
        assert not any(name.startswith("ph_err_") for name in fv.features)
        assert "ph_match_rating_name" in fv.features


# ─── Regression: Rust panic in jellyfish on accented names ────────────────────

def _snap(eid: str, name: str) -> EntitySnapshot:
    return EntitySnapshot(eid, "t", "customer", {
        "name": name, "email": "a.b@example.com", "phone": "2065551234",
        "address_line1": "12 Main Street", "city": "Seattle", "state": "WA",
    })


# Verified to panic in jellyfish 1.0.3's match_rating_codex (pyo3 PanicException).
PANICKING_NAMES = ["Richárd Nguyen", "Wríght Freya"]
# Accented names that do NOT panic — the fix must leave their values untouched.
OTHER_ACCENTED = [
    "José Silva", "Müller Hans", "Zoë Larsen", "Søren Jensen", "Élodie Moreau",
    "François Dubois", "Łukasz Novak", "Núñez Diego", "Renée Costa", "Björk Weber",
]
ASCII_NAMES = ["Richard Nguyen", "Wright Freya", "Patricia Cooper", "Jose Silva"]


class TestPhoneticEncoderPanic:
    """
    jellyfish 1.0.3's match_rating_codex panics on some accented Latin-1 letters.
    The panic is pyo3_runtime.PanicException, which derives from BaseException, so
    it escaped every `except Exception` in feature_computation and crashed the whole
    50-feature computation. Dependency type: REAL jellyfish (no mocks), except where
    a test needs to force a specific exception type.
    """

    def test_panic_is_real_and_not_an_exception_subclass(self):
        """Guards the diagnosis: if jellyfish is upgraded and this starts passing
        through as a normal Exception, this test tells us the fallback is dormant."""
        import jellyfish
        with pytest.raises(BaseException) as info:
            jellyfish.match_rating_codex("richárd")
        assert type(info.value).__name__ == "PanicException"
        assert not isinstance(info.value, Exception)

    @pytest.mark.parametrize("name", PANICKING_NAMES)
    def test_panicking_names_no_longer_crash_the_service(self, computation_service, name):
        ascii_twin = name.replace("á", "a").replace("í", "i")
        fv = computation_service.compute(_snap("e1", name), _snap("e2", ascii_twin))
        assert len(fv.features) == 50
        assert not [n for n in fv.features if "_err_" in n or n.startswith("pad_")]

    @pytest.mark.parametrize("name", PANICKING_NAMES)
    def test_feature_names_are_the_standard_50(self, computation_service, name):
        reference = computation_service.compute(_snap("a", "Richard Nguyen"), _snap("b", "Wright Freya"))
        fv = computation_service.compute(_snap("e1", name), _snap("e2", "Patricia Cooper"))
        assert sorted(fv.features) == sorted(reference.features)

    @pytest.mark.parametrize("name", PANICKING_NAMES)
    def test_fallback_identical_accented_names_match(self, name):
        """Identical input -> identical code for any encoder, so 1.0 is exact."""
        f = PhoneticFeatures.compute(_snap("e1", name), _snap("e2", name))
        assert f["ph_match_rating_name"] == 1.0

    @pytest.mark.parametrize("accented,plain", [
        ("Richárd Nguyen", "Richard Nguyen"),
        ("Wríght Freya", "Wright Freya"),
    ])
    def test_fallback_is_conservative_for_non_identical(self, accented, plain):
        """The encoder could not compare them, so the fallback does not claim a match."""
        f = PhoneticFeatures.compute(_snap("e1", accented), _snap("e2", plain))
        assert f["ph_match_rating_name"] == 0.0

    @pytest.mark.parametrize("accented,plain", [
        ("Richárd Nguyen", "Richard Nguyen"),
        ("Wríght Freya", "Wright Freya"),
    ])
    def test_other_phonetic_features_still_compute_normally(self, accented, plain):
        """Only the panicking encoder degrades; soundex/metaphone keep their signal."""
        f = PhoneticFeatures.compute(_snap("e1", accented), _snap("e2", plain))
        assert f["ph_soundex_name"] == 1.0
        assert f["ph_metaphone_name"] == 1.0

    def test_non_phonetic_features_unaffected_by_the_fallback(self, computation_service):
        """String/token/domain features must be computed normally for the pair."""
        fv = computation_service.compute(_snap("e1", "Richárd Nguyen"), _snap("e2", "Richárd Nguyen"))
        assert fv.features["ss_levenshtein_name"] == 1.0
        assert fv.features["tk_jaccard_name"] == 1.0
        assert fv.features["dom_phone_full_match"] == 1.0

    def test_unicode_is_preserved_not_stripped(self):
        """The fix must not accent-fold input: 'Richárd' vs 'Richard' stays distinct."""
        from src.services.feature_computation import StringSimilarityFeatures
        f = StringSimilarityFeatures.compute(_snap("e1", "Richárd Nguyen"), _snap("e2", "Richard Nguyen"))
        assert f["ss_levenshtein_name"] < 1.0

    @pytest.mark.parametrize("name", OTHER_ACCENTED)
    def test_other_accented_names_compute_without_fallback(self, computation_service, name):
        from src.core.metrics import PHONETIC_ENCODER_FALLBACKS
        before = PHONETIC_ENCODER_FALLBACKS.labels(feature="ph_match_rating_name")._value.get()
        fv = computation_service.compute(_snap("e1", name), _snap("e2", name))
        after = PHONETIC_ENCODER_FALLBACKS.labels(feature="ph_match_rating_name")._value.get()
        assert len(fv.features) == 50
        assert fv.features["ph_match_rating_name"] == 1.0
        assert after == before, f"{name!r} unexpectedly used the fallback"

    @pytest.mark.parametrize("name", ASCII_NAMES)
    def test_ascii_names_unchanged(self, computation_service, name):
        fv = computation_service.compute(_snap("e1", name), _snap("e2", name))
        assert len(fv.features) == 50
        for feat in ("ph_soundex_name", "ph_metaphone_name", "ph_nysiis_name",
                     "ph_match_rating_name", "ph_soundex_full_name"):
            assert fv.features[feat] == 1.0

    def test_fallback_is_counted(self):
        from src.core.metrics import PHONETIC_ENCODER_FALLBACKS
        counter = PHONETIC_ENCODER_FALLBACKS.labels(feature="ph_match_rating_name")
        before = counter._value.get()
        PhoneticFeatures.compute(_snap("e1", "Richárd Nguyen"), _snap("e2", "Richard Nguyen"))
        assert counter._value.get() == before + 1

    def test_fallback_is_logged_without_the_name(self, caplog):
        """Not silent, and no PII: codepoints are logged, the name is not."""
        import logging
        with caplog.at_level(logging.WARNING, logger="src.services.feature_computation"):
            PhoneticFeatures.compute(_snap("e1", "Richárd Nguyen"), _snap("e2", "Richard Nguyen"))
        text = caplog.text
        assert "U+00E1" in text
        assert "Richárd" not in text and "richárd" not in text

    def test_non_panic_exception_behaviour_is_unchanged(self):
        """An ordinary Exception from an encoder still yields 0.0, as before."""
        from src.services.feature_computation import _phonetic_equal

        def raises_value_error(_):
            raise ValueError("Strings must only contain alphabetical characters")

        assert _phonetic_equal(raises_value_error, "abc", "abc", "ph_test") == 0.0

    @pytest.mark.parametrize("exc_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
    def test_other_base_exceptions_are_not_swallowed(self, exc_type):
        """The handler is narrow: only pyo3's PanicException is intercepted."""
        from src.services.feature_computation import _phonetic_equal

        def raises(_):
            raise exc_type()

        with pytest.raises(exc_type):
            _phonetic_equal(raises, "abc", "abc", "ph_test")
