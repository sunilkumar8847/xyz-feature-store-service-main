"""
feature-store-service/src/services/feature_computation.py

THE CORE ENGINE: Computes all 50 matching features for entity pairs.
This is THE single source of truth for feature computation —
used identically in training AND serving to prevent training-serving skew.

Feature Categories:
  - 15 String Similarity features
  - 5  Phonetic features
  - 8  Token-based features
  - 10 Semantic (embedding) features
  - 7  Structural features
  - 5  Domain-specific features
  = 50 total
"""
from __future__ import annotations

import logging
import math
import re
import time
from functools import lru_cache
from typing import Dict, Optional, Tuple

import jellyfish
import numpy as np
from rapidfuzz import fuzz, distance

from src.domain.feature_catalog import FEATURE_COUNT, is_canonical, non_canonical_names
from src.domain.models import EntitySnapshot, FeatureVector
from src.core.config import settings
from src.core.metrics import PHONETIC_ENCODER_FALLBACKS

logger = logging.getLogger(__name__)


# ─── Errors ──────────────────────────────────────────────────────────────────

class FeatureComputationError(RuntimeError):
    """
    A feature vector could not be computed. Raised instead of returning a vector with
    substituted values: a zero is a legitimate feature value ("no similarity"), so a
    zero standing in for "computation failed" is indistinguishable from real evidence
    and silently corrupts both training and serving.
    """

    def __init__(self, group: str, detail: str):
        self.group = group
        self.detail = detail
        super().__init__(f"feature computation failed in group '{group}': {detail}")


class EmbeddingUnavailableError(FeatureComputationError):
    """The sentence-embedding model is missing or cannot produce an embedding."""

    def __init__(self, detail: str):
        super().__init__("semantic", detail)


# ─── Embedding Model (singleton) ─────────────────────────────────────────────

_embedding_model = None


def get_embedding_model():
    """
    The sentence-transformer model, loaded once. Raises EmbeddingUnavailableError if it
    cannot be loaded. It used to log a warning and return None, after which all ten
    sem_* features were served as 0.0 under their normal names.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if settings.EMBEDDING_MODEL_REVISION:
                kwargs["revision"] = settings.EMBEDDING_MODEL_REVISION
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, **kwargs)
            logger.info(
                f"Loaded embedding model: {settings.EMBEDDING_MODEL} "
                f"(revision {settings.EMBEDDING_MODEL_REVISION or 'unpinned'})"
            )
        except Exception as e:
            raise EmbeddingUnavailableError(
                f"cannot load embedding model {settings.EMBEDDING_MODEL!r}: "
                f"{type(e).__name__}: {e}"
            ) from e
    return _embedding_model


def embedding_model_loaded() -> bool:
    """True if the model is in memory. Does NOT trigger a load (safe for health checks)."""
    return _embedding_model is not None


def embedding_model_ready() -> bool:
    """True if the embedding model is loaded (loading it if necessary)."""
    try:
        return get_embedding_model() is not None
    except EmbeddingUnavailableError as e:
        logger.error(str(e))
        return False


# ─── String Normalisation Helpers ────────────────────────────────────────────

def _norm(s: Optional[str]) -> str:
    """Normalize a string for comparison: lowercase, strip, collapse whitespace."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


def _tokenize(s: str) -> set:
    """Split into tokens, remove stop words."""
    stop = {"the", "a", "an", "and", "or", "of", "in", "at", "to"}
    return {t for t in re.split(r"[\s,.-]+", _norm(s)) if t and t not in stop}


def _phone_digits(s: Optional[str]) -> str:
    """Extract digits only from phone number."""
    if not s:
        return ""
    digits = re.sub(r"\D", "", s)
    # Strip country code if 11+ digits
    if len(digits) == 11 and digits.startswith("1"):
        return digits[1:]
    return digits


def _email_parts(email: Optional[str]) -> Tuple[str, str]:
    """Split email into local and domain parts."""
    if not email or "@" not in email:
        return "", ""
    parts = email.lower().split("@", 1)
    return parts[0], parts[1]


# ─── Individual Feature Computers ────────────────────────────────────────────


class StringSimilarityFeatures:
    """15 string similarity features."""

    @staticmethod
    def compute(e1: EntitySnapshot, e2: EntitySnapshot) -> Dict[str, float]:
        n1, n2 = _norm(e1.name), _norm(e2.name)
        a1, a2 = _norm(e1.address), _norm(e2.address)
        em1, em2 = _norm(e1.email), _norm(e2.email)

        def safe_levenshtein(s1: str, s2: str) -> float:
            if not s1 and not s2:
                return 1.0
            if not s1 or not s2:
                return 0.0
            max_len = max(len(s1), len(s2))
            lev = distance.Levenshtein.distance(s1, s2)
            return 1.0 - (lev / max_len)

        return {
            # Name features
            "ss_levenshtein_name": safe_levenshtein(n1, n2),
            "ss_jaro_winkler_name": jellyfish.jaro_winkler_similarity(n1, n2) if n1 and n2 else 0.0,
            "ss_damerau_levenshtein_name": (
                1.0 - min(
                    distance.DamerauLevenshtein.distance(n1, n2) / max(len(n1), len(n2), 1),
                    1.0
                ) if n1 and n2 else 0.0
            ),
            "ss_hamming_name": (
                1.0 - (
                    sum(c1 != c2 for c1, c2 in zip(n1.ljust(len(n2)), n2.ljust(len(n1))))
                    / max(len(n1), len(n2), 1)
                ) if n1 and n2 else 0.0
            ),
            "ss_jaro_name": (
                jellyfish.jaro_similarity(n1, n2) if n1 and n2 else 0.0
            ),
            # Address features
            "ss_levenshtein_address": safe_levenshtein(a1, a2),
            "ss_jaro_winkler_address": (
                jellyfish.jaro_winkler_similarity(a1, a2) if a1 and a2 else 0.0
            ),
            "ss_damerau_address": (
                1.0 - min(
                    distance.DamerauLevenshtein.distance(a1, a2) / max(len(a1), len(a2), 1),
                    1.0
                ) if a1 and a2 else 0.0
            ),
            # Email features
            "ss_levenshtein_email": safe_levenshtein(em1, em2),
            "ss_jaro_winkler_email": (
                jellyfish.jaro_winkler_similarity(em1, em2) if em1 and em2 else 0.0
            ),
            # Cross-field
            "ss_name_addr_cross": (
                safe_levenshtein(n1[:20], a1[:20]) if n1 and a1 else 0.0
            ),
            "ss_longest_common_subseq": (
                len(distance.LCSseq.editops(n1, n2)) / max(len(n1), len(n2), 1)
                if n1 and n2 else 0.0
            ),
            "ss_common_prefix_name": (
                len(os.path.commonprefix([n1, n2])) / max(len(n1), len(n2), 1)
                if n1 and n2 else 0.0
            ),
            "ss_osa_distance_name": (
                1.0 - min(
                    distance.OSA.distance(n1, n2) / max(len(n1), len(n2), 1),
                    1.0
                ) if n1 and n2 else 0.0
            ),
            "ss_postfix_similarity": (
                safe_levenshtein(n1[-10:], n2[-10:]) if len(n1) >= 3 and len(n2) >= 3 else 0.0
            ),
        }


class PhoneticFeatures:
    """5 phonetic features."""

    @staticmethod
    def compute(e1: EntitySnapshot, e2: EntitySnapshot) -> Dict[str, float]:
        n1, n2 = _norm(e1.name), _norm(e2.name)

        def letters_only(s: str) -> str:
            return "".join(ch for ch in s if ch.isalpha())

        # jellyfish encoders reject non-letters: "Cooper, Patricia" would give first token
        # "cooper," and the unguarded call below used to fail the whole phonetic group.
        first1 = letters_only(n1.split()[0]) if n1.split() else ""
        first2 = letters_only(n2.split()[0]) if n2.split() else ""

        return {
            "ph_soundex_name": _phonetic_equal(jellyfish.soundex, first1, first2, "ph_soundex_name"),
            "ph_metaphone_name": _phonetic_equal(jellyfish.metaphone, first1, first2, "ph_metaphone_name"),
            "ph_nysiis_name": _phonetic_equal(jellyfish.nysiis, first1, first2, "ph_nysiis_name"),
            "ph_match_rating_name": _phonetic_equal(
                jellyfish.match_rating_codex, first1, first2, "ph_match_rating_name"
            ),
            "ph_soundex_full_name": _phonetic_equal(jellyfish.soundex, n1, n2, "ph_soundex_full_name"),
        }


def _is_rust_panic(exc: BaseException) -> bool:
    """
    True only for pyo3's PanicException, raised when a Rust extension such as
    jellyfish panics. pyo3 creates this class at runtime in a module that cannot be
    imported, so it is identified by module + name rather than by isinstance.
    """
    cls = type(exc)
    return cls.__module__ == "pyo3_runtime" and cls.__name__ == "PanicException"


def _phonetic_equal(encoder, s1: str, s2: str, feature_name: str) -> float:
    """
    1.0 if `encoder` maps both strings to the same phonetic code, else 0.0.

    Error handling, narrowest first:

    * ``Exception`` — e.g. jellyfish's ValueError on non-letter input. Returns 0.0,
      exactly as before this helper existed.

    * pyo3 ``PanicException`` — jellyfish 1.0.3's match_rating_codex panics (Rust
      ``Option::unwrap`` on None) on some accented Latin-1 letters, e.g. 'richárd'
      (U+00E1) and 'wríght' (U+00ED); 'josé' and 'müller' are fine. PanicException
      derives from BaseException, NOT Exception, so it used to escape every handler
      in this module and crash the whole 50-feature computation.

      Fallback: exact equality of the (already normalised) strings. This invents no
      similarity: a deterministic encoder always maps identical input to identical
      codes, so 1.0 for identical strings is exactly what the encoder would return;
      0.0 otherwise is the conservative answer when the encoder cannot tell us more.
      The input is NOT accent-stripped — Unicode is preserved as given.

    * Any other BaseException (KeyboardInterrupt, SystemExit, ...) is re-raised
      untouched. This is deliberately not a general BaseException handler.
    """
    if not s1 or not s2:
        return 0.0
    try:
        return 1.0 if encoder(s1) == encoder(s2) else 0.0
    except Exception:
        return 0.0
    except BaseException as exc:
        if not _is_rust_panic(exc):
            raise
        PHONETIC_ENCODER_FALLBACKS.labels(feature=feature_name).inc()
        # Codepoints only — names are PII and must not reach the logs.
        non_ascii = sorted({f"U+{ord(ch):04X}" for ch in s1 + s2 if ord(ch) > 127})
        logger.warning(
            "Phonetic encoder %s panicked (non-ASCII codepoints %s); "
            "using exact-equality fallback for %s",
            getattr(encoder, "__name__", "encoder"), non_ascii or "none", feature_name,
        )
        return 1.0 if s1 == s2 else 0.0


class TokenBasedFeatures:
    """8 token-based features."""

    @staticmethod
    def compute(e1: EntitySnapshot, e2: EntitySnapshot) -> Dict[str, float]:
        n1, n2 = _norm(e1.name), _norm(e2.name)
        a1, a2 = _norm(e1.address), _norm(e2.address)

        t1_name = _tokenize(n1)
        t2_name = _tokenize(n2)
        t1_addr = _tokenize(a1)
        t2_addr = _tokenize(a2)

        def jaccard(s1: set, s2: set) -> float:
            if not s1 and not s2:
                return 1.0
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        return {
            "tk_jaccard_name": jaccard(t1_name, t2_name),
            "tk_jaccard_address": jaccard(t1_addr, t2_addr),
            "tk_token_sort_ratio_name": fuzz.token_sort_ratio(n1, n2) / 100.0,
            "tk_token_set_ratio_name": fuzz.token_set_ratio(n1, n2) / 100.0,
            "tk_partial_ratio_name": fuzz.partial_ratio(n1, n2) / 100.0,
            "tk_token_sort_address": fuzz.token_sort_ratio(a1, a2) / 100.0,
            "tk_token_set_address": fuzz.token_set_ratio(a1, a2) / 100.0,
            "tk_common_token_count": (
                len(t1_name & t2_name) / max(len(t1_name | t2_name), 1)
            ),
        }


class SemanticFeatures:
    """10 semantic (embedding-based) features."""

    @staticmethod
    def _embed(text: str) -> Optional[np.ndarray]:
        """
        Embedding of `text`, or None when the text is EMPTY (a record with no name or
        address — a defined case: the comparison scores 0.0). A missing model or a
        failed encode is not that case and raises.
        """
        if not text:
            return None
        model = get_embedding_model()
        try:
            return model.encode(text, normalize_embeddings=True)
        except Exception as e:
            raise EmbeddingUnavailableError(f"encode failed: {type(e).__name__}: {e}") from e

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity (vectors already normalized)."""
        return float(np.dot(v1, v2))

    @staticmethod
    def _euclidean_norm(v1: np.ndarray, v2: np.ndarray) -> float:
        """Normalized Euclidean distance -> similarity [0,1]."""
        dist = float(np.linalg.norm(v1 - v2))
        return max(0.0, 1.0 - dist / 2.0)  # Max distance for unit vectors is 2

    @classmethod
    def compute(cls, e1: EntitySnapshot, e2: EntitySnapshot) -> Dict[str, float]:
        n1, n2 = _norm(e1.name), _norm(e2.name)
        a1, a2 = _norm(e1.address), _norm(e2.address)
        full1 = f"{n1} {a1}".strip()
        full2 = f"{n2} {a2}".strip()

        # Compute embeddings
        emb_n1 = cls._embed(n1)
        emb_n2 = cls._embed(n2)
        emb_a1 = cls._embed(a1)
        emb_a2 = cls._embed(a2)
        emb_full1 = cls._embed(full1)
        emb_full2 = cls._embed(full2)

        def safe_cos(v1, v2) -> float:
            if v1 is None or v2 is None:
                return 0.0
            return cls._cosine(v1, v2)

        def safe_euc(v1, v2) -> float:
            if v1 is None or v2 is None:
                return 0.0
            return cls._euclidean_norm(v1, v2)

        return {
            "sem_cosine_name": safe_cos(emb_n1, emb_n2),
            "sem_euclidean_name": safe_euc(emb_n1, emb_n2),
            "sem_cosine_address": safe_cos(emb_a1, emb_a2),
            "sem_euclidean_address": safe_euc(emb_a1, emb_a2),
            "sem_cosine_full": safe_cos(emb_full1, emb_full2),
            "sem_euclidean_full": safe_euc(emb_full1, emb_full2),
            "sem_cross_name_addr": safe_cos(emb_n1, emb_a2),
            "sem_angular_name": (
                1.0 - math.acos(max(-1.0, min(1.0, safe_cos(emb_n1, emb_n2)))) / math.pi
                if emb_n1 is not None and emb_n2 is not None else 0.0
            ),
            "sem_dot_product_name": safe_cos(emb_n1, emb_n2),  # Already normalized
            "sem_cosine_name_addr_concat": safe_cos(emb_full1, emb_full2),
        }


class StructuralFeatures:
    """7 structural features."""

    @staticmethod
    def compute(e1: EntitySnapshot, e2: EntitySnapshot) -> Dict[str, float]:
        f1 = e1.fields
        f2 = e2.fields

        all_fields = set(f1.keys()) | set(f2.keys())
        both_present = sum(1 for k in all_fields if f1.get(k) and f2.get(k))
        either_present = sum(1 for k in all_fields if f1.get(k) or f2.get(k))
        only_one = sum(1 for k in all_fields if bool(f1.get(k)) != bool(f2.get(k)))

        n1, n2 = _norm(e1.name), _norm(e2.name)
        len_ratio = (
            min(len(n1), len(n2)) / max(len(n1), len(n2), 1)
            if n1 and n2 else 0.0
        )

        null_count_1 = sum(1 for v in f1.values() if not v)
        null_count_2 = sum(1 for v in f2.values() if not v)
        null_diff = abs(null_count_1 - null_count_2) / max(len(f1), len(f2), 1)

        return {
            "str_field_presence_ratio": both_present / max(either_present, 1),
            "str_length_ratio_name": len_ratio,
            "str_null_count_diff": 1.0 - null_diff,
            "str_field_overlap": both_present / max(len(all_fields), 1),
            "str_schema_similarity": (
                len(set(f1.keys()) & set(f2.keys())) / max(len(set(f1.keys()) | set(f2.keys())), 1)
            ),
            "str_asymmetric_null_ratio": 1.0 - (only_one / max(len(all_fields), 1)),
            "str_word_count_ratio_name": (
                min(len(n1.split()), len(n2.split())) / max(len(n1.split()), len(n2.split()), 1)
                if n1 and n2 else 0.0
            ),
        }


class DomainSpecificFeatures:
    """5 domain-specific features."""

    @staticmethod
    def compute(e1: EntitySnapshot, e2: EntitySnapshot) -> Dict[str, float]:
        # Email domain match
        _, domain1 = _email_parts(e1.email)
        _, domain2 = _email_parts(e2.email)
        email_domain_match = 1.0 if (domain1 and domain2 and domain1 == domain2) else 0.0

        # Phone prefix match (area code = first 3 digits)
        ph1 = _phone_digits(e1.phone)
        ph2 = _phone_digits(e2.phone)
        phone_prefix_match = (
            1.0 if ph1 and ph2 and len(ph1) >= 3 and len(ph2) >= 3 and ph1[:3] == ph2[:3]
            else 0.0
        )

        # Full phone match
        phone_full_match = 1.0 if ph1 and ph2 and ph1 == ph2 else 0.0

        # Geographic distance (normalized to [0,1], 0=far, 1=same location)
        geo_similarity = 0.0
        ll1 = e1.lat_lng
        ll2 = e2.lat_lng
        if ll1 and ll2:
            dist_km = _haversine_km(ll1[0], ll1[1], ll2[0], ll2[1])
            geo_similarity = max(0.0, 1.0 - dist_km / 1000.0)  # 1000km = 0 similarity

        # Email local part similarity
        local1, _ = _email_parts(e1.email)
        local2, _ = _email_parts(e2.email)
        email_local_sim = (
            jellyfish.jaro_winkler_similarity(local1, local2)
            if local1 and local2 else 0.0
        )

        return {
            "dom_email_domain_match": email_domain_match,
            "dom_phone_prefix_match": phone_prefix_match,
            "dom_phone_full_match": phone_full_match,
            "dom_geo_similarity": geo_similarity,
            "dom_email_local_similarity": email_local_sim,
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula: great-circle distance in km."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Main Feature Computation Service ───────────────────────────────────────


import os  # noqa: E402 – needed by StringSimilarityFeatures


class FeatureComputationService:
    """
    Orchestrates computation of all 50 features for an entity pair.
    This is THE single truth for feature computation — same code path
    for training and serving (zero training-serving skew).
    """

    EXPECTED_FEATURE_COUNT = 50

    def __init__(self):
        self._string_computer = StringSimilarityFeatures()
        self._phonetic_computer = PhoneticFeatures()
        self._token_computer = TokenBasedFeatures()
        self._semantic_computer = SemanticFeatures()
        self._structural_computer = StructuralFeatures()
        self._domain_computer = DomainSpecificFeatures()
        logger.info("FeatureComputationService initialized with all 50 feature computers")

    def compute(
        self,
        entity1: EntitySnapshot,
        entity2: EntitySnapshot,
        feature_version: Optional[str] = None,
    ) -> FeatureVector:
        """
        Compute all 50 features for an entity pair.
        Returns a FeatureVector with exactly the 50 catalog features, or raises
        FeatureComputationError. It never returns substituted values.
        """
        feature_version = feature_version or settings.FEATURE_VERSION
        if feature_version != settings.FEATURE_VERSION:
            # The code computes ONE catalog version; labelling its output as another
            # would let two different feature definitions share a version string.
            raise FeatureComputationError(
                "catalog",
                f"requested feature_version {feature_version!r} but this service "
                f"computes {settings.FEATURE_VERSION!r}",
            )
        start = time.perf_counter()

        features: Dict[str, float] = {}

        groups = (
            ("string_similarity", self._string_computer),
            ("phonetic", self._phonetic_computer),
            ("token", self._token_computer),
            ("semantic", self._semantic_computer),
            ("structural", self._structural_computer),
            ("domain", self._domain_computer),
        )
        for group, computer in groups:
            try:
                features.update(computer.compute(entity1, entity2))
            except FeatureComputationError:
                raise
            except Exception as e:
                # Previously: log, then substitute 0.0 under placeholder names
                # (ss_err_0, ph_err_0, ...) and return the vector as if it were valid.
                raise FeatureComputationError(group, f"{type(e).__name__}: {e}") from e

        # The vector must be exactly the catalog. It used to be padded with pad_N = 0.0.
        if not is_canonical(features.keys()):
            raise FeatureComputationError(
                "catalog",
                f"computed {len(features)} features that do not match the "
                f"{FEATURE_COUNT}-feature catalog (e.g. {non_canonical_names(features.keys())[:3]})",
            )
        bad = sorted(k for k, v in features.items() if v is None or not math.isfinite(v))
        if bad:
            raise FeatureComputationError("catalog", f"non-finite values for {bad[:3]}")

        # Clip all to [0, 1] range
        features = {k: max(0.0, min(1.0, v)) for k, v in features.items()}

        elapsed_ms = (time.perf_counter() - start) * 1000

        return FeatureVector(
            entity_id_1=entity1.entity_id,
            entity_id_2=entity2.entity_id,
            tenant_id=entity1.tenant_id,
            features=features,
            feature_version=feature_version,
            computation_ms=elapsed_ms,
        )

    def get_feature_names(self) -> list:
        """Return sorted list of all 50 feature names."""
        # Build a minimal snapshot pair to get feature names
        dummy = EntitySnapshot(
            entity_id="dummy", tenant_id="dummy",
            entity_type="customer",
            fields={"name": "John Doe", "email": "john@example.com", "phone": "5551234567"}
        )
        fv = self.compute(dummy, dummy)
        return sorted(fv.features.keys())
