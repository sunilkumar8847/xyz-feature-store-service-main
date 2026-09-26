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

from src.domain.models import EntitySnapshot, FeatureVector
from src.core.config import settings

logger = logging.getLogger(__name__)


# ─── Embedding Model (singleton, thread-safe) ────────────────────────────────

_embedding_model = None


def get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}. Using fallback.")
            _embedding_model = None
    return _embedding_model


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

        def soundex_match(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            try:
                return 1.0 if jellyfish.soundex(s1) == jellyfish.soundex(s2) else 0.0
            except Exception:
                return 0.0

        def metaphone_match(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            try:
                return 1.0 if jellyfish.metaphone(s1) == jellyfish.metaphone(s2) else 0.0
            except Exception:
                return 0.0

        def nysiis_match(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            try:
                return 1.0 if jellyfish.nysiis(s1) == jellyfish.nysiis(s2) else 0.0
            except Exception:
                return 0.0

        def match_rating_match(s1: str, s2: str) -> float:
            if not s1 or not s2:
                return 0.0
            try:
                return 1.0 if jellyfish.match_rating_codex(s1) == jellyfish.match_rating_codex(s2) else 0.0
            except Exception:
                return 0.0

        return {
            "ph_soundex_name": soundex_match(first1, first2),
            "ph_metaphone_name": metaphone_match(first1, first2),
            "ph_nysiis_name": nysiis_match(first1, first2),
            "ph_match_rating_name": match_rating_match(first1, first2),
            "ph_soundex_full_name": soundex_match(n1, n2),
        }


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
        model = get_embedding_model()
        if model is None or not text:
            return None
        try:
            return model.encode(text, normalize_embeddings=True)
        except Exception as e:
            logger.debug(f"Embedding failed: {e}")
            return None

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
        feature_version: str = "v2.0.0",
    ) -> FeatureVector:
        """
        Compute all 50 features for an entity pair.
        Returns a FeatureVector with exactly 50 named features.
        """
        start = time.perf_counter()

        features: Dict[str, float] = {}

        # 1. String Similarity (15 features)
        try:
            features.update(self._string_computer.compute(entity1, entity2))
        except Exception as e:
            logger.error(f"String similarity computation failed: {e}")
            features.update({f"ss_err_{i}": 0.0 for i in range(15)})

        # 2. Phonetic (5 features)
        try:
            features.update(self._phonetic_computer.compute(entity1, entity2))
        except Exception as e:
            logger.error(f"Phonetic computation failed: {e}")
            features.update({f"ph_err_{i}": 0.0 for i in range(5)})

        # 3. Token-based (8 features)
        try:
            features.update(self._token_computer.compute(entity1, entity2))
        except Exception as e:
            logger.error(f"Token computation failed: {e}")
            features.update({f"tk_err_{i}": 0.0 for i in range(8)})

        # 4. Semantic (10 features)
        try:
            features.update(self._semantic_computer.compute(entity1, entity2))
        except Exception as e:
            logger.error(f"Semantic computation failed: {e}")
            features.update({f"sem_err_{i}": 0.0 for i in range(10)})

        # 5. Structural (7 features)
        try:
            features.update(self._structural_computer.compute(entity1, entity2))
        except Exception as e:
            logger.error(f"Structural computation failed: {e}")
            features.update({f"str_err_{i}": 0.0 for i in range(7)})

        # 6. Domain-specific (5 features)
        try:
            features.update(self._domain_computer.compute(entity1, entity2))
        except Exception as e:
            logger.error(f"Domain computation failed: {e}")
            features.update({f"dom_err_{i}": 0.0 for i in range(5)})

        # Ensure exactly 50 features
        actual_count = len(features)
        if actual_count != self.EXPECTED_FEATURE_COUNT:
            logger.warning(
                f"Expected {self.EXPECTED_FEATURE_COUNT} features, got {actual_count}. "
                f"Padding with zeros."
            )
            for i in range(actual_count, self.EXPECTED_FEATURE_COUNT):
                features[f"pad_{i}"] = 0.0

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
