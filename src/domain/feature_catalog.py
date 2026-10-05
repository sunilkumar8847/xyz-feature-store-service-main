"""
feature-store-service/src/domain/feature_catalog.py

The canonical 50-feature catalog: the single definition of WHICH features exist and in
WHAT ORDER they are served. Feature computation, the online store, the offline store and
the API all check against this module, so a vector that does not carry exactly these
names can never be stored or served as if it were valid.

The feature DEFINITIONS (how each value is computed) live in
src/services/feature_computation.py and are not changed by this module.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List

FEATURE_COUNT = 50

# Sorted by name: FeatureVector.as_list, the Parquet schema and every API response use
# this order.
FEATURE_NAMES: List[str] = sorted([
    # String Similarity (15)
    "ss_levenshtein_name", "ss_jaro_winkler_name", "ss_damerau_levenshtein_name",
    "ss_hamming_name", "ss_jaro_name", "ss_levenshtein_address",
    "ss_jaro_winkler_address", "ss_damerau_address", "ss_levenshtein_email",
    "ss_jaro_winkler_email", "ss_name_addr_cross", "ss_longest_common_subseq",
    "ss_common_prefix_name", "ss_osa_distance_name", "ss_postfix_similarity",
    # Phonetic (5)
    "ph_soundex_name", "ph_metaphone_name", "ph_nysiis_name",
    "ph_match_rating_name", "ph_soundex_full_name",
    # Token-based (8)
    "tk_jaccard_name", "tk_jaccard_address", "tk_token_sort_ratio_name",
    "tk_token_set_ratio_name", "tk_partial_ratio_name", "tk_token_sort_address",
    "tk_token_set_address", "tk_common_token_count",
    # Semantic (10)
    "sem_cosine_name", "sem_euclidean_name", "sem_cosine_address",
    "sem_euclidean_address", "sem_cosine_full", "sem_euclidean_full",
    "sem_cross_name_addr", "sem_angular_name", "sem_dot_product_name",
    "sem_cosine_name_addr_concat",
    # Structural (7)
    "str_field_presence_ratio", "str_length_ratio_name", "str_null_count_diff",
    "str_field_overlap", "str_schema_similarity", "str_asymmetric_null_ratio",
    "str_word_count_ratio_name",
    # Domain-specific (5)
    "dom_email_domain_match", "dom_phone_prefix_match", "dom_phone_full_match",
    "dom_geo_similarity", "dom_email_local_similarity",
])
assert len(FEATURE_NAMES) == FEATURE_COUNT and len(set(FEATURE_NAMES)) == FEATURE_COUNT

_FEATURE_NAME_SET = frozenset(FEATURE_NAMES)


def feature_names_sha256(names: Iterable[str] = FEATURE_NAMES) -> str:
    """Same fingerprint training and the inference service use for the name order."""
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def is_canonical(feature_names: Iterable[str]) -> bool:
    """True only if these are exactly the 50 catalog names (no more, no fewer)."""
    return frozenset(feature_names) == _FEATURE_NAME_SET and len(list(feature_names)) == FEATURE_COUNT


def non_canonical_names(feature_names: Iterable[str]) -> List[str]:
    names = set(feature_names)
    return sorted((names - _FEATURE_NAME_SET) | (_FEATURE_NAME_SET - names))
