"""Pair evidence and candidate context; country labels are not one-hot encoded."""

import math
import re

import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio

from .common import core_name, fold_latin

CHANNELS = ("exact_n", "exact_core", "exact_a", "fts_n", "fts_a", "fts_g")
FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "core_name_ratio",
    "name_jaccard", "address_ratio", "address_token_sort", "address_jaccard",
    "name_exact", "address_exact", "core_exact", "address_missing",
    "digit_jaccard", "digits_disjoint", "first_number_equal", "name_length_ratio",
    "address_length_ratio", "target_non_ascii", "anchor_non_ascii", "source3",
    "log_candidate_count", "retrieval_rank", "heuristic", "top_gap", "top_margin",
] + list(CHANNELS)


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.


def length_ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b), 1)


def features(anchor, candidates):
    top = sorted((c.heuristic for c in candidates), reverse=True)
    margin = top[0] - top[1] if len(top) > 1 else 1.
    output = []
    an, aa, ac = fold_latin(anchor.n), fold_latin(anchor.a), core_name(anchor.business_name)
    adigits = re.findall(r"\d+", aa)
    for candidate in candidates:
        target = candidate.record
        bn, ba, bc = fold_latin(target.n), fold_latin(target.a), core_name(target.business_name)
        bdigits = re.findall(r"\d+", ba)
        same_core = bool(ac and bc and ac == bc)
        output.append([
            ratio(an, bn)/100, token_sort_ratio(an, bn)/100, token_set_ratio(an, bn)/100,
            ratio(ac, bc)/100 if ac and bc else 0., jaccard(an.split(), bn.split()),
            ratio(aa, ba)/100 if aa and ba else 0., token_sort_ratio(aa, ba)/100 if aa and ba else 0.,
            jaccard(aa.split(), ba.split()), bool(an and an == bn), bool(aa and aa == ba), same_core,
            not aa or not ba, jaccard(adigits, bdigits),
            bool(adigits and bdigits and not set(adigits) & set(bdigits)),
            bool(adigits and bdigits and adigits[0] == bdigits[0]),
            length_ratio(an, bn), length_ratio(aa, ba), not target.business_name.isascii(),
            not anchor.business_name.isascii(), target.entity_id.startswith("S3-"),
            math.log1p(len(candidates)), 1/(candidate.rank+1), candidate.heuristic,
            top[0]-candidate.heuristic, margin,
            *[channel in candidate.channels for channel in CHANNELS],
        ])
    return np.asarray(output, dtype=np.float32).reshape((-1, len(FEATURE_NAMES)))
