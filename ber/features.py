"""Pair evidence and candidate context; country labels are not one-hot encoded."""

import math
import re
from functools import lru_cache

import numpy as np
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio, partial_ratio
from rapidfuzz.distance import JaroWinkler

from .common import core_name, fold_latin
from .views import views, script
from .retrieval import CHANNELS

EXTRA_NAMES = ["idf_overlap", "rare_overlap", "log_name_frequency", "log_address_frequency", "log_anchor_frequency",
    "name_tfidf", "address_tfidf", "roman_similarity", "acronym_equal", "jaro_winkler", "partial_ratio",
    "prefix_ratio", "containment", "monge_elkan", "house_range_overlap", "house_suffix_conflict",
    "unit_conflict", "fraction_conflict", "postcode_equal", "postcode_conflict", "locality_similarity",
    "same_script", "script_pair", "anchor_address_missing", "target_address_missing", "anchor_no_digit",
    "target_no_digit", "address_contradiction", "url_like", "all_caps", "name_length", "channel_count",
    "cross_source_support"]
FEATURE_NAMES = [
    "name_ratio", "name_token_sort", "name_token_set", "core_name_ratio",
    "name_jaccard", "address_ratio", "address_token_sort", "address_jaccard",
    "name_exact", "address_exact", "core_exact", "address_missing",
    "digit_jaccard", "digits_disjoint", "first_number_equal", "name_length_ratio",
    "address_length_ratio", "target_non_ascii", "anchor_non_ascii", "source3",
    "log_candidate_count", "retrieval_rank", "heuristic", "top_gap", "top_margin",
] + list(CHANNELS) + EXTRA_NAMES


@lru_cache(maxsize=50_000)
def extra_pair(aname, aaddress, bname, baddress, country):
    a, b = views(aname, aaddress, country), views(bname, baddress, country)
    an, bn = a["folded"], b["folded"]
    at, bt = an.split(), bn.split()
    acronym = lambda ts: "".join(t[0] for t in ts if t)
    scripts = ["EMPTY", "LATIN", "DEVANAGARI", "BENGALI", "TAMIL", "OTHER", "MIXED"]
    prefix = 0
    for x, y in zip(an, bn):
        if x != y:
            break
        prefix += 1
    soft = sum(max((JaroWinkler.normalized_similarity(t, u) for u in bt), default=0) for t in at)/max(1, len(at))
    overlap = a["house_lo"] >= 0 and b["house_lo"] >= 0 and max(a["house_lo"], b["house_lo"]) <= min(a["house_hi"], b["house_hi"])
    conflict = lambda key: bool(a[key] and b[key] and a[key] != b[key])
    return [bool(at and bt and (acronym(at) == acronym(bt) or acronym(at) == bn or an == acronym(bt))),
            JaroWinkler.normalized_similarity(an, bn) if an and bn else 0, partial_ratio(an, bn)/100 if an and bn else 0,
            prefix/max(1, len(an), len(bn)), bool(an and bn and (an in bn or bn in an)), soft,
            overlap, conflict("house_suffix"), conflict("unit"), conflict("fraction"),
            bool(a["postcode"] and a["postcode"] == b["postcode"]), conflict("postcode"),
            ratio(a["locality"], b["locality"])/100 if a["locality"] and b["locality"] else 0,
            a["script"] == b["script"], scripts.index(a["script"])*len(scripts)+scripts.index(b["script"]),
            not a["a"], not b["a"], bool(a["a"] and not any(c.isdigit() for c in a["a"])),
            bool(b["a"] and not any(c.isdigit() for c in b["a"])),
            bool(a["a"] and b["a"] and (conflict("postcode") or (a["house_lo"] >= 0 and b["house_lo"] >= 0 and not overlap))),
            bool(re.search(r"https?://|www\.", aname+" "+bname)), aname.isupper() or bname.isupper(), len(bn)]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.


def length_ratio(a, b):
    return min(len(a), len(b)) / max(len(a), len(b), 1)


def features(anchor, candidates):
    top = sorted((c.heuristic for c in candidates), reverse=True)
    margin = top[0] - top[1] if len(top) > 1 else 1.
    output = []
    an, aa, ac = fold_latin(anchor.n), fold_latin(anchor.a), core_name(anchor.business_name, anchor.country)
    adigits = re.findall(r"\d+", aa)
    for candidate in candidates:
        target = candidate.record
        v = candidate.view or views(target.business_name, target.business_address, target.country)
        bn, ba, bc = v["folded"], v["folded_address"], v["core"]
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
            candidate.evidence.get("idf_overlap", 0), candidate.evidence.get("rare_overlap", 0),
            *[math.log1p(candidate.evidence.get(k, 0)) for k in ("name_frequency", "address_frequency", "anchor_frequency")],
            *[candidate.evidence.get(k, 0) for k in ("name_tfidf", "address_tfidf", "roman_similarity")],
            *extra_pair(anchor.business_name, anchor.business_address, target.business_name, target.business_address, anchor.country),
            len(candidate.channels),
            max((token_sort_ratio(bc, (c.view or views(c.record.business_name, c.record.business_address, c.record.country))["core"])/100
                 for c in candidates[:20] if c.record.entity_id[:2] != target.entity_id[:2]), default=0),
        ])
    return np.asarray(output, dtype=np.float32).reshape((-1, len(FEATURE_NAMES)))
