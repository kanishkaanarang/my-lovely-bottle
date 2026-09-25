"""Small, shared data and metric utilities. No external identity lookups."""

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = 1
FIELDS = ["entity_id", "business_name", "business_address", "country"]
# A secondary lossy view only; raw and complete normalized names remain available.
SUFFIXES = set("inc incorporated llc ltd limited private pvt corp corporation co company llp sarl sas sasu sci sa".split())


@lru_cache(maxsize=100_000)
def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("&", " and ")
    # Preserve combining marks: \w alone destroys vowel marks in Indic scripts.
    return " ".join("".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in text).split())


def fold_latin(text):
    result, latin = [], False
    for c in unicodedata.normalize("NFD", text):
        if not unicodedata.combining(c):
            latin = "LATIN" in unicodedata.name(c, "")
        if not (latin and unicodedata.combining(c)):
            result.append(c)
    return unicodedata.normalize("NFC", "".join(result))


def core_name(text):
    return " ".join(t for t in fold_latin(normalize(text)).split() if t not in SUFFIXES)


def grams(text):
    text = "".join(core_name(text).split())
    # Hex encodes the Unicode trigram as a single safe FTS token, including marks.
    return sorted({"g" + text[i:i + 3].encode().hex() for i in range(max(0, len(text) - 2))})


@dataclass(frozen=True)
class Record:
    entity_id: str
    business_name: str
    business_address: str
    country: str

    @property
    def n(self):
        return normalize(self.business_name)

    @property
    def a(self):
        return normalize(self.business_address)


def records(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed TSV row in {path}")
            if not row["entity_id"] or not row["country"]:
                raise ValueError("Blank ID/country")
            yield Record(**row)


def truth_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected ground-truth schema")
        for row in reader:
            ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            if len(ids) != len(set(ids)) or any(not x.startswith(("S2-", "S3-")) for x in ids):
                raise ValueError("Invalid ground-truth target list")
            yield row["source1_entity_id"], set(ids)


def entity_f05(truth, prediction):
    t, p = set(truth), set(prediction)
    return float(not p) if not t else 1.25 * len(t & p) / (.25 * len(t) + len(p))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def fold(record, seed=2026):
    """Same observable signature always stays in the same split, independent of ID."""
    key = [seed, record.country, record.n, record.a]
    bucket = int(digest(key)[:16], 16) % 100
    return "train" if bucket < 70 else "calibration" if bucket < 85 else "trust"


def country_key(country):
    return hashlib.sha256(country.encode()).hexdigest()[:20]
