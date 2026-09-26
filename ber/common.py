"""Small, shared data and metric utilities. No external identity lookups."""

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = 3
TSV = dict(delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=None)
FIELDS = ["entity_id", "business_name", "business_address", "country"]
# A secondary lossy view only; raw and complete normalized names remain available.
SUFFIXES = set("inc incorporated llc ltd limited private pvt corp corporation co company llp sarl sas sasu sci sa plc lp pllc pc pa dba gmbh eurl snc scop selarl scp gie opc प्राइवेट लिमिटेड प्रा लि প্রাইভেট লিমিটেড பிரைவேட் லிமிடெட்".split())


def digits_ascii(text):
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


def clean_nulls(text):
    # Remove only complete placeholder tokens, never substrings of real names.
    return re.sub(r"(?<!\w)(?:n\s*/\s*a|null|nan|none)(?!\w)", " ", text, flags=re.I).strip()


@lru_cache(maxsize=100_000)
def normalize(text):
    text = digits_ascii(unicodedata.normalize("NFKC", clean_nulls(text))).casefold().replace("&", " and ")
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


def core_name(text, country="", fold_accents=True):
    value = normalize(text)
    tokens = (fold_latin(value) if fold_accents else value).split()
    stops = SUFFIXES | (set("l d de la le les du des societe ste etablissements ets".split()) if country == "France" else set())
    aliases = {"shri": "sri", "shree": "sri", "ent": "enterprises", "trader": "traders"} if country == "India" else {}
    return " ".join(aliases.get(t, t) for t in tokens if t not in stops)


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
        reader = csv.DictReader(f, **TSV)
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
        reader = csv.DictReader(f, **TSV)
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError("Unexpected ground-truth schema")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed ground-truth row")
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


def partition_anchors(anchors, truth, seed=2026, strategy="signature"):
    """60/20/20 grouped split stratified by country and truth-cardinality bin.

    Conflicting labels in the same observable group remain together. A separate
    noise report is needed to interpret those groups.
    """
    from collections import defaultdict
    groups, strata = defaultdict(list), defaultdict(list)
    for a in anchors:
        if strategy not in {"signature", "ambiguity"}: raise ValueError("Unknown split strategy")
        key = [a.country, a.n, a.a] if strategy == "signature" else [a.country, sorted(core_name(a.business_name, a.country).split())]
        groups[digest(key)].append(a)
    for group, members in groups.items():
        cardinality = max(len(truth[a.entity_id]) for a in members)
        strata[members[0].country, min(cardinality, 4)].append(group)
    assignment = {}
    for keys in strata.values():
        keys.sort(key=lambda k: digest([seed, k]))
        for i, key in enumerate(keys):
            fraction = i/max(1, len(keys))
            part = "train" if fraction < .6 else "calibration" if fraction < .8 else "trust"
            for a in groups[key]: assignment[a.entity_id] = part
    return assignment


def audit_inputs(data, full=False, train_only=False):
    """Explicit dataset gate. Train-only audits never open test files."""
    root = Path(data)
    paths = [root/s/f"{s}_source{i}.tsv" for s in (["train"] if train_only else ["train", "test"]) for i in (1, 2, 3)]
    paths.append(root/"train/train_ground_truth.tsv")
    counts = {}
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Missing required file: {path}")
        iterator = truth_rows(path) if "ground_truth" in path.name else records(path)
        counts[str(path.relative_to(root))] = sum(1 for _ in iterator)
    if counts["train/train_source1.tsv"] != counts["train/train_ground_truth.tsv"]:
        raise ValueError("Training anchor/ground-truth counts disagree")
    if full and not train_only and counts["test/test_source1.tsv"] != 1_732_544:
        raise ValueError("Unexpected official test anchor count (expected 1,732,544)")
    return {"files": len(paths), "counts": counts, "train_only": train_only}
