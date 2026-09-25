"""Country-partitioned, disk-backed candidate retrieval using SQLite FTS5."""

import json
import sqlite3
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz.fuzz import ratio, token_sort_ratio

from .common import (Record, SCHEMA_VERSION, core_name, country_key, digest,
                     file_hash, fold_latin, grams, normalize, records, write_json)


@dataclass(frozen=True)
class RetrievalConfig:
    per_channel: int = 30
    per_source: int = 80
    max_per_source: int = 160
    adaptive: bool = True

    def __post_init__(self):
        if self.per_channel < 1 or self.per_source < 1 or self.max_per_source < self.per_source:
            raise ValueError("Invalid candidate budgets")


def build_index(source_files, index_dir):
    """Index the entire target pool, never a validation-only subset."""
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = digest({"schema": SCHEMA_VERSION,
                          "files": [(Path(p).name, file_hash(p)) for p in source_files]})
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] == fingerprint and all((root / f).is_file() for f in previous["countries"].values()):
            return previous
        raise ValueError("Index inputs changed. Choose a new index directory; do not reuse stale artifacts.")
    connections, paths, counts = {}, {}, defaultdict(int)
    started = time.monotonic()
    try:
        for source in source_files:
            for rec in records(source):
                src = rec.entity_id[:2]
                if src not in ("S2", "S3"):
                    raise ValueError(f"Unexpected target ID: {rec.entity_id}")
                if rec.country not in connections:
                    name = country_key(rec.country) + ".sqlite"
                    partial = root / (name + ".building")
                    partial.unlink(missing_ok=True)
                    db = sqlite3.connect(partial)
                    db.execute("PRAGMA journal_mode=OFF")
                    db.execute("PRAGMA synchronous=OFF")
                    db.execute("PRAGMA cache_size=-64000")
                    db.execute("CREATE TABLE targets(id TEXT UNIQUE, name TEXT, address TEXT, src TEXT, n TEXT, a TEXT, core TEXT)")
                    db.execute("CREATE VIRTUAL TABLE search USING fts5(n,a,g,s,content='',tokenize=\"unicode61 remove_diacritics 0 categories 'L* N* Co M*'\")")
                    connections[rec.country] = db
                    paths[rec.country] = name
                db = connections[rec.country]
                cursor = db.execute("INSERT INTO targets VALUES(?,?,?,?,?,?,?)",
                                    (rec.entity_id, rec.business_name, rec.business_address, src, rec.n, rec.a, core_name(rec.business_name)))
                db.execute("INSERT INTO search(rowid,n,a,g,s) VALUES(?,?,?,?,?)",
                           (cursor.lastrowid, fold_latin(rec.n), fold_latin(rec.a), " ".join(grams(rec.business_name)), src.lower()))
                counts[rec.country] += 1
                if sum(counts.values()) % 100_000 == 0:
                    for connection in connections.values():
                        connection.commit()
                    print(f"Indexed {sum(counts.values()):,} targets in {time.monotonic()-started:.0f}s", flush=True)
        for country, db in connections.items():
            for col in ("n", "a", "core"):
                db.execute(f"CREATE INDEX by_{col} ON targets(src,{col})")
            db.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(search,'col')")
            db.execute("INSERT INTO search(search) VALUES('optimize')")
            db.commit()
            db.close()
            (root / (paths[country] + ".building")).replace(root / paths[country])
        result = {"fingerprint": fingerprint, "schema": SCHEMA_VERSION,
                  "countries": paths, "counts": dict(counts), "seconds": time.monotonic()-started}
        if not paths:
            raise ValueError("No target records found")
        write_json(manifest_path, result)
        return result
    finally:
        for db in connections.values():
            db.close()


@dataclass
class Candidate:
    record: Record
    channels: set
    rank: int
    heuristic: float


class Retriever:
    def __init__(self, index_dir, config=None):
        self.root = Path(index_dir)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA_VERSION:
            raise ValueError("Index schema mismatch")
        self.config = config or RetrievalConfig()
        self.connections = {}

    def db(self, country):
        if country not in self.manifest["countries"]:
            return None
        if country not in self.connections:
            path = self.root / self.manifest["countries"][country]
            self.connections[country] = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        return self.connections[country]

    @lru_cache(maxsize=100_000)
    def frequency(self, country, column, term):
        db = self.db(country)
        row = db.execute("SELECT doc FROM vocab WHERE term=? AND col=?", (term, column)).fetchone()
        return row[0] if row else 0

    def _search(self, country, column, terms, src, k):
        terms = [(self.frequency(country, column, t), t) for t in set(terms)]
        rare = [t for n, t in sorted(terms) if n > 0][:8]
        if not rare:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in rare)
        query = f's:"{src.lower()}" AND {column}:({expression})'
        return [r[0] for r in self.db(country).execute("SELECT rowid FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?", (query, k))]

    def retrieve(self, anchor):
        db = self.db(anchor.country)
        if db is None:
            return []
        found = {}

        def add(ids, channel):
            for rank, rowid in enumerate(ids):
                if rowid not in found:
                    row = db.execute("SELECT id,name,address FROM targets WHERE rowid=?", (rowid,)).fetchone()
                    target = Record(*row, anchor.country)
                    ns = max(ratio(fold_latin(anchor.n), fold_latin(target.n)),
                             token_sort_ratio(core_name(anchor.business_name), core_name(target.business_name))) / 100
                    ads = token_sort_ratio(fold_latin(anchor.a), fold_latin(target.a)) / 100 if anchor.a and target.a else 0
                    found[rowid] = Candidate(target, set(), rank, .55 * ns + .45 * ads)
                found[rowid].channels.add(channel)
                found[rowid].rank = min(found[rowid].rank, rank)

        for src in ("S2", "S3"):
            for column, value in (("n", anchor.n), ("core", core_name(anchor.business_name)), ("a", anchor.a)):
                if value:
                    ids = [r[0] for r in db.execute(f"SELECT rowid FROM targets WHERE src=? AND {column}=? ORDER BY rowid LIMIT ?",
                                                    (src, value, self.config.max_per_source))]
                    add(ids, "exact_" + column)
            for column, terms in (("n", fold_latin(anchor.n).split()), ("a", fold_latin(anchor.a).split()), ("g", grams(anchor.business_name))):
                add(self._search(anchor.country, column, terms, src, self.config.per_channel), "fts_" + column)
        ranked = sorted(found.values(), key=lambda x: (-x.heuristic, x.record.entity_id))
        uncertain = (not anchor.a or not ranked or ranked[0].heuristic < .75
                     or (len(ranked) > 1 and ranked[0].heuristic-ranked[1].heuristic < .03))
        if self.config.adaptive and uncertain:
            for src in ("S2", "S3"):
                for col, terms in (("n", fold_latin(anchor.n).split()), ("a", fold_latin(anchor.a).split()), ("g", grams(anchor.business_name))):
                    add(self._search(anchor.country, col, terms, src, self.config.per_channel * 2), "fts_" + col)
        budget = self.config.max_per_source if self.config.adaptive and uncertain else self.config.per_source
        selected = []
        for src in ("S2", "S3"):
            pool = (v for v in found.values() if v.record.entity_id.startswith(src + "-"))
            selected.extend(sorted(pool, key=lambda v: (-v.heuristic, v.record.entity_id))[:budget])
        return selected

    def contains(self, country, entity_id):
        db = self.db(country)
        return db is not None and db.execute("SELECT 1 FROM targets WHERE id=?", (entity_id,)).fetchone() is not None

    def close(self):
        for db in self.connections.values():
            db.close()
        self.connections.clear()
        self.frequency.cache_clear()
