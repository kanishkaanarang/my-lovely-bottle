"""Country-partitioned FTS retrieval with channel reservations and stored views."""
import heapq
import json
import math
import sqlite3
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rapidfuzz.fuzz import ratio, token_sort_ratio

from .common import Record, SCHEMA_VERSION, country_key, digest, file_hash, records, write_json
from .language import Language
from .views import chargrams, views

EXACT = ("n", "a", "core", "folded", "sorted", "compact", "postcode", "street_key", "phonetic", "roman", "indic")
CHANNELS = tuple("exact_"+c for c in EXACT) + ("fts_n", "fts_a", "fts_g", "fts_ag", "fts_roman", "rare_pair", "alias")


@dataclass(frozen=True)
class RetrievalConfig:
    per_channel: int = 30
    per_source: int = 80
    max_per_source: int = 160
    adaptive: bool = True
    reserved_per_channel: int = 2
    disabled_channels: tuple = ()
    phonetic: bool = False

    def __post_init__(self):
        if self.per_channel < 1 or self.per_source < 1 or self.max_per_source < self.per_source or self.reserved_per_channel < 1:
            raise ValueError("Invalid candidate budgets")


def build_index(source_files, index_dir, language_path=None, disabled_views=()):
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)
    language = Language(language_path)
    fingerprint = digest({"schema": SCHEMA_VERSION, "language": language.payload, "disabled_views": disabled_views,
                          "files": [(Path(p).name, file_hash(p)) for p in source_files]})
    manifest_path = root/"manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] == fingerprint and all((root/f).is_file() for f in previous["countries"].values()):
            return previous
        raise ValueError("Index inputs/schema changed; choose a new index directory")
    connections, paths, counts = {}, {}, defaultdict(int)
    started = time.monotonic()
    try:
        for source in source_files:
            for rec in records(source):
                src = rec.entity_id[:2]
                if src not in ("S2", "S3"):
                    raise ValueError("Invalid target ID")
                if rec.country not in connections:
                    name = country_key(rec.country)+".sqlite"
                    partial = root/(name+".building")
                    partial.unlink(missing_ok=True)
                    db = sqlite3.connect(partial)
                    db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-32000;")
                    db.execute("CREATE TABLE targets(id TEXT UNIQUE,name TEXT,address TEXT,src TEXT,v TEXT,"+",".join('"'+c+'" TEXT' for c in EXACT)+")")
                    db.execute("CREATE VIRTUAL TABLE search USING fts5(n,a,g,ag,roman,s,content='',tokenize=\"unicode61 remove_diacritics 0 categories 'L* N* Co M*'\")")
                    connections[rec.country], paths[rec.country] = db, name
                db = connections[rec.country]
                v = dict(views(rec.business_name, rec.business_address, rec.country, tuple(disabled_views)))
                v["roman"] = language.romanize(v["n"])
                values = [rec.entity_id, rec.business_name, rec.business_address, src, json.dumps(v, ensure_ascii=False)] + [v[c] for c in EXACT]
                cursor = db.execute("INSERT INTO targets VALUES("+",".join("?" for _ in values)+")", values)
                db.execute("INSERT INTO search(rowid,n,a,g,ag,roman,s) VALUES(?,?,?,?,?,?,?)",
                           (cursor.lastrowid, v["folded"], v["expanded"], " ".join(chargrams(v["compact"])),
                            " ".join(chargrams(v["expanded"])), v["roman"], src.lower()))
                counts[rec.country] += 1
                if sum(counts.values()) % 100_000 == 0:
                    for connection in connections.values():
                        connection.commit()
                    print(f"Indexed {sum(counts.values()):,} targets in {time.monotonic()-started:.0f}s", flush=True)
        for country, db in connections.items():
            for col in EXACT:
                db.execute(f'CREATE INDEX by_{col} ON targets(src,"{col}")')
            db.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(search,'col')")
            db.execute("INSERT INTO search(search) VALUES('optimize')")
            db.commit()
            db.close()
            (root/(paths[country]+".building")).replace(root/paths[country])
        if not paths:
            raise ValueError("No target records")
        write_json(root/"language.json", language.payload)
        result = dict(fingerprint=fingerprint, schema=SCHEMA_VERSION, countries=paths, counts=dict(counts), disabled_views=list(disabled_views),
                      seconds=time.monotonic()-started, language_hash=digest(language.payload),
                      disk_bytes=sum((root/p).stat().st_size for p in paths.values()))
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
    view: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    channel_ranks: dict = field(default_factory=dict)


class Retriever:
    def __init__(self, index_dir, config=None):
        self.root = Path(index_dir)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        if self.manifest["schema"] != SCHEMA_VERSION:
            raise ValueError("Index schema changed; rebuild")
        self.language = Language(self.root/"language.json")
        self.config = config or RetrievalConfig()
        self.connections = {}
        self.last_channels = {}

    def db(self, country):
        if country not in self.manifest["countries"]:
            raise ValueError(f"Unknown country {country!r}: no target partition; refusing silent empty predictions")
        if country not in self.connections:
            path = self.root/self.manifest["countries"][country]
            db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
            db.execute("PRAGMA mmap_size=268435456")
            db.execute("PRAGMA query_only=ON")
            self.connections[country] = db
        return self.connections[country]

    def canonical_country(self, country):
        if country in self.manifest["countries"]: return country
        choices = [c for c in self.manifest["countries"] if c.casefold().strip() == country.casefold().strip()]
        if len(choices) == 1:
            warnings.warn(f"Country spelling fallback: {country!r} -> {choices[0]!r}", RuntimeWarning)
            return choices[0]
        raise ValueError(f"Unknown country {country!r}; no safe same-country fallback is available")

    @lru_cache(maxsize=100_000)
    def frequency(self, country, column, term):
        row = self.db(country).execute("SELECT doc FROM vocab WHERE term=? AND col=?", (term, column)).fetchone()
        return row[0] if row else 0

    @lru_cache(maxsize=50_000)
    def exact_frequency(self, country, column, value):
        if not value:
            return 0
        return sum(self.db(country).execute(f'SELECT count(*) FROM targets WHERE src=? AND "{column}"=?', (s, value)).fetchone()[0] for s in ("S2", "S3"))

    def _search(self, country, column, terms, src, k, conjunction=False):
        terms = [(self.frequency(country, column, t), t) for t in set(terms)]
        rare = [t for n, t in sorted(terms) if n > 0][:2 if conjunction else 8]
        if not rare or (conjunction and len(rare) < 2):
            return []
        expression = (" AND " if conjunction else " OR ").join('"'+t.replace('"', '""')+'"' for t in rare)
        query = f's:"{src.lower()}" AND {column}:({expression})'
        return [r[0] for r in self.db(country).execute("SELECT rowid FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?", (query, k))]

    def _exact(self, country, col, value, src, av, k):
        if not value:
            return []
        cursor = self.db(country).execute(f'SELECT rowid,v FROM targets WHERE src=? AND "{col}"=?', (src, value))
        def scored():
            for rid, encoded in cursor:
                v = json.loads(encoded)
                ns = token_sort_ratio(av["core"], v["core"])/100
                ads = token_sort_ratio(av["expanded"], v["expanded"])/100 if av["a"] and v["a"] else 0
                yield (.45*ns+.55*ads, -rid, rid)
        return [r[2] for r in heapq.nlargest(k, scored())]

    def retrieve(self, anchor):
        country = self.canonical_country(anchor.country)
        if country != anchor.country:
            anchor = Record(anchor.entity_id, anchor.business_name, anchor.business_address, country)
        db = self.db(anchor.country)
        av = dict(views(anchor.business_name, anchor.business_address, anchor.country, tuple(self.manifest.get("disabled_views", ()))))
        av["roman"] = self.language.romanize(av["n"])
        lists = {}
        def add(src, channel, ids):
            if channel not in self.config.disabled_channels:
                lists[src, channel] = list(dict.fromkeys(ids))
        for src in ("S2", "S3"):
            for col in ("n", "a", "core", "folded", "sorted", "compact", "roman", "indic"):
                if "exact_"+col not in self.config.disabled_channels:
                    add(src, "exact_"+col, self._exact(anchor.country, col, av[col], src, av, self.config.per_channel))
            for col, terms in (("n", av["folded"].split()), ("a", av["expanded"].split()), ("g", chargrams(av["compact"])), ("roman", av["roman"].split())):
                if "fts_"+col not in self.config.disabled_channels:
                    add(src, "fts_"+col, self._search(anchor.country, col, terms, src, self.config.per_channel))
            aliases = set(av["aliases"]) | set(self.language.aliases(av["n"]))
            add(src, "alias", [rid for alias in sorted(aliases) if alias != av["n"] for rid in self._search(anchor.country, "n", alias.split(), src, self.config.per_channel)])
        nonempty = [set(v) for v in lists.values() if v]
        disagreement = False
        for source in ("S2", "S3"):
            source_sets = [set(ids) for (s, _), ids in lists.items() if s == source and ids]
            disagreement |= len(source_sets) > 1 and not set.intersection(*source_sets)
        frequent = self.exact_frequency(anchor.country, "n", av["n"]) > self.config.per_channel
        adaptive = self.config.adaptive and (not av["a"] or frequent or disagreement or not nonempty)
        if adaptive:
            for src in ("S2", "S3"):
                for col in ("postcode", "street_key") + (("phonetic",) if self.config.phonetic else ()):
                    add(src, "exact_"+col, self._exact(anchor.country, col, av[col], src, av, self.config.per_channel))
                add(src, "fts_ag", self._search(anchor.country, "ag", chargrams(av["expanded"]), src, self.config.per_channel))
                add(src, "rare_pair", self._search(anchor.country, "n", av["folded"].split(), src, self.config.per_channel, True))
        budget = self.config.max_per_source if adaptive else self.config.per_source
        selected = set()
        for src in ("S2", "S3"):
            channels = [(c, ids) for (s, c), ids in lists.items() if s == src and ids]
            chosen = set()
            for rank in range(self.config.reserved_per_channel):
                for _, ids in channels:
                    if rank < len(ids) and len(chosen) < budget:
                        chosen.add(ids[rank])
            scores = defaultdict(float)
            for _, ids in channels:
                for rank, rid in enumerate(ids):
                    scores[rid] += 1/(60+rank)
            for rid in sorted(scores, key=lambda r: (-scores[r], r)):
                if len(chosen) >= budget:
                    break
                chosen.add(rid)
            selected.update(chosen)
        self.last_channels = {c: set() for c in CHANNELS}
        all_ids = sorted(set(r for ids in lists.values() for r in ids))
        fetched = {}
        for start in range(0, len(all_ids), 800):
            ids = all_ids[start:start+800]
            query = "SELECT rowid,id,name,address,v FROM targets WHERE rowid IN ("+",".join("?" for _ in ids)+")"
            fetched.update({row[0]: row[1:] for row in db.execute(query, ids)})
        for (_, channel), ids in lists.items():
            self.last_channels[channel].update(fetched[r][0] for r in ids)
        result = []
        for rid in selected:
            tid, name, address, encoded = fetched[rid]
            v = json.loads(encoded)
            ranks = {c: ids.index(rid) for (_, c), ids in lists.items() if rid in ids}
            ns = token_sort_ratio(av["core"], v["core"])/100
            ads = token_sort_ratio(av["expanded"], v["expanded"])/100 if av["a"] and v["a"] else 0
            evidence = self._evidence(anchor.country, av, v)
            result.append(Candidate(Record(tid, name, address, anchor.country), set(ranks), min(ranks.values()), .55*ns+.45*ads, v, evidence, ranks))
        return sorted(result, key=lambda c: (c.record.entity_id[:2], -sum(1/(60+r) for r in c.channel_ranks.values()), c.record.entity_id))

    def _evidence(self, country, a, b):
        total = self.manifest["counts"][country]
        def idf(col, term):
            return math.log((total+1)/(self.frequency(country, col, term)+1))+1
        left, right = set(a["folded"].split()), set(b["folded"].split())
        union = left | right
        weighted = sum(idf("n", t) for t in left & right)/max(1e-9, sum(idf("n", t) for t in union))
        def cosine(col, x, y):
            x, y = set(chargrams(x)), set(chargrams(y))
            norm = math.sqrt(sum(idf(col, t)**2 for t in x)*sum(idf(col, t)**2 for t in y))
            return sum(idf(col, t)**2 for t in x & y)/norm if norm else 0
        return dict(idf_overlap=weighted, rare_overlap=sum(self.frequency(country, "n", t) <= 5 for t in left & right),
                    name_frequency=self.exact_frequency(country, "n", b["n"]), address_frequency=self.exact_frequency(country, "a", b["a"]),
                    anchor_frequency=self.exact_frequency(country, "n", a["n"]),
                    name_tfidf=cosine("g", a["compact"], b["compact"]), address_tfidf=cosine("ag", a["expanded"], b["expanded"]),
                    roman_similarity=ratio(a["roman"], b["roman"])/100 if a["roman"] and b["roman"] else 0)

    def contains(self, country, entity_id):
        return self.db(self.canonical_country(country)).execute("SELECT 1 FROM targets WHERE id=?", (entity_id,)).fetchone() is not None

    def exact_union(self, anchor):
        found = {}
        for source in ("S2", "S3"):
            for col, value in (("n", anchor.n), ("a", anchor.a)):
                if not value: continue
                for tid, name, address in self.db(anchor.country).execute(f'SELECT id,name,address FROM targets WHERE src=? AND "{col}"=?', (source, value)):
                    if tid not in found:
                        found[tid] = Candidate(Record(tid, name, address, anchor.country), set(), 0, 0.)
                    found[tid].channels.add("exact_"+col)
        return sorted(found.values(), key=lambda c: c.record.entity_id)

    def close(self):
        for db in self.connections.values():
            db.close()
        self.connections.clear()
        self.frequency.cache_clear()
        self.exact_frequency.cache_clear()
