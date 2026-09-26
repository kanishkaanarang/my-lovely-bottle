"""Small training-only alias and character transducer; no external corpus/library."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, file_hash, fold_latin, normalize, records, truth_rows, write_json, partition_anchors
from .views import script


def _align(source, target, table):
    # Monotone Viterbi alignment: each source character emits 0..3 Latin letters.
    states = {0: (0., [])}
    for i, char in enumerate(source):
        nxt = {}
        for j, (score, path) in states.items():
            for width in range(4):
                end = j + width
                if end > len(target) or len(target)-end > 3*(len(source)-i-1):
                    continue
                piece = target[j:end]
                probability = table.get(char, {}).get(piece, .001)
                value = score + math.log(probability) - .12*abs(end/ max(1, len(target))-(i+1)/len(source))
                if end not in nxt or value > nxt[end][0]:
                    nxt[end] = value, path+[(char, piece)]
        states = nxt
    return states.get(len(target), (0, []))[1]


def mine_language(source1, truth_path, targets, destination, per_country=5000, seed=2026, exclude_country=None, mining_limit=20_000, split_strategy="signature"):
    from .training import sample_anchors
    anchors, _ = sample_anchors(source1, per_country, seed)
    wanted = {a.entity_id for a in anchors}
    labels = {i: ids for i, ids in truth_rows(truth_path) if i in wanted}
    splits = partition_anchors(anchors, labels, seed, split_strategy)
    selected = [a for a in anchors if splits[a.entity_id] == "train" and a.country != exclude_country]
    counts = Counter(); anchors = {}
    for a in selected:
        if counts[a.country] < mining_limit:
            anchors[a.entity_id] = a
            counts[a.country] += 1
    labels = {i: ids for i, ids in labels.items() if i in anchors}
    reverse = defaultdict(list)
    for i, ids in labels.items():
        for target in ids:
            reverse[target].append(i)
    aliases, token_pairs = defaultdict(Counter), Counter()
    for path in targets:
        for number, target in enumerate(records(path), 1):
            if number % 1_000_000 == 0:
                print(f"Language mining: {Path(path).name}, {number:,} target rows scanned", flush=True)
            for aid in reverse.get(target.entity_id, ()):
                a = anchors[aid]
                left, right = fold_latin(a.n), fold_latin(target.n)
                if not left or not right:
                    continue
                if script(left) == "LATIN" and script(right) != "LATIN":
                    left, right = right, left
                if script(left) != "LATIN" and script(right) == "LATIN":
                    aliases[left][right] += 1
                    # Equal token-count alignment is a conservative training hypothesis.
                    if len(left.split()) == len(right.split()):
                        for x, y in zip(left.split(), right.split()):
                            if script(x) != "LATIN" and 1 <= len(x) <= 24 and 1 <= len(y) <= 30:
                                token_pairs[x, y] += 1
                elif left != right:
                    aliases[left][right] += 1
                    aliases[right][left] += 1
    table = {}
    for _ in range(5):
        counts = defaultdict(Counter)
        for (left, right), weight in token_pairs.items():
            for char, piece in _align(left, right, table):
                counts[char][piece] += weight
        table = {c: {p: n/sum(values.values()) for p, n in values.items()} for c, values in counts.items()}
    result = dict(version=1, seed=seed, exclude_country=exclude_country,
                  training_anchor_hash=digest(sorted(anchors)), training_anchor_ids=sorted(anchors), split_hash=digest(splits), per_country=per_country, split_strategy=split_strategy,
                  source_hash=file_hash(source1), truth_hash=file_hash(truth_path),
                  aliases={k: [p for p, _ in values.most_common(3)] for k, values in aliases.items()},
                  characters={c: max(values, key=values.get) for c, values in table.items()},
                  token_pairs=len(token_pairs), provenance="training fold only; no calibration/trust labels")
    write_json(destination, result)
    return result


class Language:
    def __init__(self, path=None, payload=None):
        self.payload = payload or (json.loads(Path(path).read_text()) if path else {})

    def romanize(self, text):
        text = fold_latin(normalize(text))
        if script(text) == "LATIN":
            return text
        known = self.payload.get("aliases", {}).get(text, [])
        latin = next((a for a in known if script(a) == "LATIN"), None)
        if latin:
            return latin
        mapping = self.payload.get("characters", {})
        return "".join(c if c.isascii() else mapping.get(c, c) for c in text)

    def aliases(self, text):
        return self.payload.get("aliases", {}).get(fold_latin(normalize(text)), [])
