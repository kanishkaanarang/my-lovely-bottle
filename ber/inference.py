"""Restartable inference, strict validation, and reproducible submission packaging."""

import csv
import itertools
import json
import shutil
import sqlite3
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .common import TSV, digest, file_hash, records, write_json
from .decision import decide
from .features import features, FEATURE_NAMES
from .retrieval import Retriever, RetrievalConfig
from .training import choose, load_model, predict_proba

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


_WORKER = None


def _worker_init(index_dir, model_path, threads):
    global _WORKER
    from threadpoolctl import threadpool_limits
    limiter = threadpool_limits(limits=threads)
    bundle = load_model(model_path)
    retriever = Retriever(index_dir, RetrievalConfig(**bundle["retrieval"]))
    if retriever.manifest["language_hash"] != bundle["language_hash"]:
        raise ValueError("Index language mapping differs from trained model")
    _WORKER = retriever, bundle, limiter


def _batch(task):
    number, anchors, shards, signature = task
    retriever, bundle, _ = _WORKER
    folder = Path(shards)/f"{number:07d}"
    ids_hash = digest([a.entity_id for a in anchors])
    if folder.exists():
        meta = json.loads((folder/"complete.json").read_text())
        if meta["ids_hash"] != ids_hash or meta["signature"] != signature:
            raise ValueError("Mismatched inference shard")
        for name in ("matching.tsv", "candidates.tsv"):
            if file_hash(folder/name) != meta["hashes"][name]:
                raise ValueError("Corrupt completed shard")
        meta["reused"] = True
        return meta
    temporary = Path(shards)/f"{number:07d}.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    candidate_groups = [retriever.exact_union(a) if bundle.get("kind") == "exact_union" else retriever.retrieve(a) for a in anchors]
    matrices = [np.zeros((len(cs), len(FEATURE_NAMES)), np.float32) if bundle.get("kind") == "exact_union" else features(a, cs) for a, cs in zip(anchors, candidate_groups)]
    combined = np.concatenate(matrices)
    score_rows, score_offset = [], 0
    for a, cs in zip(anchors, candidate_groups):
        score_rows.append({"offset": score_offset, "count": len(cs), "candidates": [c.record.entity_id for c in cs]})
        score_offset += len(cs)
    probs = predict_proba(bundle, combined, score_rows)
    offset = 0
    drift = {}
    with (temporary/"matching.tsv").open("w", encoding="utf-8", newline="") as mf, (temporary/"candidates.tsv").open("w", encoding="utf-8", newline="") as cf:
        mw, cw = csv.writer(mf, **TSV, lineterminator="\n"), csv.writer(cf, **TSV, lineterminator="\n")
        for anchor, cs in zip(anchors, candidate_groups):
            ids = [c.record.entity_id for c in cs]
            p = probs[offset:offset+len(ids)]
            row = {"candidates": ids, "country": anchor.country, "address_missing": not anchor.a,
                   "channel_agreement": max((len(c.channels) for c in cs), default=0)}
            pred = ([c.record.entity_id for c in cs if c.channels & {"exact_n", "exact_a"}]
                    if bundle.get("kind") == "exact_union" else decide(bundle, row, p))
            cw.writerow([anchor.entity_id, ",".join(ids)])
            mw.writerow([anchor.entity_id, ",".join(pred)])
            stats = drift.setdefault(anchor.country, {"anchors": 0, "empty": 0, "candidate_hist": [0]*8, "top_score_hist": [0]*10})
            stats["anchors"] += 1
            stats["empty"] += not pred
            stats["candidate_hist"][min(7, int(np.log2(len(ids)+1)))] += 1
            stats["top_score_hist"][min(9, int(max(p, default=0)*10))] += 1
            offset += len(ids)
    meta = {"signature": signature, "ids_hash": ids_hash, "anchors": len(anchors), "drift": drift,
            "pairs": len(combined), "hashes": {n: file_hash(temporary/n) for n in ("matching.tsv", "candidates.tsv")}}
    write_json(temporary/"complete.json", meta)
    temporary.replace(folder)
    return meta


def predict(source1, index_dir, model_path, output_dir, batch_size=500, limit=None, workers=1):
    """Bounded process pool; each worker owns a read-only SQLite connection."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import platform
    from .training import sample_anchors
    if batch_size < 1 or workers < 1 or (limit is not None and limit < 1):
        raise ValueError("Invalid batch_size/workers/limit")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bundle = load_model(model_path)
    manifest = json.loads((Path(index_dir)/"manifest.json").read_text())
    if manifest["language_hash"] != bundle["language_hash"]:
        raise ValueError("Index language mapping differs from trained model")
    if manifest.get("disabled_views", []) != bundle.get("disabled_views", []):
        raise ValueError("Index normalization differs from trained model")
    signature = digest({"model": file_hash(model_path), "source1": file_hash(source1),
                        "index": manifest["fingerprint"], "batch_size": batch_size,
                        "limit": limit, "retrieval": bundle["retrieval"], "sampling": "country_reservoir_v3"})
    state_path = root/"inference_manifest.json"
    if state_path.exists() and json.loads(state_path.read_text())["signature"] != signature:
        raise ValueError("Prediction configuration changed; use a new output directory")
    write_json(state_path, {"signature": signature, "limit": limit, "complete": False})
    shards = root/"shards"
    shards.mkdir(exist_ok=True)
    sampling_started = time.monotonic()
    if limit is None:
        iterator = iter(records(source1))
    else:
        sampled, population = sample_anchors(source1, per_country=limit, seed=2026)
        # Country reservoirs, then population-weighted mixing. Shuffle each pool
        # before truncation so sorted entity IDs do not bias the benchmark.
        from collections import defaultdict
        import random
        groups = defaultdict(list)
        for a in sampled:
            groups[a.country].append(a)
        rng = random.Random(2026)
        for group in groups.values(): rng.shuffle(group)
        chosen = [group.pop() for group in groups.values() if group][:limit]
        while len(chosen) < min(limit, sum(population.values())):
            available = [c for c in groups if groups[c]]
            if not available: break
            country = rng.choices(available, weights=[population[c] for c in available])[0]
            chosen.append(groups[country].pop())
        rng.shuffle(chosen)
        iterator = iter(chosen)
    sampling_seconds = time.monotonic()-sampling_started
    def tasks():
        for number in itertools.count():
            anchors = list(itertools.islice(iterator, batch_size))
            if not anchors:
                break
            yield number, anchors, str(shards), signature
    started, batches, total, pair_count, drift = time.monotonic(), [], 0, 0, {}
    reused_batches = 0
    executor = None
    try:
        if workers > 1:
            executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                         initializer=_worker_init, initargs=(str(index_dir), str(model_path), 1))
        else:
            _worker_init(index_dir, model_path, 1)
        pending = iter(tasks())
        while True:
            window = list(itertools.islice(pending, max(1, workers*2)))
            if not window:
                break
            results = executor.map(_batch, window) if executor else map(_batch, window)
            for task, meta in zip(window, results):
                batches.append(shards/f"{task[0]:07d}")
                reused_batches += bool(meta.get("reused"))
                total += meta["anchors"]
                pair_count += meta["pairs"]
                for country, counts in meta["drift"].items():
                    target = drift.setdefault(country, {"anchors": 0, "empty": 0, "candidate_hist": [0]*8, "top_score_hist": [0]*10})
                    for key, value in counts.items():
                        target[key] = [a+b for a, b in zip(target[key], value)] if isinstance(value, list) else target[key]+value
            print(f"Prediction: {total:,} anchors / {pair_count:,} pairs ({time.monotonic()-started:.1f}s)", flush=True)
        for output_name, shard_name, header in (("matching_results.tsv", "matching.tsv", MATCH_HEADER),
                                               ("candidate_pairs.tsv", "candidates.tsv", CANDIDATE_HEADER)):
            temporary = root/(output_name+".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as out:
                out.write("\t".join(header)+"\n")
                for folder in batches:
                    with (folder/shard_name).open(encoding="utf-8") as f:
                        shutil.copyfileobj(f, out)
            temporary.replace(root/output_name)
        result = {"signature": signature, "complete": True, "limit": limit, "anchors": total, "drift": drift,
                  "pairs": pair_count, "seconds_this_call": time.monotonic()-started, "sampling_seconds": sampling_seconds,
                  "source1_sha256": file_hash(source1), "index_fingerprint": manifest["fingerprint"],
                  "model_sha256": file_hash(model_path), "batch_size": batch_size, "workers": workers,
                  "reused_batches": reused_batches, "valid_fresh_benchmark": limit is not None and reused_batches == 0,
                  "hardware": {"platform": platform.platform(), "cpu_count": __import__("os").cpu_count()}}
        write_json(state_path, result)
        return result
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
        elif _WORKER is not None:
            _WORKER[0].close()
            _WORKER[2].restore_original_limits()


def validate(source1, index_dir, output_dir):
    """Strict, streaming check against actual input and index IDs, including candidate subset."""
    root = Path(output_dir)
    retriever = Retriever(index_dir)
    state = json.loads((root/"inference_manifest.json").read_text())
    if not state["complete"] or state["limit"] is not None:
        raise ValueError("Benchmark/partial output cannot be packaged as a submission")
    if state["source1_sha256"] != file_hash(source1) or state["index_fingerprint"] != retriever.manifest["fingerprint"]:
        raise ValueError("Validation inputs differ from prediction inputs")
    count = pairs = matches = 0
    with tempfile.TemporaryDirectory(prefix="ber-validate-", dir=root) as tmp:
        seen = sqlite3.connect(Path(tmp)/"ids.sqlite")
        seen.execute("CREATE TABLE seen(id TEXT PRIMARY KEY)")
        try:
            with (root/"matching_results.tsv").open(encoding="utf-8", newline="") as mf, (root/"candidate_pairs.tsv").open(encoding="utf-8", newline="") as cf:
                mr, cr = csv.reader(mf, **TSV), csv.reader(cf, **TSV)
                if next(mr, None) != MATCH_HEADER or next(cr, None) != CANDIDATE_HEADER:
                    raise ValueError("Incorrect output headers")
                for anchor, m, c in itertools.zip_longest(records(source1), mr, cr):
                    if anchor is None or m is None or c is None or len(m) != 2 or len(c) != 2:
                        raise ValueError("Output row count/schema mismatch")
                    if anchor.entity_id != m[0] or anchor.entity_id != c[0]:
                        raise ValueError("Missing/reordered/incorrect anchor ID")
                    seen.execute("INSERT INTO seen VALUES(?)", (anchor.entity_id,))
                    mids, cids = (m[1].split(",") if m[1] else []), (c[1].split(",") if c[1] else [])
                    if len(mids) != len(set(mids)) or len(cids) != len(set(cids)):
                        raise ValueError("Duplicate target ID within a list")
                    if not set(mids) <= set(cids):
                        raise ValueError("Prediction is absent from model candidate set")
                    if any(not x.startswith(("S2-", "S3-")) or not retriever.contains(anchor.country, x) for x in cids):
                        raise ValueError("Unknown target ID or cross-country candidate")
                    count += 1; pairs += len(cids); matches += len(mids)
            result = {"status": "PASS", "anchors": count, "candidates": pairs, "matches": matches,
                      "inference_signature": state["signature"],
                      "files": {n: file_hash(root/n) for n in ("matching_results.tsv", "candidate_pairs.tsv")}}
            write_json(root/"validation.json", result)
            return result
        finally:
            seen.close()
            retriever.close()


def package_submission(repo_dir, output_dir, model_path, destination, require_official=False):
    repo, output = Path(repo_dir), Path(output_dir)
    validation = json.loads((output/"validation.json").read_text())
    if validation["status"] != "PASS":
        raise ValueError("Run strict validation first")
    state = json.loads((output/"inference_manifest.json").read_text())
    if not state["complete"] or state["limit"] is not None or state["signature"] != validation["inference_signature"]:
        raise ValueError("Inference is incomplete or does not match the validation result")
    if state["model_sha256"] != file_hash(model_path):
        raise ValueError("Model differs from the artifact used for inference")
    for n, h in validation["files"].items():
        if file_hash(output/n) != h:
            raise ValueError("Output changed after validation")
    if require_official:
        official = json.loads((output/"official_validation.json").read_text())
        if official["returncode"] or official.get("output_hashes") != validation["files"]:
            raise ValueError("Official validator must pass on these exact output files")
    documentation = repo/"Documentation_template.md"
    if "[FILL" in documentation.read_text():
        raise ValueError("Complete Documentation_template.md before packaging")
    temporary = Path(str(destination)+".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(output/name, "output/"+name)
        for file in sorted((repo/"ber").glob("*.py")):
            z.write(file, "code/business_entity_resolution/src/ber/"+file.name)
        z.write(model_path, "code/business_entity_resolution/model.pkl")
        bundle = load_model(model_path)
        z.writestr("code/business_entity_resolution/language.json", json.dumps(bundle.get("language", {}), ensure_ascii=False))
        for name in ("requirements.txt", "LICENSE", "SUBMISSION_README.md"):
            z.write(repo/name, "code/business_entity_resolution/"+("README.md" if name == "SUBMISSION_README.md" else name))
        z.write(documentation, "Documentation_template.md")
        z.writestr("code/business_entity_resolution/inference_config.json", json.dumps({"batch_size": state["batch_size"], "model_sha256": state["model_sha256"],
                    "disabled_views": bundle.get("disabled_views", []), "language_hash": bundle["language_hash"]}, indent=2))
    temporary.replace(destination)
    return str(destination)


def fallback(source1, index_dir, output_dir, workers=1):
    """Uncapped normalized exact-name/address union. Low precision emergency baseline."""
    import pickle
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(index_dir)/"manifest.json").read_text())
    bundle = {"kind": "exact_union", "features": FEATURE_NAMES,
              "retrieval": asdict(RetrievalConfig()), "language_hash": manifest["language_hash"],
              "language": json.loads((Path(index_dir)/"language.json").read_text()), "disabled_views": manifest.get("disabled_views", [])}
    model = root/"fallback.pkl"
    with model.open("wb") as f: pickle.dump(bundle, f)
    return predict(source1, index_dir, model, root, workers=workers)
