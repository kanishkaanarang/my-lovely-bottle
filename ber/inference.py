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

from .common import digest, file_hash, records, write_json
from .features import features
from .retrieval import Retriever, RetrievalConfig
from .training import choose, load_model, predict_proba

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def predict(source1, index_dir, model_path, output_dir, batch_size=500, limit=None):
    """One independent atomic directory per batch. Resume requires identical inputs."""
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("Invalid batch_size/limit")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bundle = load_model(model_path)
    retriever = Retriever(index_dir, RetrievalConfig(**bundle["retrieval"]))
    signature = digest({"model": file_hash(model_path), "source1": file_hash(source1),
                        "index": retriever.manifest["fingerprint"], "batch_size": batch_size,
                        "limit": limit, "retrieval": bundle["retrieval"]})
    state_path = root/"inference_manifest.json"
    if state_path.exists() and json.loads(state_path.read_text())["signature"] != signature:
        raise ValueError("Prediction configuration changed; use a new output directory")
    write_json(state_path, {"signature": signature, "limit": limit, "complete": False})
    shards = root/"shards"
    shards.mkdir(exist_ok=True)
    iterator = iter(records(source1))
    if limit is not None:
        iterator = itertools.islice(iterator, limit)
    started, batches, total, pair_count = time.monotonic(), [], 0, 0
    try:
        for number in itertools.count():
            anchors = list(itertools.islice(iterator, batch_size))
            if not anchors:
                break
            folder = shards/f"{number:07d}"
            ids_hash = digest([a.entity_id for a in anchors])
            if folder.exists():
                meta = json.loads((folder/"complete.json").read_text())
                if meta["ids_hash"] != ids_hash or meta["signature"] != signature:
                    raise ValueError("Mismatched inference shard")
                for name in ("matching.tsv", "candidates.tsv"):
                    if file_hash(folder/name) != meta["hashes"][name]:
                        raise ValueError("Corrupt completed shard")
            else:
                temporary = shards/f"{number:07d}.partial"
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir()
                candidate_groups = [retriever.retrieve(a) for a in anchors]
                matrices = [features(a, cs) for a, cs in zip(anchors, candidate_groups)]
                combined = np.concatenate(matrices)
                probs = predict_proba(bundle, combined)
                offset = 0
                with (temporary/"matching.tsv").open("w", encoding="utf-8", newline="") as mf, (temporary/"candidates.tsv").open("w", encoding="utf-8", newline="") as cf:
                    mw, cw = csv.writer(mf, delimiter="\t", lineterminator="\n"), csv.writer(cf, delimiter="\t", lineterminator="\n")
                    for anchor, cs in zip(anchors, candidate_groups):
                        ids = [c.record.entity_id for c in cs]
                        pred = choose(ids, probs[offset:offset+len(ids)], bundle["threshold"], bundle["empty_gate"])
                        # This is the exact final set supplied to the model, not the raw FTS output.
                        cw.writerow([anchor.entity_id, ",".join(ids)])
                        mw.writerow([anchor.entity_id, ",".join(pred)])
                        offset += len(ids)
                meta = {"signature": signature, "ids_hash": ids_hash, "anchors": len(anchors),
                        "pairs": len(combined), "hashes": {n: file_hash(temporary/n) for n in ("matching.tsv", "candidates.tsv")}}
                write_json(temporary/"complete.json", meta)
                temporary.replace(folder)
            batches.append(folder)
            total += len(anchors); pair_count += meta["pairs"]
            print(f"Prediction: {total:,} anchors / {pair_count:,} pairs ({time.monotonic()-started:.1f}s this call)", flush=True)
        for output_name, shard_name, header in (("matching_results.tsv", "matching.tsv", MATCH_HEADER),
                                                 ("candidate_pairs.tsv", "candidates.tsv", CANDIDATE_HEADER)):
            temporary = root/(output_name+".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as out:
                out.write("\t".join(header)+"\n")
                for folder in batches:
                    with (folder/shard_name).open(encoding="utf-8") as f:
                        shutil.copyfileobj(f, out)
            temporary.replace(root/output_name)
        result = {"signature": signature, "complete": True, "limit": limit, "anchors": total,
                  "pairs": pair_count, "seconds_this_call": time.monotonic()-started,
                  "source1_sha256": file_hash(source1), "index_fingerprint": retriever.manifest["fingerprint"],
                  "model_sha256": file_hash(model_path), "batch_size": batch_size}
        write_json(state_path, result)
        return result
    finally:
        retriever.close()


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
                mr, cr = csv.reader(mf, delimiter="\t"), csv.reader(cf, delimiter="\t")
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


def package_submission(repo_dir, output_dir, model_path, destination):
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
        for name in ("requirements.txt", "LICENSE", "SUBMISSION_README.md"):
            z.write(repo/name, "code/business_entity_resolution/"+("README.md" if name == "SUBMISSION_README.md" else name))
        z.write(documentation, "Documentation_template.md")
        z.writestr("code/business_entity_resolution/inference_config.json", json.dumps({"batch_size": state["batch_size"], "model_sha256": state["model_sha256"]}, indent=2))
    temporary.replace(destination)
    return str(destination)
