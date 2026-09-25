"""Entity splits, hard negatives, separate calibration/tuning, untouched trust evaluation."""

import json
import pickle
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .common import digest, entity_f05, file_hash, fold, records, truth_rows, write_json
from .features import FEATURE_NAMES, features
from .retrieval import Retriever, RetrievalConfig


def sample_anchors(source1, per_country=5000, seed=2026):
    """Independent country reservoirs; scans every anchor, stable seed, bounded memory."""
    rng = random.Random(seed)
    samples, counts = defaultdict(list), Counter()
    for rec in records(source1):
        counts[rec.country] += 1
        if len(samples[rec.country]) < per_country:
            samples[rec.country].append(rec)
        else:
            j = rng.randrange(counts[rec.country])
            if j < per_country:
                samples[rec.country][j] = rec
    return sorted([a for group in samples.values() for a in group], key=lambda a: a.entity_id), dict(counts)


def prepare_pairs(source1, truth_path, index_dir, run_dir, retrieval=None,
                  per_country=5000, seed=2026, max_pairs=3_000_000):
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    retrieval = retrieval or RetrievalConfig()
    anchors, population = sample_anchors(source1, per_country, seed)
    required = {a.entity_id for a in anchors}
    truth = {i: t for i, t in truth_rows(truth_path) if i in required}
    if truth.keys() != required:
        raise ValueError("Sampled anchors do not have exactly the required ground truth")
    manifest = json.loads((Path(index_dir)/"manifest.json").read_text())
    signature = digest({"source1": file_hash(source1), "truth": file_hash(truth_path),
                        "index": manifest["fingerprint"], "retrieval": asdict(retrieval),
                        "per_country": per_country, "seed": seed, "features": FEATURE_NAMES})
    done = root/"pairs_manifest.json"
    if done.exists():
        result = json.loads(done.read_text())
        if result["signature"] != signature:
            raise ValueError("Pair cache configuration changed; use a new run directory")
        for name, count in result["pair_counts"].items():
            if (root/f"{name}.f32").stat().st_size != count*len(FEATURE_NAMES)*4:
                raise ValueError("Truncated feature cache")
        return result
    retriever = Retriever(index_dir, retrieval)
    files, sizes = {}, Counter()
    started = time.monotonic()
    try:
        for part in ("train", "calibration", "trust"):
            files[part] = ((root/f"{part}.f32").open("wb"), (root/f"{part}.jsonl").open("w", encoding="utf-8"))
        for i, anchor in enumerate(anchors):
            part = fold(anchor, seed)
            candidates = retriever.retrieve(anchor)
            if sum(sizes.values()) + len(candidates) > max_pairs:
                raise ValueError("Pair budget exceeded. Reduce anchors/candidates or increase max_pairs in a NEW run.")
            ids = [c.record.entity_id for c in candidates]
            labels = truth[anchor.entity_id]
            row = {"id": anchor.entity_id, "country": anchor.country, "truth": sorted(labels),
                   "candidates": ids, "offset": sizes[part], "count": len(ids)}
            features(anchor, candidates).tofile(files[part][0])
            files[part][1].write(json.dumps(row, ensure_ascii=False)+"\n")
            sizes[part] += len(ids)
            if (i+1) % 500 == 0:
                print(f"Pairs: {i+1:,}/{len(anchors):,} anchors; {sum(sizes.values()):,} candidates", flush=True)
    finally:
        for fs in files.values():
            for f in fs:
                f.close()
        retriever.close()
    result = {"signature": signature, "index_fingerprint": manifest["fingerprint"],
              "retrieval": asdict(retrieval), "seed": seed, "population": population,
              "per_country": per_country, "pair_counts": dict(sizes), "features": FEATURE_NAMES,
              "seconds": time.monotonic()-started}
    write_json(done, result)
    return result


def load_part(run_dir, part):
    root = Path(run_dir)
    with (root/f"{part}.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    count = sum(r["count"] for r in rows)
    if not rows or not count:
        raise ValueError(f"No anchors/candidates in {part}; increase the sample or check blocking")
    x = np.memmap(root/f"{part}.f32", dtype=np.float32, mode="r", shape=(count, len(FEATURE_NAMES)))
    y, weights = np.zeros(count, np.uint8), np.zeros(count, np.float32)
    for row in rows:
        sl = slice(row["offset"], row["offset"]+row["count"])
        y[sl] = [i in set(row["truth"]) for i in row["candidates"]]
        weights[sl] = 1/max(1, row["count"])
    return x, y, weights, rows


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def choose(ids, probabilities, threshold, empty_gate):
    if not len(probabilities) or max(probabilities) < empty_gate:
        return []
    return [i for i, p in zip(ids, probabilities) if p >= threshold]


def evaluate(rows, probabilities, threshold, empty_gate, population=None):
    groups = defaultdict(list)
    tp = fp = fn = retrieved = true_count = singleton_fp = singleton_n = complete = nonempty = 0
    for row in rows:
        t = set(row["truth"])
        p = set(choose(row["candidates"], probabilities[row["offset"]:row["offset"]+row["count"]], threshold, empty_gate))
        groups[row["country"]].append(entity_f05(t, p))
        tp += len(t & p); fp += len(p-t); fn += len(t-p)
        retrieved += len(t & set(row["candidates"])); true_count += len(t)
        singleton_n += not t; singleton_fp += not t and bool(p)
        nonempty += bool(t); complete += bool(t) and t <= set(row["candidates"])
    macro_country = {c: float(np.mean(v)) for c, v in groups.items()}
    macro = float(np.mean([v for group in groups.values() for v in group]))
    population = population or {c: len(v) for c, v in groups.items()}
    weighted = sum(macro_country[c]*population.get(c, 0) for c in groups)/sum(population.get(c, 0) for c in groups)
    return {"macro_f05": macro, "country_weighted_macro_f05": weighted, "by_country": macro_country,
            "anchors": len(rows), "link_recall_ceiling": retrieved/max(1, true_count),
            "complete_match_set_retrieval": complete/max(1, nonempty),
            "pair_precision": tp/max(1, tp+fp), "pair_recall": tp/max(1, tp+fn),
            "singleton_false_merge_rate": singleton_fp/max(1, singleton_n),
            "true_links": true_count, "tp": tp, "fp": fp, "fn": fn}


def train(run_dir, max_iter=180, seed=2026, exclude_country=None):
    """Fit/calibrate/tune without consulting the trust partition."""
    root = Path(run_dir)
    manifest = json.loads((root/"pairs_manifest.json").read_text())
    x, y, weights, train_rows = load_part(root, "train")
    if exclude_country:
        for row in train_rows:
            if row["country"] == exclude_country:
                weights[row["offset"]:row["offset"]+row["count"]] = 0
    if len(np.unique(y[weights > 0])) != 2:
        raise ValueError("Training needs positive and negative retrieved pairs")
    model = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=31, min_samples_leaf=20,
                                          learning_rate=.08, l2_regularization=2, early_stopping=False,
                                          random_state=seed)
    model.fit(x, y, sample_weight=weights)
    cx, cy, cw, calibration_rows = load_part(root, "calibration")
    raw = model.predict_proba(cx)[:, 1]
    probability_rows, tune_rows = [], []
    mask = np.zeros(len(cy), dtype=bool)
    for row in calibration_rows:
        if exclude_country and row["country"] == exclude_country:
            continue
        if int(digest(row["id"])[:8], 16) % 2 == 0:
            mask[row["offset"]:row["offset"]+row["count"]] = True
            probability_rows.append(row)
        else:
            tune_rows.append(row)
    if len(np.unique(cy[mask])) != 2 or not tune_rows:
        raise ValueError("Calibration subsets too small; increase per_country")
    calibrator = LogisticRegression(C=10, random_state=seed)
    calibrator.fit(logit(raw[mask]), cy[mask], sample_weight=cw[mask])
    probs = calibrator.predict_proba(logit(raw))[:, 1]
    best, trials = None, []
    # Scores >1 implement the all-empty policy exactly, including probability=1.
    for threshold in np.r_[np.arange(.1, 1., .025), 1.000001]:
        for gate in sorted(set([float(threshold), .5, .7, .85, .95, 1.000001])):
            if gate < threshold:
                continue
            result = evaluate(tune_rows, probs, float(threshold), gate, manifest["population"])
            trial = {"threshold": float(threshold), "empty_gate": gate, **result}
            trials.append(trial)
            if best is None or (trial["country_weighted_macro_f05"], threshold, gate) > (best["country_weighted_macro_f05"], best["threshold"], best["empty_gate"]):
                best = trial
    bundle = {"model": model, "calibrator": calibrator, "threshold": best["threshold"],
              "empty_gate": best["empty_gate"], "retrieval": manifest["retrieval"],
              "features": FEATURE_NAMES, "pairs_signature": manifest["signature"],
              "seed": seed, "exclude_country": exclude_country}
    destination = root/("model.pkl" if not exclude_country else f"model_without_{exclude_country}.pkl")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(bundle, f)
    temporary.replace(destination)
    write_json(destination.with_suffix(".metrics.json"), {"tuning": best, "trials": trials,
               "model_parameters": model.get_params(), "seed": seed, "exclude_country": exclude_country,
               "calibration_anchors": len(probability_rows), "tuning_anchors": len(tune_rows)})
    return destination, best


def load_model(path):
    # Pickle is executable: load only artifacts produced by this trusted pipeline.
    with Path(path).open("rb") as f:
        bundle = pickle.load(f)
    if bundle["features"] != FEATURE_NAMES:
        raise ValueError("Feature schema changed; retrain")
    return bundle


def predict_proba(bundle, x):
    if not len(x):
        return np.empty(0)
    raw = bundle["model"].predict_proba(x)[:, 1]
    return bundle["calibrator"].predict_proba(logit(raw))[:, 1]


def evaluate_trust(run_dir, model_path, country=None):
    bundle = load_model(model_path)
    root = Path(run_dir)
    manifest = json.loads((root/"pairs_manifest.json").read_text())
    if manifest["signature"] != bundle["pairs_signature"]:
        raise ValueError("Model and pair cache are from different runs")
    x, _, _, rows = load_part(root, "trust")
    probs = predict_proba(bundle, x)
    rows = [r for r in rows if country is None or r["country"] == country]
    if not rows:
        raise ValueError("No trust anchors in requested country")
    report = evaluate(rows, probs, bundle["threshold"], bundle["empty_gate"], manifest["population"])
    oracle = []
    for row in rows:
        oracle.append(entity_f05(row["truth"], set(row["truth"]) & set(row["candidates"])))
    report["oracle_macro_f05"] = float(np.mean(oracle))
    report["candidate_count_quantiles"] = dict(zip(["median", "p95", "p99", "max"], np.quantile([r["count"] for r in rows], [.5, .95, .99, 1]).tolist()))
    prefix = "trust" if country is None else f"stress_{country}"
    write_json(root/f"{prefix}_metrics.json", report)
    with (root/f"{prefix}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            p = probs[row["offset"]:row["offset"]+row["count"]]
            pred = choose(row["candidates"], p, bundle["threshold"], bundle["empty_gate"])
            f.write(json.dumps({**row, "probabilities": p.tolist(), "predicted": pred, "f05": entity_f05(row["truth"], pred)}, ensure_ascii=False)+"\n")
    return report
