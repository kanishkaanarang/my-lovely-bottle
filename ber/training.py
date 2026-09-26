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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from .common import digest, entity_f05, file_hash, fold, records, truth_rows, write_json, partition_anchors
from .features import FEATURE_NAMES, features
from .retrieval import Retriever, RetrievalConfig
from .retrieval import CHANNELS
from .views import script
from .decision import decide, fit_gate
from .models import oof_scores, score_context, PairwiseRanker, SeedEnsemble

TEST_MIX = {"US": .45, "India": .55}  # supplied audit, not inferred from test labels


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
                  per_country=5000, seed=2026, max_pairs=3_000_000, split_strategy="signature"):
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    retrieval = retrieval or RetrievalConfig()
    anchors, population = sample_anchors(source1, per_country, seed)
    required = {a.entity_id for a in anchors}
    truth = {i: t for i, t in truth_rows(truth_path) if i in required}
    if truth.keys() != required:
        raise ValueError("Sampled anchors do not have exactly the required ground truth")
    splits = partition_anchors(anchors, truth, seed, split_strategy)
    manifest = json.loads((Path(index_dir)/"manifest.json").read_text())
    signature = digest({"source1": file_hash(source1), "truth": file_hash(truth_path),
                        "index": manifest["fingerprint"], "retrieval": asdict(retrieval),
                        "per_country": per_country, "seed": seed, "features": FEATURE_NAMES, "split_hash": digest(splits), "split_strategy": split_strategy})
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
    language = retriever.language.payload
    if language and (language.get("split_hash") != digest(splits) or any(splits.get(i) != "train" for i in language.get("training_anchor_ids", []))):
        raise ValueError("Language training anchors do not match this split; rebuild mapping/index for this sample")
    files, sizes = {}, Counter()
    channel_hits, incremental_hits, strata = Counter(), Counter(), Counter()
    total_links = 0
    started = time.monotonic()
    try:
        for part in ("train", "calibration", "trust"):
            files[part] = ((root/f"{part}.f32").open("wb"), (root/f"{part}.jsonl").open("w", encoding="utf-8"))
        for i, anchor in enumerate(anchors):
            part = splits[anchor.entity_id]
            candidates = retriever.retrieve(anchor)
            if sum(sizes.values()) + len(candidates) > max_pairs:
                raise ValueError("Pair budget exceeded. Reduce anchors/candidates or increase max_pairs in a NEW run.")
            ids = [c.record.entity_id for c in candidates]
            labels = truth[anchor.entity_id]
            group = digest([anchor.country, anchor.n, anchor.a])
            target_scripts = {}
            for tid in labels:
                found = retriever.db(anchor.country).execute("SELECT name FROM targets WHERE id=?", (tid,)).fetchone()
                if found is None:
                    raise ValueError("Ground-truth target absent from full training index")
                target_scripts[tid] = script(found[0])
            cumulative = set()
            hits = {}
            for channel in CHANNELS:
                found = retriever.last_channels.get(channel, set()) & labels
                hits[channel] = sorted(found)
                if part != "trust":
                    channel_hits[channel] += len(found)
                    incremental_hits[channel] += len(found-cumulative)
                cumulative.update(found)
            if part != "trust": total_links += len(labels)
            strata[f"{part}:{anchor.country}:{min(len(labels), 4)}"] += 1
            row = {"id": anchor.entity_id, "country": anchor.country, "truth": sorted(labels),
                   "candidates": ids, "offset": sizes[part], "count": len(ids), "group": group,
                   "script": script(anchor.business_name), "address_missing": not anchor.a,
                   "true_scripts": target_scripts, "channel_hits": hits,
                   "channel_agreement": max((len(c.channels) for c in candidates), default=0)}
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
              "language_hash": manifest["language_hash"],
              "language": retriever.language.payload, "disabled_views": manifest.get("disabled_views", []),
              "language_exclude_country": retriever.language.payload.get("exclude_country"),
              "language_uses_labels": bool(retriever.language.payload),
              "split_strategy": split_strategy,
              "strata": dict(strata),
              "channel_recall": {c: channel_hits[c]/max(1, total_links) for c in CHANNELS},
              "channel_report_partitions": ["train", "calibration"],
              "channel_incremental_recall": {c: incremental_hits[c]/max(1, total_links) for c in CHANNELS},
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


def evaluate(rows, probabilities, threshold, empty_gate, population=None, bundle=None):
    groups = defaultdict(list)
    tp = fp = fn = retrieved = true_count = singleton_fp = singleton_n = complete = nonempty = 0
    for row in rows:
        t = set(row["truth"])
        pr = probabilities[row["offset"]:row["offset"]+row["count"]]
        p = set(decide(bundle, row, pr) if bundle else choose(row["candidates"], pr, threshold, empty_gate))
        groups[row["country"]].append(entity_f05(t, p))
        tp += len(t & p); fp += len(p-t); fn += len(t-p)
        retrieved += len(t & set(row["candidates"])); true_count += len(t)
        singleton_n += not t; singleton_fp += not t and bool(p)
        nonempty += bool(t); complete += bool(t) and t <= set(row["candidates"])
    macro_country = {c: float(np.mean(v)) for c, v in groups.items()}
    macro = float(np.mean([v for group in groups.values() for v in group]))
    population = population or {c: len(v) for c, v in groups.items()}
    mass = sum(population.get(c, 0) for c in groups)
    weighted = sum(macro_country[c]*population.get(c, 0) for c in groups)/mass if mass else macro
    return {"macro_f05": macro, "country_weighted_macro_f05": weighted, "by_country": macro_country,
            "anchors": len(rows), "link_recall_ceiling": retrieved/max(1, true_count),
            "complete_match_set_retrieval": complete/max(1, nonempty),
            "pair_precision": tp/max(1, tp+fp), "pair_recall": tp/max(1, tp+fn),
            "singleton_false_merge_rate": singleton_fp/max(1, singleton_n),
            "true_links": true_count, "tp": tp, "fp": fp, "fn": fn}


def train(run_dir, max_iter=180, seed=2026, exclude_country=None, monotonic=True,
          calibration="platt", disabled_features=(), max_leaf_nodes=31, decoder="threshold",
          save_oof=False, rerank=False, objective="classifier", early_stopping=False, bag_seeds=()):
    """Fit/calibrate/tune without consulting the trust partition."""
    started = time.monotonic()
    root = Path(run_dir)
    manifest = json.loads((root/"pairs_manifest.json").read_text())
    if exclude_country and manifest.get("language_uses_labels") and manifest.get("language_exclude_country") != exclude_country:
        raise ValueError("Geographic stress requires language/index/pairs rebuilt WITHOUT held-country labels")
    x, y, weights, train_rows = load_part(root, "train")
    if exclude_country:
        for row in train_rows:
            if row["country"] == exclude_country:
                weights[row["offset"]:row["offset"]+row["count"]] = 0
    if len(np.unique(y[weights > 0])) != 2:
        raise ValueError("Training needs positive and negative retrieved pairs")
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in disabled_features]
    positive = {"name_ratio", "name_token_sort", "address_ratio", "idf_overlap", "name_tfidf", "address_tfidf", "roman_similarity"}
    constraints = [int(FEATURE_NAMES[i] in positive) for i in keep] if monotonic else None
    model = HistGradientBoostingClassifier(max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=20,
                                          learning_rate=.08, l2_regularization=2, early_stopping=False,
                                          random_state=seed, monotonic_cst=constraints)
    fit_x = x[:, keep] if len(keep) != len(FEATURE_NAMES) else x
    reranker = None
    if save_oof or rerank:
        if manifest.get("language_uses_labels"):
            raise ValueError("OOF requires fold-specific language rebuilding; use a no-label-language index for this experiment")
        oof = oof_scores(model, fit_x, y, weights, train_rows)
        np.save(root/"train_oof_scores.npy", oof)
        write_json(root/"oof_manifest.json", {"pairs_signature": manifest["signature"], "folds": 3,
                   "groups": "normalized signature", "scores_sha256": file_hash(root/"train_oof_scores.npy")})
        if rerank:
            reranker = LogisticRegression(C=1, max_iter=1000, random_state=seed)
            reranker.fit(score_context(train_rows, oof), y, sample_weight=weights)
    if objective == "ranking":
        if exclude_country: raise ValueError("Ranking stress experiment requires filtered group rows")
        model = PairwiseRanker(seed).fit_groups(fit_x, y, train_rows)
    elif early_stopping:
        held = np.zeros(len(y), bool)
        for row in train_rows:
            if int(row["group"][:8], 16) % 10 == 0:
                held[row["offset"]:row["offset"]+row["count"]] = True
        if len(np.unique(y[held & (weights > 0)])) != 2: raise ValueError("Early-stopping group split too small")
        model.set_params(early_stopping=True, n_iter_no_change=12)
        model.fit(fit_x[~held], y[~held], sample_weight=weights[~held], X_val=fit_x[held], y_val=y[held], sample_weight_val=weights[held])
    else:
        model.fit(fit_x, y, sample_weight=weights)
    if bag_seeds:
        if objective != "classifier" or early_stopping or rerank:
            raise ValueError("Bagging experiment uses plain classifiers; compare it separately")
        from sklearn.base import clone
        models = [model]
        for member_seed in bag_seeds:
            member = clone(model).set_params(random_state=member_seed)
            rng = np.random.default_rng(member_seed)
            groups = sorted({r["group"] for r in train_rows})
            multiplicity = Counter(rng.choice(groups, size=len(groups), replace=True))
            bag_weights = weights.copy()
            for row in train_rows:
                bag_weights[row["offset"]:row["offset"]+row["count"]] *= multiplicity[row["group"]]
            member.fit(fit_x, y, sample_weight=bag_weights)
            models.append(member)
        model = SeedEnsemble(models)
    cx, cy, cw, calibration_rows = load_part(root, "calibration")
    raw = model.predict_proba(cx[:, keep] if len(keep) != len(FEATURE_NAMES) else cx)[:, 1]
    if reranker is not None:
        raw = reranker.predict_proba(score_context(calibration_rows, raw))[:, 1]
    probability_rows, tune_rows = [], []
    mask = np.zeros(len(cy), dtype=bool)
    for row in calibration_rows:
        if exclude_country and row["country"] == exclude_country:
            continue
        if int(row["group"][:8], 16) % 2 == 0:
            mask[row["offset"]:row["offset"]+row["count"]] = True
            probability_rows.append(row)
        else:
            tune_rows.append(row)
    if len(np.unique(cy[mask])) != 2 or not tune_rows:
        raise ValueError("Calibration subsets too small; increase per_country")
    if calibration == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw[mask], cy[mask], sample_weight=cw[mask])
        probs = calibrator.predict(raw)
    elif calibration == "platt":
        calibrator = LogisticRegression(C=10, random_state=seed)
        calibrator.fit(logit(raw[mask]), cy[mask], sample_weight=cw[mask])
        probs = calibrator.predict_proba(logit(raw))[:, 1]
    else:
        raise ValueError("Unknown calibration")
    best, trials = None, []
    # Scores >1 implement the all-empty policy exactly, including probability=1.
    for threshold in np.r_[np.arange(.1, 1., .025), 1.000001]:
        for gate in sorted(set([float(threshold), .5, .7, .85, .95, 1.000001])):
            if gate < threshold:
                continue
            result = evaluate(tune_rows, probs, float(threshold), gate, TEST_MIX)
            trial = {"threshold": float(threshold), "empty_gate": gate, **result}
            trials.append(trial)
            if best is None or (trial["country_weighted_macro_f05"], threshold, gate) > (best["country_weighted_macro_f05"], best["threshold"], best["empty_gate"]):
                best = trial
    robust = max(trials, key=lambda t: (min(t["by_country"].values()), t["threshold"], t["empty_gate"]))
    policies = {}
    for country in sorted({r["country"] for r in tune_rows}):
        picked = max(trials, key=lambda t: (t["by_country"][country], t["threshold"], t["empty_gate"]))
        policies[country] = {k: picked[k] for k in ("threshold", "empty_gate")}
    bundle = {"model": model, "reranker": reranker, "calibrator": calibrator, "calibration": calibration, "keep_features": keep,
              "threshold": best["threshold"], "policies": policies,
              "robust_policy": {k: robust[k] for k in ("threshold", "empty_gate")},
              "empty_gate": best["empty_gate"], "retrieval": manifest["retrieval"],
              "features": FEATURE_NAMES, "pairs_signature": manifest["signature"],
              "language_hash": manifest["language_hash"], "language": manifest["language"],
              "disabled_views": manifest.get("disabled_views", []), "decoder": decoder,
              "missing_rate": float(np.mean([len(set(r["truth"])-set(r["candidates"])) for r in probability_rows])),
              "anchor_gate": fit_gate(probability_rows, probs), "gate_threshold": 0.,
              "seed": seed, "exclude_country": exclude_country}
    # Tune source thresholds and the learned gate only on the tuning half.
    for country, policy in policies.items():
        subset = [r for r in tune_rows if r["country"] == country]
        policy["sources"] = {}
        for source in ("S2", "S3"):
            options = []
            for threshold in sorted(set([policy["threshold"], .35, .5, .65, .8, .9])):
                policy["sources"][source] = threshold
                score = evaluate(subset, probs, 0, 0, TEST_MIX, bundle)["macro_f05"]
                options.append((score, threshold))
            policy["sources"][source] = max(options)[1]
    gate_options = []
    for gate in (0., .25, .5, .75, .9):
        bundle["gate_threshold"] = gate
        gate_options.append((evaluate(tune_rows, probs, 0, 0, TEST_MIX, bundle)["country_weighted_macro_f05"], gate))
    bundle["gate_threshold"] = max(gate_options)[1]
    robust_gate_options = []
    for gate in (0., .25, .5, .75, .9):
        bundle["robust_policy"]["gate_threshold"] = gate
        robust_bundle = {**bundle, "policies": {}}
        by_country = evaluate(tune_rows, probs, 0, 0, TEST_MIX, robust_bundle)["by_country"]
        robust_gate_options.append((min(by_country.values()), gate))
    bundle["robust_policy"]["gate_threshold"] = max(robust_gate_options)[1]
    best = {**evaluate(tune_rows, probs, 0, 0, TEST_MIX, bundle), "threshold": bundle["threshold"],
            "empty_gate": bundle["empty_gate"], "policies": policies, "gate_threshold": bundle["gate_threshold"]}
    destination = root/("model.pkl" if not exclude_country else f"model_without_{exclude_country}.pkl")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(bundle, f)
    temporary.replace(destination)
    write_json(destination.with_suffix(".metrics.json"), {"tuning": best, "trials": trials,
               "model_parameters": model.get_params(), "seed": seed, "exclude_country": exclude_country,
               "training_seconds": time.monotonic()-started,
               "calibration": calibration,
               "calibration_brier_by_country": {c: float(brier_score_loss(cy[idx], probs[idx])) for c in sorted({r["country"] for r in tune_rows})
                    if (idx := [j for r in tune_rows if r["country"] == c for j in range(r["offset"], r["offset"]+r["count"])])},
               "calibration_anchors": len(probability_rows), "tuning_anchors": len(tune_rows)})
    return destination, best


def load_model(path):
    # Pickle is executable: load only artifacts produced by this trusted pipeline.
    with Path(path).open("rb") as f:
        bundle = pickle.load(f)
    if bundle["features"] != FEATURE_NAMES:
        raise ValueError("Feature schema changed; retrain")
    return bundle


def predict_proba(bundle, x, rows=None):
    if not len(x):
        return np.empty(0)
    if bundle.get("kind") == "exact_union":
        return np.zeros(len(x))
    keep = bundle.get("keep_features", list(range(len(FEATURE_NAMES))))
    raw = bundle["model"].predict_proba(x[:, keep] if len(keep) != len(FEATURE_NAMES) else x)[:, 1]
    if bundle.get("reranker") is not None:
        if rows is None: raise ValueError("Reranker requires anchor groups")
        raw = bundle["reranker"].predict_proba(score_context(rows, raw))[:, 1]
    return bundle["calibrator"].predict(raw) if bundle.get("calibration") == "isotonic" else bundle["calibrator"].predict_proba(logit(raw))[:, 1]


def evaluate_trust(run_dir, model_path, country=None):
    bundle = load_model(model_path)
    root = Path(run_dir)
    manifest = json.loads((root/"pairs_manifest.json").read_text())
    if manifest["signature"] != bundle["pairs_signature"]:
        raise ValueError("Model and pair cache are from different runs")
    prefix = "trust" if country is None else f"stress_{country}"
    identity = file_hash(model_path)
    existing = root/f"{prefix}_metrics.json"
    if existing.exists():
        prior = json.loads(existing.read_text())
        if prior.get("model_sha256") == identity:
            return prior
        raise ValueError("Trust already evaluated for another model. Select experiments on calibration, use an explicitly new run.")
    x, _, _, rows = load_part(root, "trust")
    probs = predict_proba(bundle, x, rows)
    rows = [r for r in rows if country is None or r["country"] == country]
    if not rows:
        raise ValueError("No trust anchors in requested country")
    report = evaluate(rows, probs, bundle["threshold"], bundle["empty_gate"], TEST_MIX, bundle)
    report["model_sha256"] = identity
    oracle = []
    for row in rows:
        oracle.append(entity_f05(row["truth"], set(row["truth"]) & set(row["candidates"])))
    report["oracle_macro_f05"] = float(np.mean(oracle))
    report["candidate_count_quantiles"] = dict(zip(["median", "p95", "p99", "max"], np.quantile([r["count"] for r in rows], [.5, .95, .99, 1]).tolist()))
    report["cohorts"] = {}
    source_scores = defaultdict(list)
    for row in rows:
        pred = decide(bundle, row, probs[row["offset"]:row["offset"]+row["count"]])
        for source in ("S2", "S3"):
            source_scores[source].append(entity_f05([i for i in row["truth"] if i.startswith(source)], [i for i in pred if i.startswith(source)]))
    report["by_source_macro_f05"] = {s: float(np.mean(v)) for s, v in source_scores.items()}
    true_count = sum(len(r["truth"]) for r in rows)
    report["channel_recall"] = {c: sum(len(r["channel_hits"].get(c, [])) for r in rows)/max(1, true_count) for c in CHANNELS}
    for key in ("script", "address_missing", "cardinality", "cross_script"):
        groups = defaultdict(list)
        for r in rows:
            value = len(r["truth"]) if key == "cardinality" else any(s != r["script"] for s in r["true_scripts"].values()) if key == "cross_script" else r[key]
            groups[str(value)].append(r)
        report["cohorts"][key] = {name: evaluate(group, probs, 0, 0, TEST_MIX, bundle) for name, group in groups.items()}
    write_json(root/f"{prefix}_metrics.json", report)
    with (root/f"{prefix}_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            p = probs[row["offset"]:row["offset"]+row["count"]]
            pred = decide(bundle, row, p)
            errors = []
            if set(row["truth"])-set(row["candidates"]): errors.append("retrieval_miss")
            if set(pred)-set(row["truth"]): errors.append("false_merge")
            if row["truth"] and not pred and set(row["truth"]) & set(row["candidates"]): errors.append("gate_or_threshold_error")
            f.write(json.dumps({**row, "probabilities": p.tolist(), "predicted": pred, "errors": errors, "f05": entity_f05(row["truth"], pred)}, ensure_ascii=False)+"\n")
    return report
