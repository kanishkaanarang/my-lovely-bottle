"""Training-only experiment tools. Never select configurations on trust scores."""
import csv
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from .common import TSV, digest, entity_f05, file_hash, fold, records, truth_rows, write_json, partition_anchors
from .retrieval import CHANNELS, Retriever, RetrievalConfig
from .training import TEST_MIX, evaluate, load_model, load_part, predict_proba, sample_anchors
from .views import views


def retrieval_sweep(source1, truth_path, index_dir, destination, budgets=(40, 80, 160), per_country=1000, seed=2026):
    anchors, _ = sample_anchors(source1, per_country, seed)
    wanted = {a.entity_id for a in anchors}
    labels = {i: t for i, t in truth_rows(truth_path) if i in wanted}
    split = partition_anchors(anchors, labels, seed)
    language = json.loads((Path(index_dir)/"language.json").read_text())
    if language and language.get("split_hash") != digest(split):
        raise ValueError("Sweep sample must match language split; pass the original per_country setting")
    anchors = [a for a in anchors if split[a.entity_id] == "calibration"]
    reports = []
    for budget in budgets:
        configs = [("all", RetrievalConfig(per_source=budget, max_per_source=budget))]
        configs += [("without_"+name, replace(configs[0][1], disabled_channels=channels)) for name, channels in {
            "suffix": ("exact_core",), "accents": ("exact_folded",), "reordering": ("exact_sorted",),
            "address": ("fts_a", "fts_ag", "exact_street_key"), "script": ("exact_roman", "fts_roman", "alias")}.items()]
        for name, config in configs:
            r = Retriever(index_dir, config)
            started = time.monotonic()
            hits = total = complete = pairs = 0
            channel = Counter(); incremental = Counter()
            try:
                for a in anchors:
                    cs = r.retrieve(a); ids = {c.record.entity_id for c in cs}; truth = labels[a.entity_id]
                    hits += len(truth & ids); total += len(truth); complete += truth <= ids; pairs += len(ids)
                    seen = set()
                    for c in CHANNELS:
                        found = r.last_channels.get(c, set()) & truth
                        channel[c] += len(found); incremental[c] += len(found-seen); seen.update(found)
            finally:
                r.close()
            reports.append(dict(name=name, budget=budget, anchors=len(anchors), recall=hits/max(total, 1),
                                complete=complete/max(len(anchors), 1), pairs=pairs, seconds=time.monotonic()-started,
                                channel_recall={k: v/max(total, 1) for k, v in channel.items()},
                                incremental_recall={k: v/max(total, 1) for k, v in incremental.items()}))
    write_json(destination, reports)
    return reports


def paired_comparison(left_path, right_path, destination, repeats=1000, seed=2026):
    def read(path):
        with Path(path).open() as f:
            return {r["id"]: r for line in f if (r := json.loads(line))}
    left, right = read(left_path), read(right_path)
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical anchor IDs")
    groups = defaultdict(list); repaired = harmed = 0
    for key, row in left.items():
        if row["truth"] != right[key]["truth"]:
            raise ValueError("Truth differs")
        delta = right[key]["f05"]-row["f05"]
        groups[row.get("group", key)].append(delta)
        repaired += delta > 0; harmed += delta < 0
    blocks = list(groups.values()); rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        means.append(sum(sum(blocks[i]) for i in selected)/sum(len(blocks[i]) for i in selected))
    result = dict(delta=np.mean([x for b in blocks for x in b]).item(),
                  ci95=np.quantile(means, [.025, .975]).tolist(), repaired=repaired, harmed=harmed,
                  unchanged=len(left)-repaired-harmed, clusters=len(blocks))
    write_json(destination, result)
    return result


def register_run(run_dir, registry, leaderboard=None, submission=None):
    root, registry = Path(run_dir), Path(registry)
    registry.mkdir(parents=True, exist_ok=True)
    entry = {"run": str(root.resolve()), "recorded_at": time.time(), "leaderboard": leaderboard,
             "submission": submission, "hardware": platform.platform()}
    for name in ("pairs_manifest.json", "model.metrics.json", "trust_metrics.json"):
        if (root/name).exists(): entry[name] = json.loads((root/name).read_text())
    entry["model_sha256"] = file_hash(root/"model.pkl") if (root/"model.pkl").exists() else None
    write_json(registry/(digest(str(root.resolve()))[:16]+".json"), entry)
    with (registry/"comparison.tsv").open("w", newline="") as f:
        writer = csv.writer(f, **TSV, lineterminator="\n")
        writer.writerow(["run", "calibration_f05", "trust_f05", "public_leaderboard", "submission"])
        for path in sorted(registry.glob("*.json")):
            r = json.loads(path.read_text())
            writer.writerow([r["run"], r.get("model.metrics.json", {}).get("tuning", {}).get("country_weighted_macro_f05", ""),
                             r.get("trust_metrics.json", {}).get("macro_f05", ""), r["leaderboard"], r["submission"]])
    return entry


def label_noise(source1, truth_path, destination, per_country=5000):
    anchors, _ = sample_anchors(source1, per_country)
    wanted = {a.entity_id for a in anchors}
    truth = {i: sorted(t) for i, t in truth_rows(truth_path) if i in wanted}
    groups = defaultdict(list)
    for a in anchors:
        v = views(a.business_name, a.business_address, a.country)
        groups[digest([a.country, v["compact"], v["expanded"]])].append(a.entity_id)
    conflicts = [ids for ids in groups.values() if len({tuple(truth[i]) for i in ids}) > 1]
    report = {"sampled_anchors": len(anchors), "ambiguous_signature_groups": conflicts,
              "interpretation": "different ID sets in identical text groups; possible duplicate targets, not proven label error"}
    write_json(destination, report)
    return report


def feature_audit(run_dir, model_path, destination, max_anchors=1000):
    bundle = load_model(model_path)
    x, y, weights, rows = load_part(run_dir, "calibration")
    rows = [r for r in rows if int(r["group"][:8], 16) % 2][:max_anchors]
    indices = [j for r in rows for j in range(r["offset"], r["offset"]+r["count"])]
    sample, labels = np.asarray(x[indices]), y[indices]
    offset = 0; local_rows = []
    for row in rows:
        local_rows.append({**row, "offset": offset}); offset += row["count"]
    baseline = log_loss(labels, predict_proba(bundle, sample, local_rows), labels=[0, 1])
    rng = np.random.default_rng(2026); result = {}
    for j, name in enumerate(bundle["features"]):
        copy = sample.copy(); rng.shuffle(copy[:, j])
        result[name] = log_loss(labels, predict_proba(bundle, copy, local_rows), labels=[0, 1])-baseline
    write_json(destination, {"calibration_only": True, "baseline_log_loss": baseline,
                             "permutation_loss_increase": result,
                             "review": "Check rank/channel dominance; permutation importance does not prove leakage."})
    return result


def official_validate(script, data, output):
    script, data, output = Path(script), Path(data), Path(output)
    command = [sys.executable, str(script), "--matching", str(output/"matching_results.tsv"),
               "--candidate", str(output/"candidate_pairs.tsv"), "--test-dir", str(data), "--check-ids"]
    result = subprocess.run(command, capture_output=True, text=True)
    report = dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                  validator_sha256=file_hash(script),
                  output_hashes={n: file_hash(output/n) for n in ("matching_results.tsv", "candidate_pairs.tsv")})
    write_json(output/"official_validation.json", report)
    if result.returncode:
        raise ValueError("Official validator failed; see official_validation.json")
    return report


def license_inventory(destination):
    # Audit evidence, not a blanket claim that all dependencies are MIT/Apache.
    entries = []
    for dist in importlib.metadata.distributions():
        md = dist.metadata
        entries.append({"name": md.get("Name"), "version": dist.version,
                        "license_expression": md.get("License-Expression"),
                        "license": md.get("License"), "classifiers": md.get_all("Classifier", [])})
    write_json(destination, {"dependencies": entries, "pretrained_models": [],
                            "model": "locally trained sklearn trees under project MIT; not a pretrained encoder",
                            "review_required": "scikit-learn/numpy use BSD terms; model-license restriction must not be misreported as dependency licenses"})
    return entries


def model_sweep(run_dir, configurations, destination):
    """Train variants with shared immutable pair files; rank ONLY tuning scores.

    Configurations can vary seed/bag_seeds, calibration, max_iter, leaf count,
    monotonicity, objective, reranking, decoder and disabled_features. Training
    hardlinks are read-only by convention; no pair preparation runs in variants.
    """
    import os
    from .training import train
    root, output = Path(run_dir), Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for config in configurations:
        variant = output/digest(config)[:12]; variant.mkdir(exist_ok=True)
        for name in ("pairs_manifest.json", "train.f32", "train.jsonl", "calibration.f32", "calibration.jsonl"):
            target = variant/name
            if not target.exists(): os.link(root/name, target)
        model, tuning = train(variant, **config)
        bundle = load_model(model)
        x, _, _, rows = load_part(variant, "calibration")
        probabilities = predict_proba(bundle, x, rows)
        with (variant/"tuning_predictions.jsonl").open("w") as f:
            from .decision import decide
            for row in rows:
                if int(row["group"][:8], 16) % 2 == 0: continue
                p = probabilities[row["offset"]:row["offset"]+row["count"]]
                prediction = decide(bundle, row, p)
                f.write(json.dumps({**row, "predicted": prediction, "f05": entity_f05(row["truth"], prediction)})+"\n")
        results.append({"config": config, "path": str(variant), "tuning": tuning})
    results.sort(key=lambda r: r["tuning"]["country_weighted_macro_f05"], reverse=True)
    if len(results) > 1:
        base = Path(results[0]["path"])/"tuning_predictions.jsonl"
        for result in results[1:]:
            result["paired_against_best"] = paired_comparison(base, Path(result["path"])/"tuning_predictions.jsonl",
                                                               Path(result["path"])/"paired.json")
    write_json(output/"selection.json", {"selected_on": "calibration tuning half only", "variants": results,
                                       "ensemble_rule": "Retain added members only when paired gains/repair overlap justify their runtime."})
    return results


def soft_ownership(rows, alpha=.1):
    """Experimental soft penalty from other observable anchor groups.

    Input is OOF or held calibration rows with probabilities. Returns adjusted
    copies without ever deleting candidates or enforcing unique ownership.
    Pair with model_sweep/paired_comparison before enabling in a submission.
    """
    if not 0 <= alpha <= 1: raise ValueError("alpha must lie in [0,1]")
    targets = defaultdict(dict)
    for row in rows:
        for tid, p in zip(row["candidates"], row["probabilities"]):
            group = row["group"]
            targets[tid][group] = max(p, targets[tid].get(group, 0))
    adjusted = []
    for row in rows:
        probabilities = []
        for tid, p in zip(row["candidates"], row["probabilities"]):
            competitor = max((v for group, v in targets[tid].items() if group != row["group"]), default=0)
            probabilities.append(p*(1-alpha*competitor))
        adjusted.append({**row, "probabilities": probabilities})
    return adjusted


def perturbation_report(source1, truth_path, index_dir, destination, per_country=1000):
    """Training calibration retrieval under accent/abbreviation perturbations.

    These are synthetic input changes, not French validation labels.
    """
    from .common import Record, fold_latin
    anchors, _ = sample_anchors(source1, per_country)
    wanted = {a.entity_id for a in anchors}
    labels = {i: t for i, t in truth_rows(truth_path) if i in wanted}
    split = partition_anchors(anchors, labels)
    r = Retriever(index_dir)
    if r.language.payload and r.language.payload["split_hash"] != digest(split):
        raise ValueError("Perturbation sample must match the mined-language split")
    totals, hits = Counter(), Counter()
    try:
        for a in anchors:
            if split[a.entity_id] != "calibration": continue
            variants = {"original": a, "accent_folded": Record(a.entity_id, fold_latin(a.business_name), fold_latin(a.business_address), a.country),
                        "abbreviated": Record(a.entity_id, a.business_name, a.business_address.replace("Road", "Rd").replace("Street", "St").replace("Boulevard", "Blvd"), a.country)}
            for name, rec in variants.items():
                candidates = {c.record.entity_id for c in r.retrieve(rec)}
                hits[name] += len(labels[a.entity_id] & candidates)
                totals[name] += len(labels[a.entity_id])
    finally: r.close()
    result = {"recall": {k: hits[k]/max(1, totals[k]) for k in totals}, "not_french_accuracy": True}
    write_json(destination, result)
    return result


def normalization_sweep(source_files, source1, truth_path, destination, per_country=1000):
    """Rebuild one normalization view at a time; original views stay available."""
    from .retrieval import build_index
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    report = {}
    for view in (None, "suffix", "accents", "abbreviations", "reordering"):
        name = view or "baseline"
        build_index(source_files, root/name, disabled_views=() if view is None else (view,))
        report[name] = retrieval_sweep(source1, truth_path, root/name, root/(name+".json"), budgets=(80,), per_country=per_country)[0]
    write_json(root/"normalization_comparison.json", report)
    return report


def draft_methodology(run_dir, destination, team="", members=""):
    """Populate measured fields only; unresolved final-run checks remain explicit."""
    root = Path(run_dir)
    report = {}
    for name in ("pairs_manifest.json", "model.metrics.json", "trust_metrics.json", "output/inference_manifest.json", "output/validation.json", "output/official_validation.json"):
        if (root/name).exists(): report[name] = json.loads((root/name).read_text())
    if "pairs_manifest.json" not in report: raise ValueError("No measured pair run")
    # Raw training IDs and mined alias tables do not belong in a readable report.
    pairs = report["pairs_manifest.json"]
    pairs.pop("language", None)
    text = "# Business entity resolution methodology draft\n\n"
    text += f"Team: {team or '[FILL team]'}\n\nMembers: {members or '[FILL members]'}\n\n"
    text += "Training-only mined language, grouped 60/20/20 splits, reserved retrieval channels, calibrated tree classifier and anchor-level gate.\n\n"
    text += "## Recorded evidence\n\n```json\n"+json.dumps(report, indent=2)+"\n```\n\n"
    text += "[FILL confirm full target pool, final run selection, observed errors, limitations and portal rules; pilot evidence is not competition accuracy]\n"
    Path(destination).write_text(text, encoding="utf-8")
    return str(destination)
