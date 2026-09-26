"""Command-line entry points used by both notebooks and submission reproduction."""

import argparse
import json
from pathlib import Path

from .demo import make_demo
from .inference import package_submission, predict, validate, fallback
from .retrieval import build_index
from .training import evaluate_trust, prepare_pairs, train


def main():
    p = argparse.ArgumentParser(description="Business entity resolution")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo"); demo.add_argument("--data", required=True)
    index = sub.add_parser("index"); index.add_argument("--data", required=True); index.add_argument("--split", choices=["train", "test"], required=True); index.add_argument("--index", required=True)
    pairs = sub.add_parser("pairs"); pairs.add_argument("--data", required=True); pairs.add_argument("--index", required=True); pairs.add_argument("--run", required=True); pairs.add_argument("--per-country", type=int, default=5000)
    pairs.add_argument("--max-pairs", type=int, default=3_000_000)
    fitting = sub.add_parser("train"); fitting.add_argument("--run", required=True); fitting.add_argument("--exclude-country")
    fitting.add_argument("--max-iter", type=int, default=180)
    fitting.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
    fitting.add_argument("--save-oof", action="store_true")
    fitting.add_argument("--rerank", action="store_true")
    fitting.add_argument("--early-stopping", action="store_true")
    fitting.add_argument("--objective", choices=["classifier", "ranking"], default="classifier")
    fitting.add_argument("--decoder", choices=["threshold", "utility"], default="threshold")
    fitting.add_argument("--disable-feature", action="append", default=[])
    evaluation = sub.add_parser("evaluate"); evaluation.add_argument("--run", required=True); evaluation.add_argument("--model", required=True); evaluation.add_argument("--country")
    prediction = sub.add_parser("predict"); prediction.add_argument("--data", required=True); prediction.add_argument("--index", required=True); prediction.add_argument("--model", required=True); prediction.add_argument("--output", required=True); prediction.add_argument("--batch-size", type=int, default=500); prediction.add_argument("--limit", type=int)
    prediction.add_argument("--workers", type=int, default=1)
    validation = sub.add_parser("validate"); validation.add_argument("--data", required=True); validation.add_argument("--index", required=True); validation.add_argument("--output", required=True)
    packaging = sub.add_parser("package"); packaging.add_argument("--repo", default="."); packaging.add_argument("--output", required=True); packaging.add_argument("--model", required=True); packaging.add_argument("--destination", required=True)
    mining = sub.add_parser("mine-language")
    mining.add_argument("--data", required=True); mining.add_argument("--output", required=True)
    mining.add_argument("--per-country", type=int, default=5000); mining.add_argument("--exclude-country")
    index.add_argument("--language")
    index.add_argument("--disable-view", action="append", default=[], choices=["suffix", "accents", "abbreviations", "reordering"])
    mining.add_argument("--split-strategy", choices=["signature", "ambiguity"], default="signature")
    pairs.add_argument("--split-strategy", choices=["signature", "ambiguity"], default="signature")
    audit = sub.add_parser("audit"); audit.add_argument("--data", required=True); audit.add_argument("--train-only", action="store_true"); audit.add_argument("--full", action="store_true")
    exact = sub.add_parser("fallback"); exact.add_argument("--data", required=True); exact.add_argument("--index", required=True); exact.add_argument("--output", required=True); exact.add_argument("--workers", type=int, default=1)
    official = sub.add_parser("official-validate"); official.add_argument("--script", required=True); official.add_argument("--test-dir", required=True); official.add_argument("--output", required=True)
    sweep = sub.add_parser("retrieval-sweep"); sweep.add_argument("--data", required=True); sweep.add_argument("--index", required=True); sweep.add_argument("--output", required=True)
    registry = sub.add_parser("register"); registry.add_argument("--run", required=True); registry.add_argument("--registry", required=True); registry.add_argument("--leaderboard", type=float); registry.add_argument("--submission")
    licenses = sub.add_parser("licenses"); licenses.add_argument("--output", required=True)
    comparison = sub.add_parser("compare"); comparison.add_argument("--left", required=True); comparison.add_argument("--right", required=True); comparison.add_argument("--output", required=True)
    importance = sub.add_parser("feature-audit"); importance.add_argument("--run", required=True); importance.add_argument("--model", required=True); importance.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "demo":
        result = str(make_demo(a.data))
    elif a.command == "index":
        result = build_index([Path(a.data)/a.split/f"{a.split}_source{s}.tsv" for s in (2, 3)], a.index, a.language, a.disable_view)
    elif a.command == "pairs":
        result = prepare_pairs(Path(a.data)/"train/train_source1.tsv", Path(a.data)/"train/train_ground_truth.tsv", a.index, a.run, per_country=a.per_country, max_pairs=a.max_pairs, split_strategy=a.split_strategy)
    elif a.command == "train":
        path, report = train(a.run, exclude_country=a.exclude_country, max_iter=a.max_iter, calibration=a.calibration,
                             save_oof=a.save_oof, rerank=a.rerank, early_stopping=a.early_stopping,
                             objective=a.objective, decoder=a.decoder, disabled_features=a.disable_feature)
        result = {"model": str(path), "tuning": report}
    elif a.command == "evaluate":
        result = evaluate_trust(a.run, a.model, a.country)
    elif a.command == "predict":
        result = predict(Path(a.data)/"test/test_source1.tsv", a.index, a.model, a.output, a.batch_size, a.limit, a.workers)
    elif a.command == "validate":
        result = validate(Path(a.data)/"test/test_source1.tsv", a.index, a.output)
    elif a.command == "package":
        result = package_submission(a.repo, a.output, a.model, a.destination)
    elif a.command == "mine-language":
        from .language import mine_language
        result = mine_language(Path(a.data)/"train/train_source1.tsv", Path(a.data)/"train/train_ground_truth.tsv",
                              [Path(a.data)/"train"/f"train_source{s}.tsv" for s in (2, 3)], a.output, a.per_country, exclude_country=a.exclude_country, split_strategy=a.split_strategy)
        result = {k: v for k, v in result.items() if k not in {"training_anchor_ids", "aliases", "characters"}}
    elif a.command == "audit":
        from .common import audit_inputs
        result = audit_inputs(a.data, a.full, a.train_only)
    elif a.command == "fallback":
        result = fallback(Path(a.data)/"test/test_source1.tsv", a.index, a.output, a.workers)
    else:
        from . import experiments as e
        if a.command == "official-validate": result = e.official_validate(a.script, a.test_dir, a.output)
        elif a.command == "retrieval-sweep": result = e.retrieval_sweep(Path(a.data)/"train/train_source1.tsv", Path(a.data)/"train/train_ground_truth.tsv", a.index, a.output)
        elif a.command == "register": result = e.register_run(a.run, a.registry, a.leaderboard, a.submission)
        elif a.command == "licenses": result = e.license_inventory(a.output)
        elif a.command == "compare": result = e.paired_comparison(a.left, a.right, a.output)
        elif a.command == "feature-audit": result = e.feature_audit(a.run, a.model, a.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
