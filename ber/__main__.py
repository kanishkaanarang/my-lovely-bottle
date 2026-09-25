"""Command-line entry points used by both notebooks and submission reproduction."""

import argparse
import json
from pathlib import Path

from .demo import make_demo
from .inference import package_submission, predict, validate
from .retrieval import build_index
from .training import evaluate_trust, prepare_pairs, train


def main():
    p = argparse.ArgumentParser(description="Business entity resolution")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo"); demo.add_argument("--data", required=True)
    index = sub.add_parser("index"); index.add_argument("--data", required=True); index.add_argument("--split", choices=["train", "test"], required=True); index.add_argument("--index", required=True)
    pairs = sub.add_parser("pairs"); pairs.add_argument("--data", required=True); pairs.add_argument("--index", required=True); pairs.add_argument("--run", required=True); pairs.add_argument("--per-country", type=int, default=5000)
    fitting = sub.add_parser("train"); fitting.add_argument("--run", required=True); fitting.add_argument("--exclude-country")
    evaluation = sub.add_parser("evaluate"); evaluation.add_argument("--run", required=True); evaluation.add_argument("--model", required=True); evaluation.add_argument("--country")
    prediction = sub.add_parser("predict"); prediction.add_argument("--data", required=True); prediction.add_argument("--index", required=True); prediction.add_argument("--model", required=True); prediction.add_argument("--output", required=True); prediction.add_argument("--batch-size", type=int, default=500); prediction.add_argument("--limit", type=int)
    validation = sub.add_parser("validate"); validation.add_argument("--data", required=True); validation.add_argument("--index", required=True); validation.add_argument("--output", required=True)
    packaging = sub.add_parser("package"); packaging.add_argument("--repo", default="."); packaging.add_argument("--output", required=True); packaging.add_argument("--model", required=True); packaging.add_argument("--destination", required=True)
    a = p.parse_args()
    if a.command == "demo":
        result = str(make_demo(a.data))
    elif a.command == "index":
        result = build_index([Path(a.data)/a.split/f"{a.split}_source{s}.tsv" for s in (2, 3)], a.index)
    elif a.command == "pairs":
        result = prepare_pairs(Path(a.data)/"train/train_source1.tsv", Path(a.data)/"train/train_ground_truth.tsv", a.index, a.run, per_country=a.per_country)
    elif a.command == "train":
        path, report = train(a.run, exclude_country=a.exclude_country); result = {"model": str(path), "tuning": report}
    elif a.command == "evaluate":
        result = evaluate_trust(a.run, a.model, a.country)
    elif a.command == "predict":
        result = predict(Path(a.data)/"test/test_source1.tsv", a.index, a.model, a.output, a.batch_size, a.limit)
    elif a.command == "validate":
        result = validate(Path(a.data)/"test/test_source1.tsv", a.index, a.output)
    else:
        result = package_submission(a.repo, a.output, a.model, a.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
