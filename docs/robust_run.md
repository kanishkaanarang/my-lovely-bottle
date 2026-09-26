# Running the robustness revision

Start in `notebooks/amazon_ml_2026.ipynb`. Use a new work directory; schema 3
cannot reuse original indexes or models. The main README and old AWS guide were
removed upstream by the repository owner and have not been restored.

## Safe development sequence

1. Install `requirements.txt` and run `python -m unittest discover -s tests -v`.
2. Run the notebook in demo mode.
3. Set `MODE="full"`, `TRAIN_ONLY=True`, `PROFILE="pilot"`, and `DATA_ROOT` to
   the parent of the supplied `train/` directory. No official test access occurs.
4. Leave `EVALUATE_TRUST=False` while comparing settings. The pilot uses 1,000
   sampled anchors per country but indexes every supplied training target.
5. Inspect calibration performance, channel coverage, memory and timing. Use
   experiments below for bounded comparisons. Geographic stress is on by default
   and rebuilds separate indexes to exclude held-country label-derived language.
6. Select a configuration, then open trust once with `EVALUATE_TRUST=True`.
7. Only after benchmarking resource use, consider `PROFILE="large"`. It samples
   200,000 anchors per country and permits up to 140 million feature pairs.
   It is not an instruction to launch expensive compute immediately.
8. Later, explicitly disable `TRAIN_ONLY`, provide the complete dataset and
   official validator, and run the mixed-country inference benchmark. Change
   benchmark directory for a fresh timing. Keep the original raw dataset private.
9. Enable `RUN_FULL_TEST` only after reviewing the estimate. Run strict and
   official ID validation, complete methodology, and package. The notebook's
   packager requires official validation bound to the exact output hashes.

## Experiment APIs

Run these inside the project environment. These are opt-in and can be expensive.
Use private output directories outside Git.

```python
from ber.experiments import (
    retrieval_sweep, normalization_sweep, perturbation_report, feature_audit,
    model_sweep, paired_comparison, register_run, label_noise, draft_methodology,
)

# Use the SAME sample size/seed as the language artifact; mismatches are rejected.
retrieval_sweep(DATA_ROOT / "train/train_source1.tsv",
                DATA_ROOT / "train/train_ground_truth.tsv", TRAIN_INDEX,
                RUN_DIR / "retrieval_sweep.json", per_country=PER_COUNTRY)

# Rebuild separate no-language indexes, disabling one view at a time.
normalization_sweep([DATA_ROOT / "train" / f"train_source{s}.tsv" for s in (2, 3)],
                    DATA_ROOT / "train/train_source1.tsv",
                    DATA_ROOT / "train/train_ground_truth.tsv",
                    RUN_DIR / "normalization_sweep", per_country=1000)

feature_audit(RUN_DIR, MODEL_PATH, RUN_DIR / "feature_audit.json")
variants = model_sweep(RUN_DIR, [
    {"max_iter": 100, "calibration": "platt"},
    {"max_iter": 180, "calibration": "isotonic", "early_stopping": True},
    {"max_iter": 100, "disabled_features": ["heuristic", "retrieval_rank"]},
    {"max_iter": 100, "bag_seeds": [2027]},
    {"objective": "ranking"},
], RUN_DIR / "variants")

register_run(RUN_DIR, WORK_ROOT / "registry", leaderboard=None, submission=None)
draft_methodology(RUN_DIR, RUN_DIR / "methodology_draft.md")
```

`model_sweep` never copies or evaluates trust. It records paired tuning-anchor
repairs and harms against the best tuning variant. Do not retain extra ensemble
members solely because the aggregate mean moved slightly.

OOF reranking requires a **no-label-language index** at present. For that
experiment, build an index without `--language`, prepare a separate run, then:

```bash
python -m ber train --run work/no_language_run --save-oof --rerank
```

The alternative ambiguity stress split must be selected in both `mine-language`
and `pairs` using `--split-strategy ambiguity`. It groups token-reordered and
suffix-reduced name variants across addresses; it is intentionally conservative.

For normalization ablations, `index --disable-view suffix` (or `accents`,
`abbreviations`, `reordering`) persists that choice in the index fingerprint.
Inference rejects a model/index normalization mismatch.

## Submission checks

```bash
python -m ber official-validate \
  --script /private/student_resource/utils/validate_submission.py \
  --test-dir /private/student_resource/dataset/test \
  --output /private/run/output
```

The supplied validator can require substantial RAM when checking all candidate
IDs. Keep its output and the strict validator's output with the run. Neither
check estimates accuracy.

Use `python -m ber licenses --output work/dependency_licenses.json` to record
the installed environment's license metadata. No encoder is downloaded. Adding
one requires a reviewed model license, parameter count, organizer clearance,
and a measured residual-cohort benefit.

The exact-union fallback is available through `python -m ber fallback --help`.
It can contain many false positives and is not automatically substituted for
the trained model. The optional `soft_ownership` and utility decoder are research
tools; they do not enforce one target per anchor or global unique ownership.
