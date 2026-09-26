# Repository structure and file responsibilities

This repository contains one notebook interface and a Python package named
`ber` (business entity resolution). The notebook controls the workflow; the
package performs the real indexing, retrieval, training, inference, validation,
and packaging. Generated data and model artifacts belong under `work/`, which is
ignored by Git.

```text
my-lovely-bottle/
├── notebooks/
│   └── amazon_ml_2026.ipynb       Main demo, training, evaluation and submission workflow
├── ber/
│   ├── __init__.py                Package marker
│   ├── __main__.py                Command-line interface (`python -m ber ...`)
│   ├── common.py                  TSV parsing, normalization, hashes, splits and F0.5
│   ├── aws.py                     S3 download/upload and safe ZIP extraction
│   ├── demo.py                    Synthetic offline dataset generator
│   ├── views.py                   Multilingual name/address representations
│   ├── language.py                Training-only aliases and character mapping
│   ├── retrieval.py               SQLite/FTS indexes and candidate generation
│   ├── features.py                Pairwise similarity feature construction
│   ├── training.py                Sampling, splits, model fit, calibration and reports
│   ├── decision.py                Anchor gate and final match-set decisions
│   ├── models.py                  Optional OOF, reranker, ranking and ensembles
│   ├── inference.py               Parallel prediction, validation, fallback and packaging
│   └── experiments.py             Controlled sweeps, audits, comparisons and registry
├── tests/
│   ├── test_pipeline.py           End-to-end, package and validation tests
│   ├── test_robustness.py         Unicode, retrieval and leakage regressions
│   └── test_experiments.py        Optional model-path tests
├── docs/
│   ├── AWS_SAGEMAKER_RUNBOOK.md   Exact AWS execution instructions
│   ├── REPOSITORY_STRUCTURE.md    This file map
│   ├── PIPELINE_AND_MODEL.md      Detailed algorithm and model explanation
│   ├── robust_run.md              Compact experiment instructions
│   ├── validation.md              Evaluation and split rules
│   ├── verification.md            What has actually been tested
│   ├── improvement_status.md      Checklist implementation status and limits
│   └── aws.md                     Short AWS notes
├── requirements.txt               Reproducible core Python dependencies
├── requirements-aws.txt           Optional Boto3 dependency for S3 transport
├── Documentation_template.md      Final competition methodology template
├── SUBMISSION_README.md           Reproduction notes placed in submission package
├── LICENSE                        Project license
└── .gitignore                     Prevents data, caches and artifacts entering Git
```

## How the files call one another

```text
amazon_ml_2026.ipynb
    ├── ber.aws → S3/ZIP handling
    ├── ber.common → input audit and fingerprints
    ├── ber.language → training-only learned language.json
    ├── ber.retrieval + ber.views → SQLite candidate indexes
    ├── ber.training + ber.features
    │       ├── ber.models (optional model variants)
    │       └── ber.decision (anchor-level output policy)
    ├── ber.inference → batched predictions and validation
    └── ber.experiments → run registry, official validator and license record
```

`ber/__main__.py` exposes most of the same operations as terminal commands. It
does not contain a second implementation; it calls the modules above. This keeps
notebook and command-line behavior consistent.

## Input files

The expected extracted layout is:

```text
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── utils/
    └── validate_submission.py
```

`source1` contains anchors whose matching S2/S3 entity IDs must be predicted.
The S2/S3 files are the target universe. Training ground truth lists the correct
target IDs for each training anchor. `TRAIN_ONLY=True` extracts and audits only
the training files.

## Generated work directory

For `MODE="full"`, schema version 3 writes under `work/full/v3/`:

```text
work/full/v3/
├── student_resource.zip           Optional downloaded private input
├── data/                           Private extracted input
├── registry/                       Comparable run summaries
└── runs/robust_v3_pilot/
    ├── language.json               Learned aliases/transliteration mapping
    ├── index_train/                Country SQLite training-target indexes
    ├── index_test/                 Test indexes, only when TRAIN_ONLY=False
    ├── train.f32                   Memory-mapped training features
    ├── calibration.f32             Memory-mapped calibration features
    ├── trust.f32                   Untouched evaluation features
    ├── *.jsonl                     Anchor metadata/candidate IDs for each partition
    ├── pairs_manifest.json         Fingerprints, counts and retrieval report
    ├── model.pkl                   Model, calibrator and decision policy bundle
    ├── model.metrics.json          Calibration/tuning report
    ├── experiment.json             Run configuration and selected results
    ├── without_US/                 Geographic-stress artifacts
    ├── without_India/              Geographic-stress artifacts
    ├── benchmark_workers_*/        Small test-runtime benchmark shards
    ├── output/                     Full matching and candidate output
    └── team_submission.zip         Final validated package
```

The feature arrays are raw `float32` matrices and the JSONL files store their
offsets. This avoids keeping every training pair in Python objects. Manifests
contain file and configuration fingerprints so stale artifacts cannot silently
be reused with a different dataset, language map, index, or model.

## Module-by-module behavior

### `common.py`

Reads literal tab-separated files without treating quote characters specially,
cleans placeholder nulls, normalizes Unicode digits, produces deterministic
hashes, computes the entity-level F0.5 score, and creates grouped 60/20/20
train/calibration/trust partitions. Identical normalized signatures stay in the
same split to reduce leakage.

### `views.py` and `language.py`

`views.py` creates multiple loss-tolerant representations while preserving the
original value: normalized, accent-folded, suffix-reduced, compact, token-sorted,
Indic-normalized, phonetic, alias, postcode, house-number and street-key views.
It recognizes Latin, Devanagari, Bengali and Tamil scripts.

`language.py` learns business aliases and a small character transducer from the
training links only. This helps compare Indic and Latin spellings without an
external transliteration service. Its artifact records the exact split and
training anchors that produced it.

### `retrieval.py`

Builds one SQLite database per country from every S2/S3 target. Exact indexes and
FTS5 text indexes provide several retrieval channels. Each channel reserves some
candidates; reciprocal-rank fusion fills the remaining budget. Retrieval is
adaptive when sources disagree, names are frequent, or addresses are absent.

The retriever intentionally returns a bounded candidate set. This changes an
otherwise impossible comparison against millions of targets into at most a few
hundred plausible pairs per anchor.

### `features.py`

Turns each anchor-candidate pair into a numeric vector. Features cover name and
address edit similarities, token overlap, IDF-weighted evidence, character
TF-IDF cosine similarity, acronym/prefix/containment signals, retrieval channel
evidence, target-name frequency, structured address agreement, scripts,
romanized similarity, source indicators and missing/contradiction flags.

### `training.py`, `decision.py`, and `models.py`

`training.py` trains the default pair classifier, calibrates its probabilities,
and tunes thresholds without opening the trust partition. `decision.py` adds an
anchor-level gate for deciding when no target should be returned and supports an
optional expected-F0.5 decoder. `models.py` contains opt-in research variants:
out-of-fold scoring, a context reranker, pairwise logistic ranking, and seed
ensembles. These variants are not the default production model.

### `inference.py`

Loads one read-only index and model per worker, predicts in bounded batches, and
writes atomic resumable shards. It combines shards into `matching_results.tsv`
and `candidate_pairs.tsv`, checks IDs and country membership, records drift and
runtime information, invokes the supplied validator, and creates the final ZIP.

### `experiments.py`

Keeps model and retrieval comparisons away from the main path. It implements
calibration-only sweeps, paired bootstrap comparisons, feature permutation
audits, normalization ablations, label-noise warnings, run registration,
dependency-license inventory, and methodology drafting.

## Safe modification rules

- Change `RUN_NAME` after changing features, retrieval, sample sizes, or model
  settings.
- Never commit `work/`, the source ZIP, extracted data, indexes, models, or
  predictions.
- Select experiments on calibration results. Open trust once after selecting a
  configuration.
- Keep `TRAIN_ONLY=True` until training and model selection are complete.
- Treat optional ranking, reranking, bagging, utility decoding and Soundex as
  experiments until measured comparisons justify them.
