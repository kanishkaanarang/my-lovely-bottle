# How the entity-resolution system works

## The problem being solved

For every Source 1 business entity, the system must return zero, one, or several
matching entity IDs from Sources 2 and 3. Names and addresses can differ because
of abbreviations, punctuation, token order, missing fields, aliases, accents,
transliteration, and writing systems such as Devanagari. A direct comparison of
every Source 1 row with every S2/S3 row would be far too large.

The solution is a two-stage entity-resolution system:

```text
S1 anchor
   ↓
multilingual normalization and structured views
   ↓
high-recall retrieval from country-specific SQLite/FTS indexes
   ↓
at most ~320 candidate pairs per anchor in the full profile
   ↓
pairwise similarity features
   ↓
calibrated histogram gradient-boosting classifier
   ↓
country/source thresholds + learned empty-result gate
   ↓
zero, one, or multiple S2/S3 entity IDs
```

The largest engineering idea is **retrieve first, classify second**. Retrieval
makes the search computationally feasible; classification combines many weak
matching signals into a probability for each retrieved pair.

## Stage 1: input protection and validation

The parser reads TSV files literally, preserves Unicode, converts digits from
other scripts to ASCII where appropriate, and removes known null placeholders.
The input audit checks required headers, source ID conventions, country values,
duplicate IDs, row counts, and required files.

Every important artifact records hashes of its inputs and configuration. A
model trained from one index or split cannot silently run with a different one.

## Stage 2: multilingual representations

Each name and address is represented in several ways instead of forcing all
information through one destructive normalization:

- original normalized Unicode text;
- Latin accent-folded text;
- legal-suffix-reduced business name;
- compact and token-sorted names;
- local abbreviation-expanded addresses;
- postcode, house-number range, unit and street key;
- Indic-normalized text;
- optional token-level Soundex;
- URL/trade-name aliases; and
- learned romanized representations.

The original text always remains available. Lossy views are extra evidence, not
replacements. This matters for Devanagari and other Indic text: the system can
retrieve an exact native-script match, compare a lightly folded native form,
and compare a learned Latin rendering without discarding the original spelling.

The learned language artifact uses training links only. It mines aliases and
iteratively aligns non-Latin character sequences with Latin spellings. The
artifact stores its training IDs and split hash so it cannot leak across a held
country or incompatible data split.

## Stage 3: candidate retrieval

Every S2/S3 target is stored in a country-specific SQLite database. B-tree
indexes support exact views, while SQLite FTS5 supports token and character-gram
search. Retrieval uses channels such as:

- exact normalized, core, folded, compact, sorted and Indic names;
- name and address token search;
- rare-token conjunctions;
- address character trigrams;
- postcode and house/street matches;
- trade-name and URL aliases;
- learned romanized names; and
- optional phonetic keys.

Each channel receives a reservation so one strong but narrow channel cannot
consume the complete budget. Reciprocal-rank fusion then fills unused positions.
When names are common, addresses are missing, or S2/S3 channels disagree, the
retriever activates additional evidence. Complete exact-match buckets are ranked
by textual evidence before truncation, avoiding row-order bias.

In the normal full-data configuration, retrieval allows up to 160 candidates
per source, so an anchor normally produces at most 320 S1-target pairs. Retrieval
recall is therefore the ceiling for the downstream model: a true target absent
from this candidate set cannot be recovered by classification.

## Stage 4: pairwise features

For each retrieved pair, the system produces about one hundred numeric features.
The major groups are:

| Feature family | What it measures |
| --- | --- |
| Name similarity | edit ratio, token-sort ratio, Jaro-Winkler, prefix, containment, acronym and Monge-Elkan signals |
| Address similarity | edit/token similarity, structured number/range/fraction/unit agreement and contradictions |
| Weighted text evidence | rare-token overlap, IDF scores and character TF-IDF cosine similarity |
| Retrieval evidence | channel hits, ranks, reciprocal-rank score and cross-channel agreement |
| Frequency | whether a normalized name is unusually common or distinctive |
| Language/script | script pair, Indic fold and learned romanized similarity |
| Data quality | missing name/address, no digits, conflicting structured fields and weak locality/postcode signals |
| Source context | S2/S3 indicator and bounded cross-source name corroboration |

The features are stored as `float32` memory maps. This allows training to read a
large matrix without materializing it as millions of Python objects.

## Stage 5: the main trained model

The default and primary model is scikit-learn's
`HistGradientBoostingClassifier`. It is a CPU-based ensemble of shallow decision
trees trained on the pairwise feature vectors. Important default settings are:

```text
max_iter          = 180 in full mode
max_leaf_nodes    = 31
min_samples_leaf  = 20
learning_rate     = 0.08
l2_regularization = 2
random_state      = 2026
```

The model uses monotonic constraints for selected similarities such as name,
address, IDF and romanized similarity. All else equal, increasing one of these
positive similarity signals is prevented from lowering the raw match score.

Histogram gradient boosting is used because it handles nonlinear interactions,
mixed feature scales, missing/contradictory signals, and millions of rows more
efficiently than conventional exact tree boosting. The current implementation
does not train a neural network and does not use a pretrained language model.

## Stage 6: probability calibration and decision policy

The raw tree score is not treated as a trustworthy probability. The calibration
partition is divided by anchor group:

1. one half fits Platt calibration using logistic regression; and
2. the other half selects decision thresholds and the empty-result policy.

The default policy tunes for the challenge's entity-level F0.5 metric, which
weights precision more heavily than recall. Country-specific and source-specific
thresholds are learned where evidence exists. For an unseen country, the system
uses the policy that performed best in the worst observed calibration country,
instead of pretending it has labels for the unseen country.

A small logistic anchor gate looks at the distribution of candidate scores for
one S1 entity. It helps decide when all candidates should be rejected. Candidates
above the selected thresholds can still produce multiple matches, which is
necessary because the correct answer is a set rather than a single class.

## Stage 7: leakage-resistant evaluation

Anchors are deterministically split 60/20/20 into training, calibration, and
trust partitions, stratified by country and match cardinality. Duplicate
normalized signatures stay together. Retrieval/channel reports used during
development exclude trust.

Geographic stress rebuilds the learned language mapping, index, candidate pairs,
and model while withholding one country. This tests whether improvements depend
too heavily on country-specific labels. Trust evaluation remains disabled while
configurations are being compared, then is opened once for the selected system.

Reported diagnostics include:

- macro entity F0.5 and country-weighted F0.5;
- pair precision and recall;
- true-link retrieval ceiling;
- complete-match-set retrieval;
- false merges for genuinely empty anchors;
- per-country calibration error;
- script, missing-address and cardinality cohorts; and
- retrieval miss, false merge and threshold/gate error tags.

## Stage 8: scalable inference

Inference uses a bounded process pool. Each worker opens the SQLite index in
read-only mode, loads one model bundle, limits native numerical threads, and
handles batches of anchors. Batches are written as atomic shards with checksums.
A stopped run can reuse valid completed shards, while changed data, index, model,
batch size, retrieval settings or benchmark sampling produces a new signature
and is rejected in the old output directory.

Before full inference, a fresh 3,000-anchor country-aware benchmark estimates
runtime and reports candidate-count/top-score histograms. Final output receives
both internal structural validation and the competition's supplied validator.
The package is bound to hashes of the exact validated output files.

## Optional experiments versus the default system

The repository contains additional methods for controlled comparison:

- isotonic instead of Platt calibration;
- pairwise logistic ranking;
- out-of-fold score-context reranking;
- grouped seed bagging;
- an expected-F0.5 utility decoder;
- soft ownership adjustments;
- Soundex retrieval;
- feature and normalization ablations; and
- retrieval-budget sweeps.

These are opt-in experiments. They are not automatically part of the primary
model and should be promoted only after a paired calibration comparison and a
final untouched-trust evaluation.

## What is deliberately absent

There is no external company database, geocoder, web lookup, pretrained encoder,
or GPU model. A multilingual encoder remains a possible experiment only after
checking organizer rules, license, model size, runtime, and measured incremental
gain. The system currently gains multilingual robustness through multiple text
views and training-only character alignment.

The code also avoids global one-target-one-owner enforcement. One target may
legitimately correspond to more than one anchor, and hard transitive closure can
amplify an incorrect link. Cross-source evidence is used as a bounded feature
rather than as irreversible graph merging.
