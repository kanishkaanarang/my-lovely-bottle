# Robustness revision: implementation and evidence

The supplied improvement checklist is tracked below. “Implemented” means code
and checks exist, not that the idea has demonstrated a competition-score gain.
Only the reattached archive's **training** files were inspected for this
revision. Official test records were not opened. Synthetic test fixtures are
used to exercise inference and packaging.

## Default pipeline

The notebook defaults to demo mode. In full mode, `TRAIN_ONLY=True` and
`EVALUATE_TRUST=False`. It builds grouped, cardinality-stratified 60/20/20
train/calibration/trust partitions. Each calibration group stays entirely in
either probability fitting or policy tuning. Language artifacts record their
training anchor IDs and split hash; mismatches fail rather than reuse labels
from another split. Geographic stress rebuilds language, indexes and features
without the held country's labels.

Schema 3 indexes are incompatible with the original baseline. Use the notebook's
new `v3` working directory. Do not reuse original indexes, feature caches or models.

## Checklist

| Items | Status and implementation |
| --- | --- |
| A1–A4 | Implemented: literal TSV I/O with `QUOTE_NONE` and `quotechar=None`, embedded/full-field placeholder cleaning, ASCII digit conversion, schema/count gates. Training-only audit opens four files; explicit official full mode requires seven and 1,732,544 test anchors. |
| A5 | Text views are stored in SQLite at indexing time, including originals, folded/core/compact/sorted/Indic/expanded/romanized views. |
| B1–B8, B10–B11 | Implemented country-specific abbreviation maps, extended suffixes, native suffixes, French elisions, Indian variants, trade names, raw structured numbers, compact names and secondary Indic folding. These are lossy extra views, never replacements for original text. Native suffix dictionaries are deliberately small, not comprehensive language coverage. |
| B9, C9 | Local token Soundex is implemented as an opt-in retrieval channel (`RetrievalConfig(phonetic=True)`). |
| C1–C8, C10, C12 | Implemented channel reservations plus reciprocal-rank filling, score-ranked complete exact buckets, address trigrams, postcode/street keys, rare-token AND queries, sorted and compact names, and adaptive additional channels. No pre-ranking exact bucket is cut by row ID. |
| C11, J2 | Training-only full-name aliases and an iteratively aligned character transducer. No external transliteration corpus/library. Mapping quality must be measured; this is not a guarantee of standard transliteration. Artifacts cannot be applied to a different training partition. |
| C13 | Case/whitespace country aliases fall back with a warning. A truly unknown partition fails explicitly; cross-country guessing or silently returning no matches would undermine validation. |
| C14, G4 | Calibration-only retrieval budget/channel sweeps and incremental recall reports. Pair-preparation reports omit trust metrics. Reports distinguish raw channel recall from final retained-candidate recall/oracle. |
| D1–D12, J6 | Implemented IDF/rare overlap, target frequencies, acronym, Jaro-Winkler/partial/prefix/containment, character IDF cosine, Monge-Elkan, ranges/fractions/units/bis/ter, postcode and weak locality proxies, text-only URL alias extraction, script-pair and learned romanized similarity, quality/missingness/contradiction flags. “Locality” is a comma-delimited proxy, not geocoding. |
| E1 | Large profile samples 200k anchors per country: approximately 120k train and 20k policy-tuning anchors per country after grouped 60/20/20 splitting. Pilot remains the default. 140m pair capacity is configurable; full-scale time/RAM/disk have not been measured. |
| E2, J7 | Optional grouped three-fold OOF score export and second-pass score/rank/margin reranking. Currently enabled only for indexes with no label-mined language: it rejects mined-language caches because proper OOF would require refitting language and retrieval in every fold. Never reports in-sample predictions as OOF. |
| E3 | Optional monotonic constraints on selected similarity features, enabled in the default classifier. |
| E4–E6 | Optional grouped early stopping, configuration sweeps, anchor-group bootstrap seed bagging, Platt/isotonic calibration, and within-anchor pairwise logistic ranking. These are experimental alternatives, selected on calibration only. |
| E7, G7 | Calibration permutation audit and feature-disable/retrain experiments. Rank/channel dominance is flagged for review, not automatically called leakage. |
| E8, J9 | Bounded cross-source name corroboration is a soft feature. No transitive closure or hard ownership enforcement. |
| E9 | Optional `soft_ownership` helper adjusts cached probabilities using competing anchor groups. It preserves every candidate and allows multiple owners. This is a calibration experiment, not enabled in the production inference path. |
| E10 | Pending: no pretrained multilingual encoder is loaded. A specific model, MIT/Apache-2.0 model license, <=8B parameter count, organizer permission and residual-cohort gain are prerequisites. No unreviewed dependency or model download was added. |
| F1 | Tuning uses the supplied approximately 45/55 US/India mix, explicitly configured in `TEST_MIX`. No test labels or new test inspection. |
| F2–F3 | Country/source thresholds and a learned anchor-level gate, trained on the probability-fitting half and tuned on the separate tuning half. |
| F4 | Opt-in expected-F0.5 decoder with an independent-Bernoulli cardinality distribution plus a Poisson missing-link approximation fitted from calibration retrieval misses. Independence is an approximation; decoder runtime may be substantial for large blocks. |
| F5 | Unseen-country threshold/top-score policy maximizes worst observed calibration-country performance; its learned-gate threshold is also selected by the worst-country score. It does not measure French accuracy. |
| G1 | Country/cardinality-stratified grouped 60/20/20 split. Conflicting labels in identical signatures stay together. |
| G2 | Optional `ambiguity` split groups sorted suffix-reduced names within country, across addresses. It covers reordering/suffix variants but is not exhaustive semantic/typo clustering. |
| G3 | Geographic stress is enabled by default. Its language artifacts exclude the held country's labels; trust remains closed until explicit experiment selection. Stress costs extra indexing/training. |
| G5, G8 | Trust cohort/source/script/cardinality/missingness reports, cross-script cohorts, and retrieval-miss/false-merge/gate-or-threshold error tags. |
| G6, J8 | Paired anchor differences, repair/harm counts and grouped bootstrap intervals. Use these to justify extra models; ensemble variants are never automatically promoted to the notebook default. |
| G9–G10 | Synthetic regressions inspired by the six supplied case studies, accent/abbreviation checks, and training-calibration perturbation-report tooling. Real case rows remain private. Synthetic perturbations do not establish French accuracy. |
| G11–G12 | JSON run registry and TSV comparison table with optional public leaderboard score and submission identifier. No automatic portal upload. |
| G13 | Optional indistinguishable-signature label audit. Different target-ID sets are a warning, not proof of incorrect labels. |
| J5 | Rebuild-and-compare normalization-view ablations for suffixes, accents, expansion and reordering. Other views can compensate; this measures the marginal contribution of that view. |
| H1–H4 | Bounded process-pool inference, read-only SQLite per worker, 3,000 mixed-country reservoir benchmark, cached text-pair features with IDs restored, batched row fetch and mmap. |
| H5–H6 | Index/pair/train/inference timing, index disk size, inference CPU/platform and per-country candidate/top-score histograms. Distribution drift is not labeled accuracy. |
| I1 | Uncapped normalized exact-name/address union fallback, resumable and strictly validated. It is an emergency rule and can produce false merges; not the default matcher. Official fallback generation remains disabled in training-only mode. |
| I2 | Supplied-validator wrapper always passes `--check-ids`, captures output and binds validation to output hashes. Strict validation remains mandatory. Official full-output validation is pending the future full run. |
| I3 | Package extraction/rebuilt-index/inference round-trip test, also run in a newly installed Python environment. Learned language artifacts ship with the package. |
| I4 | Installed-distribution license inventory. Core NumPy/scikit-learn dependencies use BSD terms; do not claim they are all MIT/Apache. The locally fitted model is covered by the project's MIT license; no pretrained model is present. |
| I5, J1 | Organizer/portal confirmation pending. The user indicated two days remaining, but exact cutoff, submission quota, final-selection procedure and pretrained/library permission were not supplied. |
| I6 | Measured methodology draft generation is implemented. Final team details, full-pool results, runtime and official validation must be filled after the real selected run. They cannot truthfully be inferred from a reduced-pool pilot. Packaging continues to reject unresolved markers. |
| J3 | Trust evaluation is cached by model hash. A different model cannot overwrite a completed trust evaluation in the same run. Experiments select on calibration; public leaderboard score is logging-only. |
| J4 | No negative subsampling is introduced. All retrieved candidate pairs are kept, with anchor-normalized weights. If subsampling is added later, prevalence-restoring weights and unchanged calibration sampling are required. |

## Local evidence

- Nine automated test methods cover metric/Unicode/TSV edge cases, channel ranking,
  train-only mapping guards, serial/parallel equivalence, resume, fallback,
  package extraction and index rebuilding, and optional model experiments.
- All 12 notebook code cells were executed on generated data, including
  geographic stress. Published notebook outputs are empty.
- A scan of the reattached training TSVs read 2,206,821 Source 1 rows,
  5,034,616 Source 2 rows, and 5,285,603 Source 3 rows. The name script detector
  counted 339,386 Indic-or-mixed names in Source 2 and 206,290 in Source 3.
- A private compatibility pilot used 606 real training anchors, including the
  six audit cases, their true targets and 10,000 distractors. It exercised
  language mining, indexing, features, fitting/calibration and held-out reports.
  Its reduced target pool invalidates competition-score comparisons.

No claim is made about complete AWS throughput, 100k+ training performance,
French accuracy, public leaderboard gains or final acceptance. These need the
corresponding real runs and organizer confirmations.
