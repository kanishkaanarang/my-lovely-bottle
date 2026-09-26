# What the metrics mean

The supplied task is multi-match, per-reference entity resolution. A candidate miss is an unavoidable false negative for that blocker. Always compare model score to the oracle score for its exact final candidate set.

## Split protocol

- Reservoir sample anchors independently within each observed training country; retain the full S2/S3 target pool for search.
- Group `(country, normalized name, normalized address)`, stratify groups by country and true-cardinality bin, and assign a seeded 60% training, 20% calibration, 20% trust split. Exact duplicate signatures stay together, independently of IDs.
- Train only on retrieved training candidates. Every candidate gets its label from that anchor's supplied true set. True links outside the candidate set are never injected to inflate retrieval recall.
- Split calibration anchors again by deterministic group hash: one half fits the probability calibrator and anchor gate; the other selects thresholds. Duplicate signatures cannot leak across these halves.
- Evaluate trust anchors with the frozen model, calibrator and decision policy. Keep per-anchor candidates, probabilities, truth and selected predictions for paired analysis.

Country-balanced sampling makes the unweighted sample macro score different from the deployment mixture. Both sample macro and a country-weighted score using the supplied approximately 45/55 US/India mixture are reported. Threshold selection uses the latter. Neither estimates the unseen French score. The default policy for an unseen country is chosen using worst-country calibration performance, not fabricated French labels.

The observed training data assign each target ID to a single anchor. This assumption supports grouping anchor/link supervision; if a different dataset shares targets across anchors, group the connected components before using this split. Exact signatures do not capture all near-duplicate or family relationships. Add a stricter near-duplicate split before interpreting small gains as generalization.

## Geography

`train(..., exclude_country="India")` gives India candidates zero training weight and excludes India from probability calibration and threshold tuning. When language mappings use labels, rebuild the language artifact, index and pairs without India labels first; the trainer rejects a leaking cache. Reverse US/India for the second stress test. Unlabeled target text remains available. The notebook implements both rebuilds and leaves trust closed until `EVALUATE_TRUST=True`.

France is automatically indexed and predicted in full test mode. It has no training labels; no French validation score is claimed. Preserve raw Unicode, separate accent folding from canonical text, and examine candidate/score distributions for drift.

## Decision policy

The default policy combines country/source-conditioned pair thresholds, a top-score threshold and a learned anchor-level gate based on scores, margins, candidate count, agreement and missingness. There is no one-match-per-source constraint. The optional utility decoder approximates missing links and independent candidate outcomes; it must be compared on calibration before adoption. Missing retrieval evidence can still be mistaken for a singleton.

The fitted classifier uses anchor-normalized sample weights. Its sigmoid calibration is an empirical score transformation under that weighting, not a guarantee of perfectly calibrated marginal probabilities. Selection is evaluated end to end on the frozen trust partition.

Alias mining is bound to the exact training split. Optional OOF score generation rejects label-mined language caches because those mappings must be refitted per OOF fold. Use a separate no-language index for that experiment until nested language/retrieval OOF is implemented. The optional ambiguity split groups sorted suffix-reduced names across addresses; it is a stress test, not exhaustive semantic clustering.

## Honest reporting

Synthetic-demo scores only prove software behavior. Do not report them as challenge accuracy. Indexing the full target pool and testing a synthetic fixture do not establish full AWS throughput. Retain candidate-recall metrics and runtime with every real-data experiment. Do not select repeatedly on the trust set or public leaderboard.
