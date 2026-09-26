# Implementation verification

## Robustness revision (26 September 2026)

Nine automated test methods passed in a newly created Python 3.13 environment
installed from the project requirements. Coverage includes optional ranking,
OOF/reranking, isotonic calibration, grouped early stopping and bootstrap bagging;
parallel/serial output equality and resume; literal TSV/Unicode edge cases;
training-only language guards; and extraction of a submission followed by a fresh
index build and inference from its packaged source and language artifact.

All 12 notebook code cells passed in that clean environment, including separate
US/India geographic-stress builds. The supplied official validator also passed
with `--check-ids` on the synthetic notebook outputs (36 anchors). This does not
validate a real competition submission.

The reattached archive was used only on its training side. A training-only
compatibility pilot used 606 actual anchors and a reduced target pool. See
[improvement_status.md](improvement_status.md) for measured evidence and the
remaining full-scale/organizer-dependent checks. Earlier baseline evidence below
is retained as history, not a claim about this revision's competition accuracy.

Verified locally on 25 September 2026 using Python 3.13, NumPy 2.1.2, scikit-learn 1.8.0, RapidFuzz 3.14.1, nbformat 5.10.4, nbclient 0.10.0 and ipykernel 6.29.5.

- All 12 executable notebook cells completed in a real Jupyter kernel in demo mode, including training, probability calibration, threshold selection, trust evaluation, prediction and strict validation.
- Unit/integration tests cover the macro metric's singleton and multi-match cases, Indic Unicode combining marks, Latin accent folding, signature-grouped splits, unsafe ZIP rejection, training-to-prediction flow, unseen-country output, stable resume, malformed output rejection, changed-configuration rejection, benchmark-output rejection and final package structure.
- A separate real-data compatibility pilot used 500 training anchors and 21,718 target records (their true targets plus a reduced distractor pool). It completed indexing, feature generation, fitting, calibration and trust evaluation. This reduced pool is **not** a valid estimate of competition accuracy or full-data retrieval quality. No pilot data, labels, model or per-record outputs are committed.
- Published notebook code cells have no saved outputs or execution counts.

Not verified: provisioning or running on AWS; S3 transfers with your IAM role; complete 10-million-target index performance; full test inference time; French accuracy; trained multilingual retrieval; leaderboard acceptance. The notebook exposes the full-data benchmark and validation steps needed to establish these.
