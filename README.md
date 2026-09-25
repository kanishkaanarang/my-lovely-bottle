# Business entity resolution · Amazon ML Challenge 2026

A portable Jupyter notebook and a reusable Python pipeline for matching every Source 1 business to zero, one or many records in Sources 2 and 3. Team identity and cloud-specific settings are intentionally left unset.

**Start here: [notebooks/amazon_ml_2026.ipynb](notebooks/amazon_ml_2026.ipynb).**

The notebook defaults to a complete offline demo. Real-data runs use all target records for retrieval and a bounded, country-stratified sample of training anchors for supervised fitting. This is a tested implementation baseline, not a claim of competition-winning accuracy or full-scale AWS throughput.

## Run locally or on AWS

Use Python 3.11–3.13, SQLite with FTS5, and persistent local disk. No GPU is required.

```bash
git clone https://github.com/kanishkaanarang/my-lovely-bottle.git
cd my-lovely-bottle
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name business-er --display-name "Business ER"
python -m unittest discover -s tests -v
```

Open the notebook in your existing Jupyter/SageMaker environment and select this kernel. JupyterLab itself is supplied by SageMaker or your own environment. To use S3, install `requirements-aws.txt`. After installation, save `python -m pip freeze > work/environment.lock.txt` with the private run artifacts to capture all transitive and optional AWS dependencies.

Run demo mode first. For the competition dataset, set `MODE="full"`, `DATA_ROOT` to the extracted `dataset/` directory, and a persistent `WORK_ROOT`. Set team details, training sample size and candidate budgets in the configuration cell. Benchmark first, then enable `RUN_FULL_TEST`.

No AWS infrastructure is provisioned, no instances are started, and no competition submission is uploaded by this repository. Optional S3 transfers use your existing IAM role. Never put credentials or private dataset files in GitHub.

## What is implemented

- **Retrieval:** exact normalized name/address, suffix-reduced names, BM25 name/address search and name character trigrams. Countries are discovered from the data; France is not filtered out. Ambiguous anchors get a larger candidate budget.
- **Matching:** histogram gradient boosting on string, numeric, missingness, source and retrieval-context features. Training negatives come from the real retrieval procedure; no true matches are injected into candidate lists.
- **Validation:** deterministic signature-grouped training/calibration/trust splits. Probability calibration and threshold/empty-gate selection use separate halves of the calibration partition. Trust metrics include macro F0.5, per-country results, singleton false merges, candidate tails and the oracle blocking ceiling.
- **Geographic stress:** optional models excluding US or India labels from training and calibration, evaluated on that country's trust anchors.
- **Inference:** memory-bounded batches, atomic completed shards, checksummed resume and matching/candidate TSVs generated together.
- **Submission:** streaming, disk-backed strict validator; both files must contain every test anchor and predictions must be a subset of scored candidates. Packaging includes source, fitted model, requirements and completed methodology.

## Why this shape

The supplied-data audit found 24.23 million records, up to 11 matches per anchor, 5.58% training singletons, and France only in test. On a diagnostic 11,090-anchor sample, exact-name-or-address blocking recovered only 28.8% of true links. Those are **prior EDA findings**, not the score of this implementation. They motivate multiple retrieval channels, careful negative examples, one-to-many output and country stress tests.

F0.5 is calculated per anchor then averaged. For nonempty truth, `1.25 * TP / (0.25 * len(truth) + len(prediction))`; an empty true set scores 1 only for an empty prediction. Higher is better. Never substitute pair F1 or overall pair accuracy.

## Scale and limits

The full target pool is indexed on disk. Country-balanced reservoir sampling bounds supervised training memory; feature matrices are memory mapped. Set `max_pairs` explicitly for larger runs. Index build and pair preparation restart their unfinished stage; inference resumes at completed batches. Run only one writer per artifact directory.

SQLite lexical search is a straightforward CPU baseline. Full 10-million-target search, index size, and 1.7-million-anchor inference time have not yet been measured on AWS. Do not assume a GPU accelerates this pipeline. Benchmark on your actual CPU, memory and storage; lower candidate budgets only after measuring recall loss.

Unicode marks are preserved and a separate Latin-accent-folded view is used. **This is not transliteration or a multilingual embedding model.** Cross-script names currently depend largely on address retrieval. Learned aliases/transliteration, multilingual retrieval, near-duplicate clustering, richer set utility and graph competition are documented next experiments, not implemented features. The current split groups exact normalized signatures, not all semantically related businesses.

No pretrained model or external business database is used. Code and newly trained artifacts are intended for the MIT-licensed project; dependencies retain their respective licenses. Review the current competition rules before the final package. The model is a small locally fitted tree ensemble, not a foundation model.

## CLI

```bash
python -m ber demo --data work/demo/data
python -m ber index --data work/demo/data --split train --index work/demo/index_train
python -m ber pairs --data work/demo/data --index work/demo/index_train --run work/demo/run --per-country 160
python -m ber train --run work/demo/run
python -m ber evaluate --run work/demo/run --model work/demo/run/model.pkl
python -m ber index --data work/demo/data --split test --index work/demo/index_test
python -m ber predict --data work/demo/data --index work/demo/index_test --model work/demo/run/model.pkl --output work/demo/output
python -m ber validate --data work/demo/data --index work/demo/index_test --output work/demo/output
```

Use `python -m ber --help` for packaging and other options. Only load model pickle files you produced and trust.

## Files

| Path | Purpose |
| --- | --- |
| `notebooks/amazon_ml_2026.ipynb` | Guided AWS/local execution and configuration |
| `ber/` | Retrieval, features, training, evaluation, inference, validation and S3 transport |
| `tests/` | Metric/Unicode tests and complete synthetic pipeline/resume checks |
| `Documentation_template.md` | Your required methodology; fill before packaging |
| `docs/aws.md` | AWS execution and storage checklist |
| `docs/validation.md` | Evaluation interpretation and leakage limits |

Notebook outputs are intentionally empty in Git. Keep executed notebooks, source IDs, raw cases, data and all run artifacts private.

Software references: [SQLite FTS5](https://www.sqlite.org/fts5.html), [scikit-learn classifier](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [Boto3 credentials](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html).
