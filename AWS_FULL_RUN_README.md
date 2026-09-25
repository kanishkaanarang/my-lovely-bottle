# Complete AWS run guide

This guide runs the Amazon ML Challenge 2026 entity-resolution notebook on
Amazon SageMaker Studio JupyterLab. Keep the private dataset in Amazon S3 and
put active SQLite indexes, models, and prediction shards on the JupyterLab
space's EBS disk.

## 1. Choose one AWS Region

Sign in to the [AWS Console](https://console.aws.amazon.com/) and select a
region from the top-right menu. Use that same region for SageMaker Studio and
the S3 bucket.

## 2. Upload the input ZIP to private S3 storage

1. Open the [Amazon S3 console](https://console.aws.amazon.com/s3/).
2. Choose **Create bucket**.
3. Enter a globally unique name, such as
   `amazon-ml-challenge-your-name-2026`.
4. Keep **Block all public access** enabled and leave default encryption
   enabled.
5. Create the bucket, create a `private/` folder, and upload
   `student_resource.zip`.
6. Copy the object's S3 URI. It will resemble:

```text
s3://amazon-ml-challenge-your-name-2026/private/student_resource.zip
```

Never make the bucket public, commit credentials, or add the dataset to Git.

## 3. Create SageMaker Studio

Skip this section if your AWS account already provides SageMaker Studio.

1. Open the [SageMaker AI console](https://console.aws.amazon.com/sagemaker/).
2. Expand **Admin configurations** and choose **Domains**.
3. Choose **Create domain**.
4. Choose **Set up for single user (Quick setup)**.
5. Wait for the domain and user profile to finish provisioning.
6. Choose **Open Studio**.

In an organization-managed account, use the domain and execution role supplied
by the administrator.

## 4. Create the JupyterLab space

1. In Studio, choose **JupyterLab**.
2. Choose **Create JupyterLab space**.
3. Name it `amazon-ml-2026`; a private space is sufficient.
4. Select the latest CPU SageMaker Distribution image.
5. Start with an instance providing approximately 16 vCPUs and 64 GiB RAM:
   `ml.m7i.4xlarge`, `ml.m6i.4xlarge`, or `ml.m5.4xlarge`, subject to
   regional availability and account quotas.
6. Allocate 250–500 GB of EBS storage if your account permits it. If the UI
   applies a lower maximum, ask the domain administrator to raise the space
   storage limit. A 100 GB space can be used for the initial benchmark, but
   full-run storage has not yet been measured.
7. Choose **Run space**, wait for it to become ready, and choose
   **Open JupyterLab**.

A GPU is unnecessary for this CPU-based SQLite search and tree-model pipeline.

## 5. Clone and install the project

Open a JupyterLab terminal and run:

```bash
cd /home/sagemaker-user
git clone https://github.com/kanishkaanarang/my-lovely-bottle.git
cd my-lovely-bottle

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-aws.txt
python -m ipykernel install --user --name business-er --display-name "Business ER"

mkdir -p work
python -m pip freeze > work/environment.lock.txt
```

Run the automated checks before paying for a long job:

```bash
python -m unittest discover -s tests -v
```

The result should end with `OK`.

## 6. Run the offline demonstration

1. Open `notebooks/amazon_ml_2026.ipynb`.
2. Select the **Business ER** kernel.
3. Leave `MODE = "demo"`.
4. Run every cell in order.

The demo must complete indexing, training, calibration, prediction, and strict
validation. Its metrics confirm software behavior; they are not challenge
accuracy estimates.

## 7. Configure the real-data run

Restart the notebook kernel after the demo. In the first code cell, use the
number of vCPUs assigned to the instance:

```python
THREADS = 16
```

Use this configuration in the main configuration cell, replacing the sample
bucket name:

```python
MODE = "full"
TEAM_NAME = ""
TEAM_MEMBERS = ""
RUN_NAME = "baseline_v1"

WORK_ROOT = Path("/home/sagemaker-user/ber-work") / MODE
DATA_ROOT = WORK_ROOT / "data" / "student_resource" / "dataset"

S3_INPUT_URI = (
    "s3://amazon-ml-challenge-your-name-2026/"
    "private/student_resource.zip"
)
S3_OUTPUT_URI = (
    "s3://amazon-ml-challenge-your-name-2026/"
    "results/team_submission.zip"
)

PER_COUNTRY = 5000
MAX_TRAINING_PAIRS = 3_000_000
MAX_ITER = 180
RUN_GEOGRAPHIC_STRESS = False

RUN_FULL_TEST = False
BENCHMARK_ANCHORS = 300
INFERENCE_BUDGET_HOURS = 12.0
ALLOW_RUNTIME_OVERRUN = False

PACKAGE_SUBMISSION = False
UPLOAD_PACKAGE_TO_S3 = False
```

Leaving the team fields blank is acceptable during experiments. Fill them
before producing the final package.

## 8. Download and verify the data

Run the imports, configuration, and input-loading cells. The notebook downloads
the ZIP from S3, extracts it under `WORK_ROOT`, and checks for:

```text
train/train_source1.tsv
train/train_source2.tsv
train/train_source3.tsv
train/train_ground_truth.tsv
test/test_source1.tsv
test/test_source2.tsv
test/test_source3.tsv
```

If AWS returns `AccessDenied`, the SageMaker execution role needs
`s3:GetObject` for the input object. It also needs `s3:PutObject` for the
results prefix if the final package will be uploaded to S3. Use the role rather
than access keys in the notebook.

## 9. Build the training index

Run the training-index cell. It indexes all Source 2 and Source 3 training
targets under:

```text
/home/sagemaker-user/ber-work/full/index_train
```

A completed fingerprinted index is reused. An interrupted, incomplete index is
rebuilt. Never run two writers against the same index directory.

## 10. Prepare pairs, train, and evaluate

Run the pair-preparation, model-training, and trust-evaluation cells in order.
The default full configuration samples up to 5,000 Source 1 anchors per country
while retaining the complete target pool for retrieval.

The model is saved at:

```text
/home/sagemaker-user/ber-work/full/runs/baseline_v1/model.pkl
```

Record and review:

- trust macro F0.5 and country-weighted F0.5;
- per-country metrics;
- true-link and complete-set retrieval recall;
- oracle blocking ceiling;
- singleton false merges;
- candidate-count tails; and
- the displayed worst trust anchors.

If pair preparation exhausts memory, choose a new `RUN_NAME` and reduce
`MAX_TRAINING_PAIRS`. Do not reduce retrieval budgets without measuring the
resulting recall loss.

## 11. Build the test index and benchmark inference

Run the test-index and benchmark cell while keeping:

```python
RUN_FULL_TEST = False
```

The test index includes all observed countries, including France. The benchmark
processes 300 anchors and reports measured anchors per second plus an estimated
full inference duration with a 35% buffer.

The estimate excludes index construction, training, validation, packaging, and
upload time. If it is too slow, stop the JupyterLab application, select a
larger CPU instance, and restart the same space. The EBS files remain in the
space.

Use a new benchmark directory when timing another machine:

```python
benchmark_dir = RUN_DIR / "benchmark_m7i_8xlarge"
```

Reusing completed benchmark shards would make the new timing misleading.

## 12. Run complete inference

After accepting the benchmark estimate, change:

```python
RUN_FULL_TEST = True
```

Keep `ALLOW_RUNTIME_OVERRUN = False` and set
`INFERENCE_BUDGET_HOURS` to the maximum expected duration. Run the
full-inference cell.

Prediction is written in verified atomic batches. Reconnect and rerun the same
cell after an interruption; completed batches are reused. Do not change the
dataset, model, retrieval configuration, batch size, or output directory during
a resumed run. Use a new `RUN_NAME` for different settings.

## 13. Require strict validation to pass

The full-inference cell invokes the validator. Continue only when it returns:

```python
{"status": "PASS", ...}
```

The principal output files are:

```text
/home/sagemaker-user/ber-work/full/runs/baseline_v1/output/matching_results.tsv
/home/sagemaker-user/ber-work/full/runs/baseline_v1/output/candidate_pairs.tsv
```

Validation checks anchor completeness and order, duplicate IDs, target
existence, country consistency, and that predictions are a subset of the exact
candidate set scored by the model.

## 14. Complete the methodology

Open `Documentation_template.md` and replace every `[FILL ...]` marker with
the final run's team information, configurations, retrieval metrics, validation
metrics, runtimes, disk usage, hashes, limitations, and error analysis. Also
fill `TEAM_NAME` and `TEAM_MEMBERS` in the notebook.

The packager rejects an incomplete methodology.

## 15. Package and back up the submission

After validation passes and the methodology is complete, set:

```python
PACKAGE_SUBMISSION = True
UPLOAD_PACKAGE_TO_S3 = True
```

Run the final cell. The local package is:

```text
/home/sagemaker-user/ber-work/full/runs/baseline_v1/team_submission.zip
```

With uploading enabled, the same package is copied to the exact
`S3_OUTPUT_URI`. Preserve the model, metrics, experiment record, validation
record, both TSV files, environment lock, methodology, and final ZIP.

Submit the file required by the current competition portal instructions. This
repository does not upload anything to the competition portal.

## 16. Stop paid compute safely

1. Return to SageMaker Studio.
2. Open **Running instances**.
3. Find the `amazon-ml-2026` JupyterLab application.
4. Choose **Stop**.

Stopping the application stops its compute instance while retaining files in
the space. Deleting the space deletes its associated EBS data, so delete it
only after all required artifacts are safely backed up.

## Operating rules

- Use S3 for the private input and durable backups.
- Use EBS-backed local paths for SQLite indexes and active run artifacts.
- Do not run SQLite from an S3-mounted directory.
- Run only one writer per index, run, or output directory.
- Benchmark actual hardware before enabling full inference.
- Never report demo or reduced-pilot metrics as competition accuracy.

## AWS references

- [SageMaker quick setup](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)
- [Create a JupyterLab space](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-create-space.html)
- [SageMaker JupyterLab guide](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide.html)
- [Stop Studio applications](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html)
- [Amazon S3 getting started](https://docs.aws.amazon.com/AmazonS3/latest/userguide/GetStartedWithS3.html)
