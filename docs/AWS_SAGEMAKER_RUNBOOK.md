# Running the project on AWS SageMaker Studio

This guide starts from the state shown in the AWS setup screenshot: the
SageMaker domain and user profile are already `Ready` in `us-east-1`. The first
run stays on CPU and uses the training files only. Do not choose a GPU instance
for the current pipeline. SQLite/FTS retrieval, feature construction, and
scikit-learn histogram gradient boosting do not gain useful acceleration from a
GPU.

## 1. Put the private dataset in S3

The dataset must never be committed to GitHub. Create a private S3 bucket in the
same region as Studio (`us-east-1`) and upload `student_resource.zip` to it.

1. Open the AWS Console and confirm **US East (N. Virginia) / us-east-1** in the
   top-right region menu.
2. Search for **S3**, open it, and choose **Create bucket**.
3. Use a globally unique name, for example `kanishka-amazon-ml-2026-<random>`.
4. Leave **Block all public access** enabled. Leave ACLs disabled.
5. Create the bucket, open it, create a folder named `input`, and upload
   `student_resource.zip` into that folder. Using a folder is optional, but it
   makes the URI below and later cleanup easier to follow.
6. Copy the object URI. It will look like:

   ```text
   s3://kanishka-amazon-ml-2026-abc123/input/student_resource.zip
   ```

The SageMaker execution role must have `s3:GetObject` for this object. If the
notebook later uploads the final package, it also needs `s3:PutObject` for the
chosen output prefix. Do not create access keys or paste credentials into the
notebook; Studio supplies temporary credentials through its execution role.

## 2. Create the JupyterLab space

1. Search for **Amazon SageMaker AI** in the AWS Console.
2. Open **SageMaker Studio** under **Applications and IDEs**.
3. Open the existing `Ready` user profile.
4. In Studio, select **JupyterLab** and then **Create JupyterLab space**.
5. Name it `amazon-ml-2026` and keep it private.
6. For the first demo, use a small CPU instance. For the real training pilot,
   use `ml.m5.2xlarge` (8 vCPU, 32 GiB RAM) if that type is available in the
   account. Do not begin with the large profile.
7. Set the EBS storage to **100 GB** for the pilot. The default 5 GB is too small
   for the expanded TSV data, SQLite indexes, feature arrays, stress-test
   indexes, and output shards. If the measured pilot suggests the full run needs
   more than 100 GB, stop and resize the space or request a higher domain storage
   limit before the large run.
8. Use the latest SageMaker Distribution CPU image and choose **Run space**,
   followed by **Open JupyterLab**.

Instance choice is deliberately staged:

| Stage | Suggested compute | Purpose |
| --- | --- | --- |
| Demo | the smallest available CPU instance | Confirm the environment and notebook flow |
| Full-data pilot | `ml.m5.2xlarge` | Index all training targets and train on 1,000 anchors per country |
| Large profile | consider `ml.m5.4xlarge` only after measuring the pilot | More RAM and CPU for a much larger feature cache |

Availability and charges vary by account and region. Check the SageMaker pricing
page before changing instance type. With a $100 balance, run the demo and pilot,
inspect time/disk/RAM, and then decide whether the large profile is affordable.

## 3. Clone the repository and install the environment

Open **File > New > Terminal** in JupyterLab and run:

```bash
cd /home/sagemaker-user
git clone https://github.com/kanishkaanarang/my-lovely-bottle.git
cd my-lovely-bottle

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-aws.txt
python -m ipykernel install --user --name ber-2026 --display-name "Python (BER 2026)"
```

Verify the code before spending time on the real data:

```bash
python -m unittest discover -s tests -v
```

All tests should end with `OK`. If installation fills the disk, stop the space
and increase its storage; do not delete random files from the project.

## 4. Run the offline demo first

1. In JupyterLab, open `notebooks/amazon_ml_2026.ipynb`.
2. Select **Kernel > Change Kernel > Python (BER 2026)**.
3. Leave `MODE` as `demo`.
4. Choose **Kernel > Restart Kernel and Run All Cells**.
5. Confirm that the notebook generates demo data, builds indexes, trains the
   model, validates predictions, and reaches the final cell without an error.

The demo is synthetic and small. Its score is a software check, not a measure of
competition performance.

## 5. Configure the training-only pilot

In the notebook configuration cell, change only the following values:

```python
MODE = "full"
TRAIN_ONLY = True
PROFILE = "pilot"

S3_INPUT_URI = "s3://YOUR-BUCKET/input/student_resource.zip"
S3_OUTPUT_URI = "s3://YOUR-BUCKET/output/team_submission.zip"

EVALUATE_TRUST = False
RUN_FULL_TEST = False
PACKAGE_SUBMISSION = False
UPLOAD_PACKAGE_TO_S3 = False
```

Keep these defaults for the first real-data run:

```python
PER_COUNTRY = 1000
MAX_TRAINING_PAIRS = 700_000
USE_MINED_LANGUAGE = True
RUN_GEOGRAPHIC_STRESS = True
```

Use a new `RUN_NAME` whenever model, sample, retrieval, or feature settings
change. Cached artifacts are fingerprinted, and the code intentionally refuses
to combine incompatible indexes, pairs, and models.

Restart the kernel and run the notebook cells in order. In training-only mode it
will:

1. download the ZIP with the Studio execution role;
2. extract and audit only the four training files;
3. learn training-only aliases and Indic-to-Latin character mappings;
4. index every S2/S3 training target in country-specific SQLite databases;
5. sample 1,000 S1 anchors per country and create candidate pairs;
6. train and calibrate the pair classifier;
7. create leakage-safe geographic stress models; and
8. leave the official test files unopened.

The longest pilot stages are indexing S2/S3, producing feature pairs, and the
two geographic stress rebuilds. Notebook output prints progress and artifact
locations. Work is stored under `/home/sagemaker-user/my-lovely-bottle/work`, on
the space's EBS volume, so completed compatible stages resume after a kernel or
space restart.

## 6. Inspect the pilot before scaling up

Check these items before changing `PROFILE` to `large`:

- the free-disk value printed in the configuration cell;
- `work/full/v3/runs/robust_v3_pilot/pairs_manifest.json` for pair counts,
  channel recall, and preparation time;
- the model `.metrics.json` file for calibration/tuning metrics;
- the run registry under `work/full/v3/registry`;
- whether the process approaches the instance RAM limit; and
- whether the pilot leaves enough credit for test indexing and inference.

Do not interpret the pilot score as the final expected leaderboard score. It
uses a sampled anchor set, although retrieval is measured against the complete
training target index.

## 7. Run a larger training experiment only after the pilot

Stop the JupyterLab space before changing its instance type. If the pilot shows
that more capacity is necessary, choose a larger CPU instance and restart it.
Then set:

```python
PROFILE = "large"
RUN_NAME = "robust_v3_large_01"
```

The large profile requests 200,000 anchors per country and permits up to 140
million candidate pairs. That is a capacity ceiling, not a promise that the run
fits a specific machine. Watch disk and memory during the first pair-generation
stage. If it approaches the limit, stop, preserve the pilot, and reduce the
sample or candidate budget in a new run.

## 8. Test inference and final submission run

Only do this after model selection is finished and the competition test data may
be opened. In a fresh run name, set:

```python
TRAIN_ONLY = False
RUN_FULL_TEST = False
BENCHMARK_ANCHORS = 3000
```

Run through the test-index and benchmark cells. The notebook samples across
countries, reports candidates and top-score distributions, and estimates total
inference time with a safety margin. If the estimate and remaining credit are
acceptable, enable:

```python
RUN_FULL_TEST = True
PACKAGE_SUBMISSION = True
UPLOAD_PACKAGE_TO_S3 = True
```

Run the inference, validation, and packaging cells. Packaging succeeds only
after strict validation and the supplied official validator pass against the
exact output files. The final archive is saved in the run directory and, when
enabled, uploaded to `S3_OUTPUT_URI`.

Download the result from S3 and submit it through the competition portal. The
repository deliberately does not automate portal submission.

## 9. Stop charges every time

When finished for the day:

1. Save the notebook.
2. In Studio, return to **JupyterLab**, select `amazon-ml-2026`, and choose
   **Stop space**. Closing the browser tab does not stop compute.
3. Confirm that the space no longer shows `Running`.
4. Keep the space only while its EBS artifacts are useful. Stopping compute does
   not remove EBS storage charges.
5. Before deleting a space, copy valuable artifacts to S3. Deleting the space
   deletes its EBS volume.
6. Check **Billing and Cost Management > Cost Explorer** and the `$10` budget
   alert after each major run.

AWS references: [create a JupyterLab space](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-create-space.html),
[configure a space](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-user-guide-configure-space.html),
[JupyterLab EBS and compute model](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-admin-guide.html),
and [delete unused resources](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl-admin-guide-clean-up.html).
