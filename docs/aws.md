# Running on AWS

Use an existing SageMaker Studio JupyterLab space or an EC2 instance with Jupyter and a persistent EBS volume. This project creates no infrastructure. The first implementation uses CPU search and tree fitting; a GPU does not accelerate its SQLite index.

1. Clone the repository into your notebook workspace.
2. Install `requirements.txt` in the selected Python kernel. Install `requirements-aws.txt` only for optional S3 transfers. Restart the kernel after installation.
3. Run the notebook in demo mode. Check that training, trust evaluation, prediction and strict validation finish.
4. Put the supplied ZIP in a private S3 bucket or upload it to persistent disk. The role needs read access to the chosen input object and, only if used, write access to the chosen output object. KMS-encrypted objects may need the corresponding key permissions. Follow your account's policies rather than embedding access keys.
5. Set full mode and input/work paths. Index artifacts, model caches, prediction shards and exports all live under `WORK_ROOT`. Use local EBS-backed storage for SQLite, not an S3 mount. Reserve disk for indexes and candidate files, which can greatly exceed input size.
6. Choose a CPU/memory/storage configuration you can benchmark. Start with a bounded training sample and observe memory, disk, index time and candidate tails before scaling. No full-scale instance-size or cost estimate is asserted here.
7. Run the inference probe and compare the buffered estimate with your remaining runtime. Keep a fallback artifact. Increase concurrency only after measuring; the baseline has one index writer and one inference writer per directory.
8. Complete methodology, run strict validation, package, and optionally upload the final ZIP to your S3 location. Upload `matching_results.tsv` to the competition portal yourself.
9. Stop paid compute when finished. Preserve artifacts on durable storage first.

## Interruptions

- **Index creation:** a completed fingerprinted index is reused; an unfinished index rebuilds. Do not move or edit index files during a run.
- **Feature preparation:** completed pair manifests can be reused. An incomplete preparation stage restarts. Changing sample, seed or retrieval requires a new run directory.
- **Prediction:** each batch is written to a temporary directory, then renamed only after both TSV fragments and their checksums exist. Reruns reuse verified batches. Change model, source data, batch size or benchmark limit only in a new output directory.
- **Benchmark:** a resumed run's elapsed time is not a fresh throughput measurement. Use a new benchmark directory when measuring another instance or configuration.

Optional S3 package upload does not checkpoint every shard to S3. Persistent disk is therefore essential; if you use disposable/spot infrastructure, arrange your own durable artifact backups or rerun from durable completed stages.

References: [SageMaker Studio policies](https://docs.aws.amazon.com/sagemaker/latest/dg/scheduled-notebook-policies-studio.html), [Boto3 IAM role credential chain](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html), [S3 upload behavior](https://docs.aws.amazon.com/boto3/latest/guide/s3-uploading-files.html).
