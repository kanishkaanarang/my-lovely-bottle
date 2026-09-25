# Reproduce final inference

This directory includes the fitted `model.pkl`, `src/ber/`, requirements and an inference configuration. Use only this trusted model artifact; Python pickle can execute code.

With Python 3.11–3.13 and SQLite FTS5:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m ber index --data /path/to/dataset --split test --index work/index_test
python -m ber predict --data /path/to/dataset --index work/index_test --model model.pkl --output output --batch-size 500
python -m ber validate --data /path/to/dataset --index work/index_test --output output
```

Use the batch size recorded in `inference_config.json` if it differs from 500. Both output files will be regenerated from the supplied test data and model. Candidate settings, features, calibration and thresholds are embedded in the fitted artifact. No external business data is required.

To reproduce training instead, run `python -m ber pairs` against a complete training target index, followed by `python -m ber train` and `python -m ber evaluate`. The methodology records the final sampling settings and seed. Keep the test set's France records; they are valid output anchors.
