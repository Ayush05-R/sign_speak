# Usage Manual

This guide lists how to run each part of the project and the parameters you can pass.

## Quickstart (Python)

```bash
python scripts/simple_pipeline.py
```

Common options:

```bash
--input-dir <DIR>
--output-csv <FILE>
--output-dir <DIR>
--max-per-class <INT>
--min-det-conf <FLOAT>
--min-pres-conf <FLOAT>
--min-side <INT>
--max-side <INT>
--skip-xgb
--skip-eval
--run-static / --no-run-static
--run-sentence / --no-run-sentence
--detector-max-side <INT>
--frame-skip <INT>
```

---

## Quickstart (Git Bash)

```bash
bash scripts/run_all.sh
```

Options:

```bash
--input-dir <DIR>
--output-csv <FILE>
--output-dir <DIR>
--min-det-conf <FLOAT>
--min-pres-conf <FLOAT>
--min-side <INT>
--max-side <INT>
--skip-eval
--run-webcam
--run-sentence
--skip-xgb
```

Example:

```bash
bash scripts/run_all.sh --input-dir data/raw/static --run-webcam --run-sentence
```

---

## 1) Build Hand Vectors (Images -> CSV)

```bash
python -m ml.pipeline.data_collection.images_to_vectors
```

Common parameters:

```bash
--input-dir <DIR>
--output <FILE>
--output-dir <DIR>
--max-images <INT>
--max-per-class <INT>
--model-path <FILE>
--running-mode IMAGE|VIDEO
--max-hands <INT>
--min-det-conf <FLOAT>
--min-pres-conf <FLOAT>
--min-track-conf <FLOAT>
--normalize-rotation / --no-normalize-rotation
--mirror-left / --no-mirror-left
--min-side <INT>
--max-side <INT>
--retry-min-side <INT>
--log-every <INT>
--quiet
--extensions <CSV>
--seed <INT>
```

Example:

```bash
python -m ml.pipeline.data_collection.images_to_vectors \
  --input-dir data/raw/static \
  --output data/processed/static/static_data.csv \
  --min-det-conf 0.3 --min-pres-conf 0.3 \
  --log-every 500
```

---

## 2) Train Static Model

```bash
python -m ml.pipeline.training.train_static --data data/processed/static/static_data.csv
```

Parameters:

```bash
--data <FILE|DIR>
--train-data <FILE|DIR>
--test-data <FILE|DIR>
--out-dir <DIR>
--test-size <FLOAT>
--seed <INT>
--quiet
--skip-xgb
--rf-n-estimators <INT>
--rf-max-depth <INT>
--rf-min-samples-leaf <INT>
--rf-n-jobs <INT>
--rf-class-weight <balanced|none>
--use-scaler / --no-use-scaler
--xgb-n-estimators <INT>
--xgb-max-depth <INT>
--xgb-learning-rate <FLOAT>
--xgb-subsample <FLOAT>
--xgb-colsample-bytree <FLOAT>
--xgb-n-jobs <INT>
```

Example:

```bash
python -m ml.pipeline.training.train_static --data data/processed/static/static_data.csv --skip-xgb
```

---

## 3) Evaluate on Images

```bash
python -m ml.pipeline.inference.eval_static_images --input-dir data/raw/static
```

Parameters:

```bash
--input-dir <DIR>
--model-path <FILE>
--labels-path <FILE>
--hand-model-path <FILE>
--running-mode IMAGE|VIDEO
--max-hands <INT>
--min-det-conf <FLOAT>
--min-pres-conf <FLOAT>
--min-track-conf <FLOAT>
--normalize-rotation / --no-normalize-rotation
--mirror-left / --no-mirror-left
--min-side <INT>
--max-side <INT>
--max-test <INT>
--log-every <INT>
--quiet
--extensions <CSV>
--seed <INT>
```

---

## 4) Live Webcam Inference

```bash
python -m ml.pipeline.inference.run_static
```

Parameters:

```bash
--model-path <FILE>
--labels-path <FILE>
--hand-model-path <FILE>
--camera <INT>
--width <INT>
--height <INT>
--no-flip
--draw-landmarks
--show-fps
--frame-skip <INT>
--history <INT>
--stable-frames <INT>
--clear-frames <INT>
--ema-alpha <FLOAT>
--detector-max-side <INT>
--max-hands <INT>
--min-det-conf <FLOAT>
--min-pres-conf <FLOAT>
--min-track-conf <FLOAT>
--normalize-rotation / --no-normalize-rotation
--mirror-left / --no-mirror-left
--require-two-hands / --no-require-two-hands
--run-sentence-builder / --no-run-sentence-builder
```

---

## 5) Live Sentence Builder (All-in-one)

```bash
python -m ml.pipeline.inference.run_sentence_builder
```

Parameters:

```bash
--model-path <FILE>
--labels-path <FILE>
--hand-model-path <FILE>
--camera <INT>
--width <INT>
--height <INT>
--no-flip
--draw-landmarks
--show-fps
--frame-skip <INT>
--max-hands <INT>
--min-det-conf <FLOAT>
--min-pres-conf <FLOAT>
--min-track-conf <FLOAT>
--normalize-rotation / --no-normalize-rotation
--mirror-left / --no-mirror-left
--require-two-hands / --no-require-two-hands
--ema-alpha <FLOAT>
--detector-max-side <INT>
--history <INT>
--stable-frames <INT>
--clear-frames <INT>
--cooldown-frames <INT>
--fade-frames <INT>
--space-label <LABEL>
--fullstop-label <LABEL>
```

---

## 6) Live Dataset Capture

```bash
python -m ml.pipeline.data_collection.collect_dataset_live
```

Parameters:

```bash
--output-dir <DIR>
--num-images <INT>
--delay-ms <INT>
--flip
--label <LABEL>
--labels-file <FILE>
```

Examples:

```bash
# Capture a single label and exit
python -m ml.pipeline.data_collection.collect_dataset_live --label space

# Capture labels listed in a file (one per line)
python -m ml.pipeline.data_collection.collect_dataset_live --labels-file labels.txt
```

