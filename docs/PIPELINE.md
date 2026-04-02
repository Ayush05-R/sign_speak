# Pipeline Guide

This guide covers data preparation, vector extraction, and training for the static sign model.

## Quick Pipeline (Python)

```bash
python scripts/simple_pipeline.py
```

Key flags:
```bash
--input-dir data/raw/static
--max-per-class 500
--min-det-conf 0.3 --min-pres-conf 0.3
--skip-xgb
--skip-eval
```

## 1) Vector Extraction

**From folder-per-class images (recommended):**
```bash
python -m ml.pipeline.data_collection.images_to_vectors --input-dir data/raw/static --output data/processed/static/static_data.csv
```

**From train/test split:**
```bash
python -m ml.pipeline.data_collection.images_to_vectors --input-dir data/processed/static --output-dir data/processed/static/hand_vectors
```

Helpful flags:
```bash
--min-det-conf 0.3 --min-pres-conf 0.3 --min-side 256
--normalize-rotation --mirror-left
--log-every 500 --quiet
```

If hands are missed often, try:
```bash
--retry-min-side 384
```

## 2) Train

```bash
python -m ml.pipeline.training.train_static --data data/processed/static/static_data.csv
```

Faster training:
```bash
python -m ml.pipeline.training.train_static --skip-xgb
```

Quiet logs:
```bash
python -m ml.pipeline.training.train_static --quiet
```

## 3) Evaluate on Images

```bash
python -m ml.pipeline.inference.eval_static_images --input-dir data/raw/static
```

## 4) Live Inference

All-in-one live view (prediction + sentence):
```bash
python -m ml.pipeline.inference.run_sentence_builder
```

If your signs require both hands:
```bash
python -m ml.pipeline.inference.run_sentence_builder --require-two-hands
```

