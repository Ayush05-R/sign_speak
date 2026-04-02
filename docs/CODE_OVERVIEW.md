# Code Overview

This document explains the code in the `mediapipe`, `ml`, and `scripts` directories in simple terms.

## Pipeline At A Glance

1. `images_to_vectors.py` turns images into hand-landmark vectors (CSV).
2. `train_static.py` trains a model from those vectors.
3. `eval_static_images.py` tests the model on images.
4. `run_static.py` does live single-label prediction.
5. `run_sentence_builder.py` does live prediction plus sentence building.

## mediapipe

| Path | What it does |
| --- | --- |
| `mediapipe/hand_landmarker.task` | Pretrained MediaPipe hand landmark model used by the detector. |

## ml

| Path | What it does |
| --- | --- |
| `ml/detection/hand_detector.py` | MediaPipe Tasks wrapper for hand landmarks and handedness. |
| `ml/features/feature_vector.py` | Converts landmarks into a normalized feature vector (rotation and left-hand mirroring supported). |
| `ml/utils/image_utils.py` | Image resize helper for faster or more reliable detection. |
| `ml/models/` | Trained models and label lists (e.g., `static_model.pkl`). |

### ml/pipeline/data_collection

| Path | What it does |
| --- | --- |
| `ml/pipeline/data_collection/images_to_vectors.py` | Reads folder-per-class images, detects hands, converts landmarks into vectors, writes CSV(s). Supports train/test split folders and sampling caps. |
| `ml/pipeline/data_collection/collect_dataset_live.py` | Captures images from webcam into `data/raw/static/<label>/`. Supports `--label` or `--labels-file`. |

### ml/pipeline/training

| Path | What it does |
| --- | --- |
| `ml/pipeline/training/train_static.py` | Trains RandomForest and optionally XGBoost, selects best, saves model + labels. Can split train/test or use provided splits. |

### ml/pipeline/inference

| Path | What it does |
| --- | --- |
| `ml/pipeline/inference/run_static.py` | Live webcam prediction of the current label. Supports smoothing, FPS overlay, and optional chaining to sentence builder. |
| `ml/pipeline/inference/run_sentence_builder.py` | All-in-one live view: prediction + sentence building, cooldown, fade-out on full-stop, FPS overlay, and frame skip. |
| `ml/pipeline/inference/eval_static_images.py` | Tests the trained model on folder-per-class images and prints accuracy/report. |

## scripts

| Path | What it does |
| --- | --- |
| `scripts/run_all.sh` | Bash pipeline: vectors -> train -> eval -> optional webcam and sentence builder. |
| `scripts/simple_pipeline.py` | Python pipeline with more flags and defaults; runs full end-to-end flow. |

## Generated Artifacts

| Pattern | What it is |
| --- | --- |
| `__pycache__/` and `*.pyc` | Python bytecode cache. Safe to delete; should be ignored by git. |
