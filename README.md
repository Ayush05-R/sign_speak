# Sign Speak - Reference Sign Language Translator

Reference implementation for a sign-language translator. This repo focuses on a fast static pipeline built on MediaPipe hand landmarks.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the MediaPipe hand landmarker model (once):
   ```bash
   python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task','mediapipe/hand_landmarker.task'); print('Downloaded')"
   ```

## Quickstart (Git Bash)

Run the full pipeline in one shot (Python, recommended):
```bash
python scripts/simple_pipeline.py
```

Run the Bash pipeline (Git Bash):
```bash
bash scripts/run_all.sh
```

## Manual Workflow

1. Vector extraction:
```bash
python -m ml.pipeline.data_collection.images_to_vectors --input-dir data/raw/static --output data/processed/static/static_data.csv
```

2. Train:
```bash
python -m ml.pipeline.training.train_static --data data/processed/static/static_data.csv
```

3. Evaluate:
```bash
python -m ml.pipeline.inference.eval_static_images --input-dir data/raw/static
```

4. Live inference:
```bash
python -m ml.pipeline.inference.run_static
```

5. Live sentence builder (uses `space` and `full-stop` labels):
```bash
python -m ml.pipeline.inference.run_sentence_builder
```

## Dataset Capture

Create your own dataset from webcam:
```bash
python -m ml.pipeline.data_collection.collect_dataset_live
```

## Project Layout

- `backend/` - backend service scaffold (API, services, schemas)
- `ml/` - ML code and utilities
- `ml/detection/` - hand detector wrapper
- `ml/features/` - feature vector conversion
- `ml/pipeline/` - data collection, training, inference scripts
- `ml/models/` - trained models and label lists
- `scripts/` - `run_all.sh`, `simple_pipeline.py`
- `docs/` - `PIPELINE.md`, `INFERENCE.md`, `DATASET.md`
- `data/raw/` - raw folder-per-class images
- `data/processed/` - processed splits and vector CSVs

## Docs

- `docs/PIPELINE.md` - data prep and training
- `docs/INFERENCE.md` - live inference and sentence builder
- `docs/DATASET.md` - dataset capture flow
- `docs/USAGE.md` - full command/parameter reference
- `docs/NEW_DATASET_PIPELINE.md` - end-to-end new dataset workflow

## Notes

- Use `--log-every` and `--quiet` for cleaner logs during vector extraction and evaluation.
- For missed detections, try `--retry-min-side 384`.
- Live inference can be smoothed with `--history`, `--stable-frames`, and sped up with `--frame-skip`.
- EMA smoothing is available with `--ema-alpha` (0 = off, 0.4 default).
- For higher FPS, use `--detector-max-side 640` in live inference to downscale frames before detection.
- `run_sentence_builder` is the all-in-one live view (prediction + sentence) and supports `--show-fps`, `--draw-landmarks`, `--frame-skip`, `--clear-frames`, `--camera`, `--width`, `--height`, and `--no-flip`.
- `run_static` no longer auto-launches the sentence builder; use `--run-sentence-builder` if you want that chaining.
- For two-hand datasets, pass `--require-two-hands` in live inference to skip one-hand frames.
- Landmark normalization (rotation + left-hand mirroring) is enabled by default; keep training and inference settings consistent. Use `--no-normalize-rotation` or `--no-mirror-left` to disable.
