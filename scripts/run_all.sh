#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="data/raw/static"
OUTPUT_CSV="data/processed/static/static_data.csv"
OUTPUT_DIR="data/processed/static/hand_vectors"
MIN_DET_CONF="0.3"
MIN_PRES_CONF="0.3"
MIN_SIDE="256"
MAX_SIDE="0"
SKIP_EVAL="0"
RUN_WEBCAM="0"
RUN_SENTENCE="0"
SKIP_XGB="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-csv) OUTPUT_CSV="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --min-det-conf) MIN_DET_CONF="$2"; shift 2 ;;
    --min-pres-conf) MIN_PRES_CONF="$2"; shift 2 ;;
    --min-side) MIN_SIDE="$2"; shift 2 ;;
    --max-side) MAX_SIDE="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL="1"; shift 1 ;;
    --run-webcam) RUN_WEBCAM="1"; shift 1 ;;
    --run-sentence) RUN_SENTENCE="1"; shift 1 ;;
    --skip-xgb) SKIP_XGB="1"; shift 1 ;;
    -h|--help)
      echo "Usage: $0 [--input-dir DIR] [--output-csv FILE] [--output-dir DIR] [--min-det-conf N] [--min-pres-conf N] [--min-side N] [--max-side N] [--skip-eval] [--run-webcam] [--run-sentence] [--skip-xgb]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "InputDir not found: $INPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "mediapipe/hand_landmarker.task" ]]; then
  echo "Missing mediapipe/hand_landmarker.task. Download it first." >&2
  exit 1
fi

HAS_SPLIT="0"
if [[ -d "$INPUT_DIR/train" && -d "$INPUT_DIR/test" ]]; then
  HAS_SPLIT="1"
fi

echo "Step 1/4: Build vectors from images"
if [[ "$HAS_SPLIT" == "1" ]]; then
  python -m ml.pipeline.data_collection.images_to_vectors \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --min-det-conf "$MIN_DET_CONF" \
    --min-pres-conf "$MIN_PRES_CONF" \
    --min-side "$MIN_SIDE" \
    --max-side "$MAX_SIDE"
  TRAIN_DATA="$OUTPUT_DIR"
else
  python -m ml.pipeline.data_collection.images_to_vectors \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT_CSV" \
    --min-det-conf "$MIN_DET_CONF" \
    --min-pres-conf "$MIN_PRES_CONF" \
    --min-side "$MIN_SIDE" \
    --max-side "$MAX_SIDE"
  TRAIN_DATA="$OUTPUT_CSV"
fi

echo "Step 2/4: Train model"
if [[ "$SKIP_XGB" == "1" ]]; then
  python -m ml.pipeline.training.train_static --data "$TRAIN_DATA" --skip-xgb
else
  python -m ml.pipeline.training.train_static --data "$TRAIN_DATA"
fi

if [[ "$SKIP_EVAL" == "0" ]]; then
  echo "Step 3/4: Evaluate on images"
  python -m ml.pipeline.inference.eval_static_images \
    --input-dir "$INPUT_DIR" \
    --min-det-conf "$MIN_DET_CONF" \
    --min-pres-conf "$MIN_PRES_CONF" \
    --min-side "$MIN_SIDE" \
    --max-side "$MAX_SIDE"
else
  echo "Step 3/4: Skipped eval"
fi

if [[ "$RUN_WEBCAM" == "1" ]]; then
  echo "Step 4/4: Live webcam inference (press Q to quit)"
  python -m ml.pipeline.inference.run_static
else
  echo "Step 4/4: Skipped webcam (use --run-webcam to enable)"
fi

if [[ "$RUN_SENTENCE" == "1" ]]; then
  echo "Step 5/5: Live sentence builder (press Q to quit)"
  python -m ml.pipeline.inference.run_sentence_builder
else
  echo "Step 5/5: Skipped sentence builder (use --run-sentence to enable)"
fi
