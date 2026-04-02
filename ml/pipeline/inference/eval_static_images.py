"""
Evaluate the trained static model on images from folder-per-class.
Loads model and labels, samples test images, extracts vectors, predicts, reports accuracy.
Use run_static for real-time webcam testing.
"""
import argparse
import os
import random
import sys
import time

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from ml.detection.hand_detector import HandDetector
from ml.features.feature_vector import landmarks_to_vector_two_hands
from ml.utils.image_utils import resize_for_detection

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_TEST_DIR = os.path.join(ROOT, "data", "processed", "static", "asl_processed", "test")
FALLBACK_TEST_DIR = os.path.join(ROOT, "data", "preprocessed", "static", "asl_processed", "test")
FALLBACK_TEST_DIR_2 = os.path.join(ROOT, "data", "raw", "static", "asl_alphabets")
if os.path.isdir(DEFAULT_TEST_DIR):
    DEFAULT_TEST_DIR = DEFAULT_TEST_DIR
elif os.path.isdir(FALLBACK_TEST_DIR):
    DEFAULT_TEST_DIR = FALLBACK_TEST_DIR
else:
    DEFAULT_TEST_DIR = FALLBACK_TEST_DIR_2


def list_images_by_class(imgs_root: str, extensions: tuple[str, ...]) -> dict[str, list[str]]:
    """Return {class_name: [path, ...]} for immediate subdirs of imgs_root."""
    by_class: dict[str, list[str]] = {}
    for name in sorted(os.listdir(imgs_root)):
        subdir = os.path.join(imgs_root, name)
        if not os.path.isdir(subdir):
            continue
        paths = []
        for f in os.listdir(subdir):
            if f.lower().endswith(extensions):
                paths.append(os.path.join(subdir, f))
        if paths:
            by_class[name] = paths
    return by_class


def stratified_sample(
    by_class: dict[str, list[str]],
    max_total: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Sample up to max_total (label, path) stratified across classes."""
    n_classes = len(by_class)
    if n_classes == 0:
        return []
    per_class = max(1, max_total // n_classes)
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label, paths in sorted(by_class.items()):
        chosen = rng.sample(paths, min(per_class, len(paths)))
        selected.extend((label, p) for p in chosen)
    if len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate static model on images (folder-per-class). Use run_static for real-time video."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=DEFAULT_TEST_DIR,
        help="Root directory with one subdir per class (test images)",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(ROOT, "ml", "models", "static_model.pkl"),
        help="Path to trained static_model.pkl",
    )
    parser.add_argument(
        "--labels-path",
        default=os.path.join(ROOT, "ml", "models", "static_labels.txt"),
        help="Path to static_labels.txt",
    )
    parser.add_argument(
        "--hand-model-path",
        default=os.path.join(ROOT, "mediapipe", "hand_landmarker.task"),
        help="Path to hand_landmarker.task",
    )
    parser.add_argument(
        "--running-mode",
        default="IMAGE",
        help="Running mode for MediaPipe hand landmarker (IMAGE or VIDEO)",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect",
    )
    parser.add_argument(
        "--min-det-conf",
        type=float,
        default=0.4,
        help="Minimum hand detection confidence",
    )
    parser.add_argument(
        "--min-pres-conf",
        type=float,
        default=0.4,
        help="Minimum hand presence confidence",
    )
    parser.add_argument(
        "--min-track-conf",
        type=float,
        default=0.5,
        help="Minimum hand tracking confidence (VIDEO mode only)",
    )
    parser.add_argument(
        "--normalize-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate hand so wrist->middle MCP aligns upward (+Y)",
    )
    parser.add_argument(
        "--mirror-left",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror left-hand X axis to reduce handedness variance",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=256,
        help="Upscale images so the shortest side is at least this size (0 = disable)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Downscale images so the longest side is at most this size (0 = disable)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Progress log frequency (0 = disable)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=200,
        help="Max test images to use (stratified; use different seed than training to reduce overlap)",
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png,bmp",
        help="Comma-separated image extensions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for test sampling (use different from training seed)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print("Model not found:", args.model_path, "- train first: python -m ml.pipeline.training.train_static")
        return
    if not os.path.isfile(args.labels_path):
        print("Labels not found:", args.labels_path)
        return

    with open(args.labels_path) as f:
        labels_list = [line.strip() for line in f if line.strip()]
    model = joblib.load(args.model_path)
    detector = HandDetector(
        model_path=args.hand_model_path,
        max_hands=args.max_hands,
        min_hand_detection_confidence=args.min_det_conf,
        min_hand_presence_confidence=args.min_pres_conf,
        min_tracking_confidence=args.min_track_conf,
        running_mode=args.running_mode,
    )

    extensions = tuple("." + ext.strip().lstrip(".").lower() for ext in args.extensions.split(","))
    by_class = list_images_by_class(args.input_dir, extensions)
    if not by_class:
        if not args.quiet:
            print("No class folders or images in", args.input_dir)
        detector.close()
        return

    selected = stratified_sample(by_class, args.max_test, args.seed)
    if not selected:
        if not args.quiet:
            print("No test images selected.")
        detector.close()
        return

    y_true, y_pred = [], []
    timestamp_ms = 0
    skipped = 0
    total = len(selected)
    start = time.perf_counter()

    for idx, (label, path) in enumerate(selected, start=1):
        img = cv2.imread(path)
        if img is None:
            skipped += 1
            continue
        img = resize_for_detection(img, min_side=args.min_side, max_side=args.max_side)
        result = detector.detect(img, timestamp_ms)
        timestamp_ms += 1
        if not result.hand_landmarks:
            skipped += 1
            continue
        vec = landmarks_to_vector_two_hands(
            result.hand_landmarks,
            result.handedness,
            normalize_rotation=args.normalize_rotation,
            mirror_left=args.mirror_left,
        )
        if vec is None:
            skipped += 1
            continue
        pred = model.predict(vec.reshape(1, -1))[0]
        y_true.append(label)
        y_pred.append(int(pred))
        if args.log_every > 0 and (idx % args.log_every == 0) and not args.quiet:
            elapsed = time.perf_counter() - start
            rate = idx / elapsed if elapsed > 0 else 0.0
            print(f"[{idx}/{total}] {rate:.1f} img/s (skipped={skipped})")

    detector.close()

    if not y_true:
        print("No valid predictions (all images skipped).")
        return

    y_true = np.array(y_true)
    y_pred_str = [labels_list[i] for i in y_pred]
    acc = np.mean(y_true == np.array(y_pred_str))
    if not args.quiet:
        print(f"Test images: {len(y_true)} (skipped {skipped})")
        print(f"Accuracy: {acc:.4f}")

    if not args.quiet:
        print(classification_report(y_true, y_pred_str, zero_division=0))
        print("For real-time video: python -m ml.pipeline.inference.run_static")


if __name__ == "__main__":
    main()
