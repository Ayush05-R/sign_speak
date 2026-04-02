"""
Convert folder-per-class images into landmark vectors for static training.

Supports:
  1) A single folder-per-class dataset.
  2) A split dataset with train/ and test/ subfolders (each folder-per-class).

Sampling:
  - max_per_class (default 500) caps images per class for faster runs.
  - max_images (default 0) optionally caps total images after per-class sampling.
"""
import argparse
import csv
from dataclasses import dataclass
import os
import random
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from ml.detection.hand_detector import HandDetector
from ml.features.feature_vector import landmarks_to_vector_two_hands, get_feature_dim_two_hands
from ml.utils.image_utils import resize_for_detection

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
N_FEATURES = get_feature_dim_two_hands()  # 126 (both hands)

def _pick_default_input_dir() -> str:
    candidates = [
        os.path.join(ROOT, "data", "raw", "ASL"),
        os.path.join(ROOT, "data", "raw", "IndianHS"),
    ]
    for path in candidates:
        if not os.path.isdir(path):
            continue
        train_dir = os.path.join(path, "train")
        test_dir = os.path.join(path, "test")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            return path
        # Accept folder-per-class if it has at least one subdir
        try:
            if any(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)):
                return path
        except OSError:
            continue
    return candidates[-1]


DEFAULT_INPUT_DIR = _pick_default_input_dir()
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "processed", "hand_vectors")
DEFAULT_OUTPUT_CSV = os.path.join(ROOT, "data", "processed", "static_data.csv")

@dataclass(frozen=True)
class VectorizeSettings:
    max_images: int
    max_per_class: int
    min_side: int
    max_side: int
    retry_min_side: int
    seed: int
    log_every: int
    quiet: bool
    normalize_rotation: bool
    mirror_left: bool


def list_images_by_class(imgs_root: str, extensions: tuple[str, ...]) -> dict[str, list[str]]:
    """Return {class_name: [path, ...]} for immediate subdirs of imgs_root."""
    by_class: dict[str, list[str]] = {}
    for name in sorted(os.listdir(imgs_root)):
        subdir = os.path.join(imgs_root, name)
        if not os.path.isdir(subdir):
            continue
        paths = []
        for f in sorted(os.listdir(subdir)):
            if f.lower().endswith(extensions):
                paths.append(os.path.join(subdir, f))
        if paths:
            by_class[name] = paths
    return by_class


def select_images(
    by_class: dict[str, list[str]],
    max_total: int,
    max_per_class: int,
    seed: int,
) -> list[tuple[str, str]]:
    """Select images with optional per-class and total caps."""
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label, paths in sorted(by_class.items()):
        if max_per_class > 0 and len(paths) > max_per_class:
            chosen = rng.sample(paths, max_per_class)
        else:
            chosen = list(paths)
        selected.extend((label, p) for p in chosen)
    if max_total > 0 and len(selected) > max_total:
        rng.shuffle(selected)
        selected = selected[:max_total]
    return selected


def resolve_split_dirs(input_dir: str) -> dict[str, str] | None:
    """Return split directories if input_dir contains train/ and test/ subfolders."""
    train_dir = os.path.join(input_dir, "train")
    test_dir = os.path.join(input_dir, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return {"train": train_dir, "test": test_dir}
    return None


def write_vectors_for_dir(
    input_dir: str,
    output_csv: str,
    detector: HandDetector,
    extensions: tuple[str, ...],
    settings: VectorizeSettings,
    timestamp_ms: int,
) -> tuple[int, int, int]:
    """Write vectors for one folder-per-class dataset. Returns (written, skipped, last_ts)."""
    by_class = list_images_by_class(input_dir, extensions)
    if not by_class:
        if not settings.quiet:
            print("No class folders or images found in", input_dir)
        return 0, 0, timestamp_ms

    total_images = sum(len(paths) for paths in by_class.values())
    if not settings.quiet:
        print(f"Found {len(by_class)} classes, {total_images} images in {input_dir}")

    selected = select_images(by_class, settings.max_images, settings.max_per_class, settings.seed)
    if not selected:
        if not settings.quiet:
            print("No images selected in", input_dir)
        return 0, 0, timestamp_ms
    if not settings.quiet:
        print(
            f"Selected {len(selected)} images "
            f"(max_per_class={settings.max_per_class}, max_total={settings.max_images})"
        )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    skipped = 0
    written = 0
    total = len(selected)
    start = time.perf_counter()
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"f{i}" for i in range(N_FEATURES)])

        for idx, (label, path) in enumerate(selected, start=1):
            raw_img = cv2.imread(path)
            if raw_img is None:
                skipped += 1
                continue
            img = resize_for_detection(raw_img, min_side=settings.min_side, max_side=settings.max_side)
            result = detector.detect(img, timestamp_ms)
            timestamp_ms += 1
            if not result.hand_landmarks:
                if settings.retry_min_side > settings.min_side:
                    retry_img = resize_for_detection(
                        raw_img,
                        min_side=settings.retry_min_side,
                        max_side=settings.max_side,
                    )
                    result = detector.detect(retry_img, timestamp_ms)
                    timestamp_ms += 1
                if not result.hand_landmarks:
                    skipped += 1
                    continue
            vec = landmarks_to_vector_two_hands(
                result.hand_landmarks,
                result.handedness,
                normalize_rotation=settings.normalize_rotation,
                mirror_left=settings.mirror_left,
            )
            if vec is None:
                skipped += 1
                continue
            writer.writerow([label] + vec.tolist())
            written += 1
            if settings.log_every > 0 and (idx % settings.log_every == 0):
                if not settings.quiet:
                    elapsed = time.perf_counter() - start
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{idx}/{total}] {rate:.1f} img/s "
                        f"(written={written}, skipped={skipped})"
                    )

    total_processed = written + skipped
    if total_processed > 0:
        skip_pct = 100.0 * skipped / total_processed
        if not settings.quiet:
            print(
                f"Wrote {written} rows to {output_csv} "
                f"(skipped {skipped} images, {skip_pct:.1f}%)."
            )
    else:
        if not settings.quiet:
            print("No valid images processed for", output_csv)
    return written, skipped, timestamp_ms


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmark vectors from images and write CSV(s) for static training."
    )
    parser.add_argument(
        "--input-dir", "-i",
        default=DEFAULT_INPUT_DIR,
        help="Root directory with class subfolders OR a root containing train/ and test/ splits",
    )
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path when input-dir is a single folder-per-class dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory when input-dir contains train/ and test/ splits",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max total images per split; 0 = no total cap",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=500,
        help="Max images per class; 0 = no per-class cap",
    )
    parser.add_argument(
        "--model-path",
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
        "--retry-min-side",
        type=int,
        default=0,
        help="Retry detection with a larger min-side when no hand is found (0 = disable)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Progress log frequency (0 = disable)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output",
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png,bmp",
        help="Comma-separated image extensions to scan",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling",
    )
    args = parser.parse_args()

    extensions = tuple("." + ext.strip().lstrip(".").lower() for ext in args.extensions.split(","))
    settings = VectorizeSettings(
        max_images=args.max_images,
        max_per_class=args.max_per_class,
        min_side=args.min_side,
        max_side=args.max_side,
        retry_min_side=args.retry_min_side,
        seed=args.seed,
        log_every=args.log_every,
        quiet=args.quiet,
        normalize_rotation=args.normalize_rotation,
        mirror_left=args.mirror_left,
    )
    detector = HandDetector(
        model_path=args.model_path,
        max_hands=args.max_hands,
        min_hand_detection_confidence=args.min_det_conf,
        min_hand_presence_confidence=args.min_pres_conf,
        min_tracking_confidence=args.min_track_conf,
        running_mode=args.running_mode,
    )
    timestamp_ms = 0
    split_dirs = resolve_split_dirs(args.input_dir)

    if split_dirs:
        os.makedirs(args.output_dir, exist_ok=True)
        for split_name, split_dir in split_dirs.items():
            output_csv = os.path.join(args.output_dir, f"{split_name}.csv")
            _, _, timestamp_ms = write_vectors_for_dir(
                split_dir,
                output_csv,
                detector,
                extensions,
                settings,
                timestamp_ms,
            )
        detector.close()
        if not args.quiet:
            print("Next: python -m ml.pipeline.training.train_static --data", args.output_dir)
        return

    _, _, _ = write_vectors_for_dir(
        args.input_dir,
        args.output,
        detector,
        extensions,
        settings,
        timestamp_ms,
    )
    detector.close()
    if not args.quiet:
        print("Next: python -m ml.pipeline.training.train_static --data", args.output)


if __name__ == "__main__":
    main()
