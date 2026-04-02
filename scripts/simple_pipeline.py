#!/usr/bin/env python
"""
Simple end-to-end pipeline runner.
Order: vectors -> train -> eval -> run_static -> run_sentence_builder
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def has_split(input_dir: str) -> bool:
    return (
        os.path.isdir(os.path.join(input_dir, "train"))
        and os.path.isdir(os.path.join(input_dir, "test"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the static pipeline with configurable params.")
    parser.add_argument("--input-dir", default=os.path.join(ROOT, "data", "raw", "static"))
    parser.add_argument("--output-csv", default=os.path.join(ROOT, "data", "processed", "static", "static_data.csv"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "data", "processed", "static", "hand_vectors"))

    # Vector extraction params
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--min-det-conf", type=float, default=0.3)
    parser.add_argument("--min-pres-conf", type=float, default=0.3)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--max-side", type=int, default=0)
    parser.add_argument("--retry-min-side", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--normalize-rotation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mirror-left", action=argparse.BooleanOptionalAction, default=True)

    # Training params
    parser.add_argument("--skip-xgb", action="store_true")
    parser.add_argument("--use-scaler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-eval", action="store_true")

    # Inference params
    parser.add_argument("--ema-alpha", type=float, default=0.4)
    parser.add_argument("--detector-max-side", type=int, default=640)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--clear-frames", type=int, default=10)

    parser.add_argument("--run-static", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-sentence", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    os.chdir(ROOT)

    split = has_split(args.input_dir)

    # Step 1: vectors
    vec_cmd = [
        sys.executable, "-m", "ml.pipeline.data_collection.images_to_vectors",
        "--input-dir", args.input_dir,
        "--max-images", str(args.max_images),
        "--max-per-class", str(args.max_per_class),
        "--min-det-conf", str(args.min_det_conf),
        "--min-pres-conf", str(args.min_pres_conf),
        "--min-side", str(args.min_side),
        "--max-side", str(args.max_side),
        "--retry-min-side", str(args.retry_min_side),
        "--log-every", str(args.log_every),
        bool_flag("normalize-rotation", args.normalize_rotation),
        bool_flag("mirror-left", args.mirror_left),
    ]
    if args.quiet:
        vec_cmd.append("--quiet")

    if split:
        vec_cmd += ["--output-dir", args.output_dir]
        train_data = args.output_dir
    else:
        vec_cmd += ["--output", args.output_csv]
        train_data = args.output_csv

    run(vec_cmd)

    # Step 2: train
    train_cmd = [
        sys.executable, "-m", "ml.pipeline.training.train_static",
        "--data", train_data,
        bool_flag("use-scaler", args.use_scaler),
    ]
    if args.skip_xgb:
        train_cmd.append("--skip-xgb")
    run(train_cmd)

    # Step 3: eval
    if not args.skip_eval:
        eval_cmd = [
            sys.executable, "-m", "ml.pipeline.inference.eval_static_images",
            "--input-dir", args.input_dir,
            "--min-det-conf", str(args.min_det_conf),
            "--min-pres-conf", str(args.min_pres_conf),
            bool_flag("normalize-rotation", args.normalize_rotation),
            bool_flag("mirror-left", args.mirror_left),
        ]
        if args.quiet:
            eval_cmd.append("--quiet")
        run(eval_cmd)

    # Step 4: run_static
    if args.run_static:
        run_static_cmd = [
            sys.executable, "-m", "ml.pipeline.inference.run_static",
            "--ema-alpha", str(args.ema_alpha),
            "--detector-max-side", str(args.detector_max_side),
            "--frame-skip", str(args.frame_skip),
            "--history", str(args.history),
            "--stable-frames", str(args.stable_frames),
            "--clear-frames", str(args.clear_frames),
            bool_flag("normalize-rotation", args.normalize_rotation),
            bool_flag("mirror-left", args.mirror_left),
        ]
        run(run_static_cmd)

    # Step 5: sentence builder
    if args.run_sentence:
        run_sentence_cmd = [
            sys.executable, "-m", "ml.pipeline.inference.run_sentence_builder",
            "--ema-alpha", str(args.ema_alpha),
            "--detector-max-side", str(args.detector_max_side),
            "--history", str(max(1, args.history)),
            "--stable-frames", str(max(1, args.stable_frames)),
            bool_flag("normalize-rotation", args.normalize_rotation),
            bool_flag("mirror-left", args.mirror_left),
        ]
        run(run_sentence_cmd)


if __name__ == "__main__":
    main()
