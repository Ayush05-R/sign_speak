"""
Live static sign inference: webcam -> hand landmarks -> feature vector -> classifier -> label.
Optimized with optional smoothing, frame skipping, and configurable detector thresholds.
"""
import argparse
from collections import Counter, deque
import os
import subprocess
import sys
import time

import cv2
import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from ml.detection.hand_detector import HandDetector
from ml.features.feature_vector import landmarks_to_vector_two_hands
from ml.utils.image_utils import resize_for_detection


def load_labels(labels_path: str) -> list[str]:
    with open(labels_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def majority_label(history: deque[str]) -> str | None:
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32)
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / (np.sum(exp_scores) + 1e-9)


def predict_with_ema(model, vec: np.ndarray, ema_probs: np.ndarray | None, alpha: float, n_classes: int):
    if alpha <= 0:
        pred = model.predict(vec.reshape(1, -1))[0]
        return int(pred), ema_probs
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec.reshape(1, -1))[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec.reshape(1, -1))
        scores = scores[0] if hasattr(scores, "__len__") else scores
        probs = softmax(np.asarray(scores))
    else:
        pred = int(model.predict(vec.reshape(1, -1))[0])
        probs = np.zeros(n_classes, dtype=np.float32)
        probs[pred] = 1.0
    ema_probs = probs if ema_probs is None else (alpha * probs + (1.0 - alpha) * ema_probs)
    return int(np.argmax(ema_probs)), ema_probs


def set_capture_size(cap: cv2.VideoCapture, width: int, height: int) -> None:
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))


def draw_text_with_bg(
    frame,
    text: str,
    org: tuple[int, int],
    font,
    font_scale: float,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    thickness: int = 2,
    padding: int = 6,
) -> None:
    if not text:
        return
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = min(frame.shape[1], x + tw + padding)
    y2 = min(frame.shape[0], y + baseline + padding)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def run_sentence_builder(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "ml.pipeline.inference.run_sentence_builder",
        "--model-path",
        args.model_path,
        "--labels-path",
        args.labels_path,
        "--hand-model-path",
        args.hand_model_path,
        "--camera",
        str(args.camera),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--max-hands",
        str(args.max_hands),
        "--min-det-conf",
        str(args.min_det_conf),
        "--min-pres-conf",
        str(args.min_pres_conf),
        "--min-track-conf",
        str(args.min_track_conf),
        "--ema-alpha",
        str(args.ema_alpha),
        "--detector-max-side",
        str(args.detector_max_side),
        "--frame-skip",
        str(args.frame_skip),
        "--history",
        str(args.history),
        "--stable-frames",
        str(args.stable_frames),
        "--clear-frames",
        str(args.clear_frames),
    ]
    if not args.normalize_rotation:
        cmd.append("--no-normalize-rotation")
    if not args.mirror_left:
        cmd.append("--no-mirror-left")
    if args.no_flip:
        cmd.append("--no-flip")
    if args.draw_landmarks:
        cmd.append("--draw-landmarks")
    if args.show_fps:
        cmd.append("--show-fps")
    if args.require_two_hands:
        cmd.append("--require-two-hands")
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="Live static sign inference (webcam).")
    parser.add_argument("--model-path", default=os.path.join(ROOT, "ml", "models", "static_model.pkl"))
    parser.add_argument("--labels-path", default=os.path.join(ROOT, "ml", "models", "static_labels.txt"))
    parser.add_argument("--hand-model-path", default=os.path.join(ROOT, "mediapipe", "hand_landmarker.task"))
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--width", type=int, default=640, help="Capture width (0 = default)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (0 = default)")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror flip")
    parser.add_argument("--draw-landmarks", action="store_true", help="Draw landmarks on frame")
    parser.add_argument("--show-fps", action="store_true", help="Overlay FPS")
    parser.add_argument("--frame-skip", type=int, default=0, help="Skip detection on N frames (0 = no skip)")
    parser.add_argument("--history", type=int, default=5, help="Frames for majority vote")
    parser.add_argument("--stable-frames", type=int, default=3, help="Frames required to accept a new label")
    parser.add_argument("--clear-frames", type=int, default=10, help="Clear label after N no-hand frames")
    parser.add_argument("--ema-alpha", type=float, default=0.4, help="EMA smoothing factor (0 = off)")
    parser.add_argument("--detector-max-side", type=int, default=0, help="Downscale frame before detection (0 = off)")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-det-conf", type=float, default=0.4)
    parser.add_argument("--min-pres-conf", type=float, default=0.4)
    parser.add_argument("--min-track-conf", type=float, default=0.5)
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
        "--require-two-hands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only predict when two hands are detected",
    )
    parser.add_argument(
        "--run-sentence-builder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run sentence builder after closing the static window",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print("Train first: python -m ml.pipeline.training.train_static")
        return
    if not os.path.isfile(args.labels_path):
        print("Missing", args.labels_path)
        return

    labels = load_labels(args.labels_path)
    model = joblib.load(args.model_path)

    cap = cv2.VideoCapture(args.camera)
    set_capture_size(cap, args.width, args.height)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    detector = HandDetector(
        model_path=args.hand_model_path,
        max_hands=args.max_hands,
        min_hand_detection_confidence=args.min_det_conf,
        min_hand_presence_confidence=args.min_pres_conf,
        min_tracking_confidence=args.min_track_conf,
        running_mode="VIDEO",
    )

    history: deque[str] = deque(maxlen=max(1, args.history))
    pending_label = None
    pending_count = 0
    last_label = ""
    no_hand_count = 0
    ema_probs = None
    frame_idx = 0
    start = time.time()
    last_fps_time = time.perf_counter()
    fps = 0.0
    last_detected_hands = True
    clear_threshold = max(1, args.clear_frames)

    print("Live static inference. Q = quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        run_detection = (args.frame_skip <= 0) or (frame_idx % (args.frame_skip + 1) == 0)
        frame_idx += 1

        if run_detection:
            ts = int((time.time() - start) * 1000)
            detect_frame = resize_for_detection(frame, min_side=0, max_side=args.detector_max_side)
            result = detector.detect(detect_frame, ts)
            detected_hands = False
            if result.hand_landmarks:
                if args.draw_landmarks:
                    for landmarks in result.hand_landmarks:
                        for lm in landmarks:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                if not args.require_two_hands or len(result.hand_landmarks) >= 2:
                    detected_hands = True
                    no_hand_count = 0
                    vec = landmarks_to_vector_two_hands(
                        result.hand_landmarks,
                        result.handedness,
                        normalize_rotation=args.normalize_rotation,
                        mirror_left=args.mirror_left,
                    )
                    if vec is not None:
                        pred_idx, ema_probs = predict_with_ema(
                            model, vec, ema_probs, args.ema_alpha, len(labels)
                        )
                        history.append(labels[pred_idx])
            if not detected_hands:
                no_hand_count += 1
            last_detected_hands = detected_hands
        else:
            if not last_detected_hands:
                no_hand_count += 1

        if no_hand_count >= clear_threshold:
            history.clear()
            pending_label = None
            pending_count = 0
            last_label = ""

        smoothed = majority_label(history)
        if smoothed is not None:
            if smoothed == pending_label:
                pending_count += 1
            else:
                pending_label = smoothed
                pending_count = 1
            if pending_count >= max(1, args.stable_frames):
                last_label = pending_label

        if args.show_fps:
            now = time.perf_counter()
            dt = now - last_fps_time
            if dt > 0:
                fps = 1.0 / dt
            last_fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        draw_text_with_bg(
            frame,
            last_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            (0, 0, 0),
            thickness=2,
            padding=6,
        )
        cv2.imshow("Static inference", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    if args.run_sentence_builder:
        print("Launching sentence builder...")
        run_sentence_builder(args)


if __name__ == "__main__":
    main()
