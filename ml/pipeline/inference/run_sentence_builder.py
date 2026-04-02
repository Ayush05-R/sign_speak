"""
Live sentence builder using the static sign classifier.
Shows current prediction and builds a sentence with space/full-stop labels.
"""
import argparse
import os
import sys
import time
from collections import Counter, deque

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
        labels = [line.strip() for line in f if line.strip()]
    return labels


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


def draw_text_with_bg_fade(
    frame,
    text: str,
    org: tuple[int, int],
    font,
    font_scale: float,
    text_color: tuple[int, int, int],
    bg_color: tuple[int, int, int],
    alpha: float,
    thickness: int = 2,
    padding: int = 6,
) -> None:
    if not text:
        return
    alpha = max(0.0, min(1.0, alpha))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    x1 = max(0, x - padding)
    y1 = max(0, y - th - padding)
    x2 = min(frame.shape[1], x + tw + padding)
    y2 = min(frame.shape[0], y + baseline + padding)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), bg_color, -1)
    cv2.putText(
        overlay,
        text,
        (x - x1, y - y1),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi


def fit_text_to_width(
    text: str,
    max_width: int,
    font,
    font_scale: float,
    thickness: int,
    prefix: str = "...",
) -> str:
    if not text or max_width <= 0:
        return text
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    if tw <= max_width:
        return text
    trimmed = text
    while trimmed:
        trimmed = trimmed[1:]
        candidate = f"{prefix}{trimmed}"
        (cw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if cw <= max_width:
            return candidate
    return prefix.strip()


def normalize_label(label: str, space_label: str, fullstop_label: str) -> str:
    if label == space_label:
        return " "
    if label == fullstop_label:
        return "."
    return label


def append_char(sentence: str, ch: str) -> str:
    if ch == " ":
        if not sentence or sentence.endswith(" "):
            return sentence
        return sentence + " "
    if ch == ".":
        if not sentence:
            return sentence
        if sentence.endswith("."):
            return sentence
        return sentence + "."
    return sentence + ch


def main():
    parser = argparse.ArgumentParser(description="Live sentence builder with static sign model.")
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
    parser.add_argument("--ema-alpha", type=float, default=0.4, help="EMA smoothing factor (0 = off)")
    parser.add_argument("--detector-max-side", type=int, default=0, help="Downscale frame before detection (0 = off)")
    parser.add_argument("--history", type=int, default=7, help="Frames to use for majority vote")
    parser.add_argument("--stable-frames", type=int, default=6, help="Frames required to accept a label")
    parser.add_argument("--clear-frames", type=int, default=12, help="Clear state after N no-hand frames")
    parser.add_argument("--cooldown-frames", type=int, default=12, help="Cooldown after accepting a label")
    parser.add_argument("--fade-frames", type=int, default=18, help="Frames to fade out sentence after full-stop")
    parser.add_argument(
        "--require-two-hands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only predict when two hands are detected",
    )
    parser.add_argument("--space-label", default="space")
    parser.add_argument("--fullstop-label", default="full-stop")
    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print("Model not found:", args.model_path)
        print("Train first: python -m ml.pipeline.training.train_static")
        return
    if not os.path.isfile(args.labels_path):
        print("Labels not found:", args.labels_path)
        return

    labels = load_labels(args.labels_path)
    model = joblib.load(args.model_path)
    detector = HandDetector(
        model_path=args.hand_model_path,
        max_hands=args.max_hands,
        min_hand_detection_confidence=args.min_det_conf,
        min_hand_presence_confidence=args.min_pres_conf,
        min_tracking_confidence=args.min_track_conf,
        running_mode="VIDEO",
    )

    cap = cv2.VideoCapture(args.camera)
    set_capture_size(cap, args.width, args.height)
    if not cap.isOpened():
        print("Could not open webcam.")
        detector.close()
        return

    history: deque[str] = deque(maxlen=max(1, args.history))
    sentence = ""
    last_label = None
    last_committed = ""
    stable_count = 0
    cooldown = 0
    ema_probs = None
    no_hand_count = 0
    last_detected_hands = True
    frame_idx = 0
    last_fps_time = time.perf_counter()
    fps = 0.0
    fade_text = ""
    fade_left = 0
    start = time.time()
    clear_threshold = max(1, args.clear_frames)

    print("Sentence builder running. Q = quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts = int((time.time() - start) * 1000)
        run_detection = (args.frame_skip <= 0) or (frame_idx % (args.frame_skip + 1) == 0)
        frame_idx += 1

        if run_detection:
            detect_frame = resize_for_detection(frame, min_side=0, max_side=args.detector_max_side)
            result = detector.detect(detect_frame, ts)
        else:
            result = None

        current_label = None
        detected_hands = False
        if run_detection and result is not None and result.hand_landmarks:
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
                    current_label = labels[pred_idx]
                    if cooldown <= 0:
                        history.append(current_label)
            last_detected_hands = detected_hands
        elif run_detection:
            last_detected_hands = False

        if not run_detection and not last_detected_hands:
            no_hand_count += 1
        elif run_detection and not detected_hands:
            no_hand_count += 1

        if no_hand_count >= clear_threshold:
            history.clear()
            last_label = None
            last_committed = ""
            stable_count = 0
            cooldown = 0

        if cooldown > 0:
            cooldown -= 1
            stable_count = 0
            history.clear()
            maj = None
        else:
            maj = majority_label(history)
            if maj is None:
                stable_count = 0
            else:
                if maj == last_label:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_label = maj

            if maj is not None and stable_count >= args.stable_frames:
                last_committed = maj
                ch = normalize_label(maj, args.space_label, args.fullstop_label)
                if ch == "." and sentence:
                    sentence = append_char(sentence, ch)
                    fade_text = sentence
                    fade_left = max(1, args.fade_frames)
                    sentence = ""
                else:
                    sentence = append_char(sentence, ch)
                cooldown = max(0, args.cooldown_frames)
                stable_count = 0
                history.clear()

        if cooldown > 0 and last_committed:
            display_label = last_committed
        else:
            display_label = maj if maj is not None else "-"
        draw_text_with_bg(
            frame,
            f"Pred: {display_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            (0, 0, 0),
            thickness=2,
            padding=6,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        max_text_width = max(0, w - 40)
        padding = 8
        (_, text_h), text_baseline = cv2.getTextSize("Ag", font, font_scale, thickness)
        line_height = text_h + text_baseline + (2 * padding) + 4
        sentence_y = h - 20
        fade_y = max(20, sentence_y - line_height)
        if fade_left > 0 and fade_text:
            fade_alpha = fade_left / max(1, args.fade_frames)
            fade_text_trim = fit_text_to_width(
                f"Sentence: {fade_text}",
                max_text_width,
                font,
                font_scale,
                thickness,
            )
            draw_text_with_bg_fade(
                frame,
                fade_text_trim,
                (20, fade_y),
                font,
                font_scale,
                (255, 255, 255),
                (0, 0, 0),
                alpha=fade_alpha,
                thickness=thickness,
                padding=padding,
            )
            fade_left -= 1

        sentence_text = f"Sentence: {sentence}"
        sentence_text = fit_text_to_width(sentence_text, max_text_width, font, font_scale, thickness)
        draw_text_with_bg(
            frame,
            sentence_text,
            (20, sentence_y),
            font,
            font_scale,
            (255, 255, 255),
            (0, 0, 0),
            thickness=thickness,
            padding=padding,
        )
        if args.show_fps:
            now = time.perf_counter()
            dt = now - last_fps_time
            if dt > 0:
                fps = 1.0 / dt
            last_fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Sentence Builder", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
