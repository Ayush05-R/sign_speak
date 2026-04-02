"""
Hand detection via MediaPipe Tasks API.
Exposes HandDetector for use by data collection and inference.
"""
import os
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH_DEFAULT = os.path.join("mediapipe", "hand_landmarker.task")
MAX_HANDS_DEFAULT = 2
MIN_HAND_DETECTION_CONFIDENCE_DEFAULT = 0.5
MIN_HAND_PRESENCE_CONFIDENCE_DEFAULT = 0.5
MIN_TRACKING_CONFIDENCE_DEFAULT = 0.5
RUNNING_MODE_DEFAULT = "VIDEO"

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
FINGER_TIPS: list[int] = [4, 8, 12, 16, 20]
FINGER_PIPS: list[int] = [3, 6, 10, 14, 18]
NUM_LANDMARKS = 21


class HandDetector:
    """MediaPipe HandLandmarker wrapper. detect() returns hand_landmarks and handedness."""

    def __init__(
        self,
        model_path: str = MODEL_PATH_DEFAULT,
        max_hands: int = MAX_HANDS_DEFAULT,
        min_hand_detection_confidence: float = MIN_HAND_DETECTION_CONFIDENCE_DEFAULT,
        min_hand_presence_confidence: float = MIN_HAND_PRESENCE_CONFIDENCE_DEFAULT,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE_DEFAULT,
        running_mode: str = RUNNING_MODE_DEFAULT,
    ):
        if isinstance(running_mode, str):
            mode_key = running_mode.strip().upper()
            if not hasattr(mp_vision.RunningMode, mode_key):
                raise ValueError(f"Invalid running_mode: {running_mode}")
            running_mode = getattr(mp_vision.RunningMode, mode_key)

        self._running_mode = running_mode
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame, timestamp_ms: int | None = None):
        import mediapipe as mp
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._running_mode == mp_vision.RunningMode.IMAGE:
            return self._landmarker.detect(mp_img)
        if timestamp_ms is None:
            raise ValueError("timestamp_ms is required for VIDEO or LIVE_STREAM modes.")
        return self._landmarker.detect_for_video(mp_img, int(timestamp_ms))

    def count_fingers(self, landmarks, hand_label: str) -> int:
        thumb_up = (
            landmarks[4].x < landmarks[3].x if hand_label == "Right"
            else landmarks[4].x > landmarks[3].x
        )
        other = sum(
            landmarks[tip].y < landmarks[pip].y
            for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:])
        )
        return int(thumb_up) + other

    def close(self) -> None:
        self._landmarker.close()
