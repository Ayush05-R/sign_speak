"""
Convert MediaPipe hand landmarks to a normalized feature vector for ML.
Normalizes by wrist (origin) and scale (hand size) for position/scale invariance.
"""
import numpy as np

# Wrist = 0, middle finger MCP = 9 (palm); use wrist-to-MCP distance as scale
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def _rotate_xy(arr: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate all XY coordinates by angle (radians) around origin."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    arr[:, 0] = c * x - s * y
    arr[:, 1] = s * x + c * y
    return arr


def landmarks_to_vector(
    landmarks,
    eps: float = 1e-6,
    normalize_rotation: bool = False,
    mirror_x: bool = False,
) -> np.ndarray | None:
    """
    Convert one hand's 21 landmarks to a flat feature vector.

    Steps:
      1. Extract x, y, z for all 21 landmarks (63 values).
      2. Subtract wrist (landmark 0) so pose is relative to wrist.
      3. Scale by hand size (distance from wrist to middle MCP) so size is normalized.
      4. (Optional) Mirror X axis for left-hand normalization.
      5. (Optional) Rotate so wrist->middle MCP aligns upward (+Y).

    Args:
        landmarks: list of 21 objects with .x, .y, .z (e.g. from MediaPipe result)
        eps: small value to avoid division by zero when computing scale
        normalize_rotation: align wrist->middle MCP to +Y to reduce rotation variance
        mirror_x: flip X axis (useful for left-hand normalization)

    Returns:
        shape (63,) float array, or None if scale is too small (invalid hand).
    """
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = arr[WRIST_IDX]
    arr = arr - wrist
    scale = np.linalg.norm(arr[MIDDLE_MCP_IDX]) + eps
    if scale < 1e-4:
        return None
    arr = arr / scale
    if mirror_x:
        arr[:, 0] *= -1.0
    if normalize_rotation:
        base = arr[MIDDLE_MCP_IDX][:2]
        base_norm = np.linalg.norm(base) + eps
        if base_norm > 1e-4:
            angle = np.arctan2(base[1], base[0])
            target = np.pi / 2  # align to +Y axis
            arr = _rotate_xy(arr, target - angle)
    return arr.flatten()


def get_feature_dim() -> int:
    """Return number of features per frame (21 * 3 = 63)."""
    return 21 * 3


def get_feature_dim_two_hands() -> int:
    """Return number of features for two hands (63 * 2 = 126)."""
    return 21 * 3 * 2


def _hand_label(handedness_list) -> str:
    """Get 'Left' or 'Right' from handedness list for one hand (MediaPipe Category)."""
    if not handedness_list:
        return "Right"
    c = handedness_list[0]
    name = getattr(c, "category_name", "Right")
    if isinstance(name, bytes):
        name = name.decode("utf-8") if name else "Right"
    return name if name in ("Left", "Right") else "Right"


def landmarks_to_vector_two_hands(
    hand_landmarks,
    handedness,
    eps: float = 1e-6,
    normalize_rotation: bool = False,
    mirror_left: bool = False,
) -> np.ndarray | None:
    """
    Convert one or two hands to a 126-dim vector: [left_hand_63, right_hand_63].
    Order is fixed so Left always in first 63, Right in last 63. Missing hand is zeros.
    Returns None only if no valid hand is present.
    """
    dim_one = 63
    zeros = np.zeros(dim_one, dtype=np.float32)
    left_vec = None
    right_vec = None
    for i, landmarks in enumerate(hand_landmarks):
        label = _hand_label(handedness[i]) if i < len(handedness) else "Right"
        mirror_x = mirror_left and label == "Left"
        vec = landmarks_to_vector(
            landmarks,
            eps=eps,
            normalize_rotation=normalize_rotation,
            mirror_x=mirror_x,
        )
        if vec is None:
            continue
        if label == "Left":
            left_vec = vec
        else:
            right_vec = vec
    if left_vec is None and right_vec is None:
        return None
    left_vec = left_vec if left_vec is not None else zeros
    right_vec = right_vec if right_vec is not None else zeros
    return np.concatenate([left_vec, right_vec]).astype(np.float32)
