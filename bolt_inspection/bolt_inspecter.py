qqimport time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ================= CONFIGURATION =================
# Camera index:
# 0 = built-in camera
# 1 = external camera
CAMERA_INDEX = 0

# ================= CALIBRATION ===================
# Fixed camera calibration: millimeters per pixel
MM_PER_PIXEL = 0.157834

# Detection tuning
MIN_BOLT_AREA = 1200
MAX_BOLT_AREA_RATIO = 0.45
MIN_ASPECT_RATIO = 1.5
MIN_BOLT_LENGTH_RATIO = 0.10
DETECTION_MAX_DIM = 960
DETECTION_MEDIAN_BLUR = 5
USE_TRAINED_BOLT_MODEL = True
MODEL_CONFIDENCE = 0.15
MODEL_IOU = 0.45
MODEL_INPUT_SIZE = 640
MODEL_BOX_PADDING_RATIO = 0.10
TRACK_MATCH_DISTANCE_RATIO = 0.75
TRACK_MAX_LENGTH_DELTA_RATIO = 0.35
TRACK_CENTER_BONUS = 80.0
TRACK_SIZE_BONUS = 35.0
TRACK_HOLD_FRAMES = 12
TRACK_SMOOTHING = 0.65

# Display
DISPLAY_SCALE = 1.0

# Search only near center or full frame
USE_INSPECTION_ZONE = False
INSPECTION_ZONE_RATIO = 0.7

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / ".venv"
    / "Model_training"
    / "bolt.pt"
)
_BOLT_MODEL = None
_BOLT_MODEL_LOAD_ATTEMPTED = False
_BOLT_MODEL_ERROR = None


# ================= UTILITY HELPERS =================
def open_camera(index):
    """Open webcam."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open webcam at index {index}. "
            "Check the camera connection and CAMERA_INDEX setting."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def prepare_detection_gray(gray):
    """Downscale and denoise high-resolution frames before bolt detection."""
    if gray is None or gray.size == 0:
        return np.zeros((8, 8), dtype=np.uint8), 1.0

    gray = np.ascontiguousarray(gray)
    height, width = gray.shape[:2]
    max_dim = max(height, width)
    scale = 1.0

    if max_dim > DETECTION_MAX_DIM:
        scale = DETECTION_MAX_DIM / float(max_dim)
        resized_width = max(8, int(round(width * scale)))
        resized_height = max(8, int(round(height * scale)))
        gray = cv2.resize(
            gray,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    blur_size = max(1, int(DETECTION_MEDIAN_BLUR))
    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        gray = cv2.medianBlur(gray, blur_size)

    return gray, scale


def prepare_detection_frame(frame, scale):
    """Resize and denoise the color frame to match detection scale."""
    if frame is None or frame.size == 0:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    detection_frame = np.ascontiguousarray(frame)
    if scale != 1.0:
        resized_width = max(8, int(round(frame.shape[1] * scale)))
        resized_height = max(8, int(round(frame.shape[0] * scale)))
        detection_frame = cv2.resize(
            detection_frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    blur_size = max(1, int(DETECTION_MEDIAN_BLUR))
    if blur_size > 1:
        if blur_size % 2 == 0:
            blur_size += 1
        detection_frame = cv2.medianBlur(detection_frame, blur_size)

    return np.ascontiguousarray(detection_frame)


def scale_bolt_geometry(bolt, scale_x, scale_y):
    """Scale bolt geometry between detection and display resolutions."""
    if bolt is None:
        return None

    scaled = bolt.copy()

    has_box = "box" in scaled and scaled["box"] is not None
    if has_box:
        scaled_box = scaled["box"].astype(np.float32).copy()
        scaled_box[:, 0] *= float(scale_x)
        scaled_box[:, 1] *= float(scale_y)
        scaled["box"] = scaled_box
        scaled["center"] = tuple(np.mean(scaled_box, axis=0))

    elif "center" in scaled and scaled["center"] is not None:
        scaled["center"] = (
            float(scaled["center"][0]) * float(scale_x),
            float(scaled["center"][1]) * float(scale_y),
        )

    if "length" in scaled and scaled["length"] is not None:
        scaled["length"] = float(scaled["length"]) * float(max(scale_x, scale_y))

    if "search_zone" in scaled and scaled["search_zone"] is not None:
        zone_x, zone_y, zone_w, zone_h = scaled["search_zone"]
        scaled["search_zone"] = (
            int(round(zone_x * scale_x)),
            int(round(zone_y * scale_y)),
            int(round(zone_w * scale_x)),
            int(round(zone_h * scale_y)),
        )

    return scaled


def get_inspection_zone(gray):
    """Return centered search region if enabled."""
    frame_h, frame_w = gray.shape
    if not USE_INSPECTION_ZONE:
        return gray, (0, 0, frame_w, frame_h)

    zone_w = int(frame_w * INSPECTION_ZONE_RATIO)
    zone_h = int(frame_h * INSPECTION_ZONE_RATIO)
    x0 = max(0, (frame_w - zone_w) // 2)
    y0 = max(0, (frame_h - zone_h) // 2)
    return gray[y0:y0 + zone_h, x0:x0 + zone_w], (x0, y0, zone_w, zone_h)


def load_bolt_model():
    """Load the trained bolt detector once and reuse it."""
    global _BOLT_MODEL
    global _BOLT_MODEL_LOAD_ATTEMPTED
    global _BOLT_MODEL_ERROR

    if _BOLT_MODEL is not None:
        return _BOLT_MODEL

    if _BOLT_MODEL_LOAD_ATTEMPTED:
        return None

    _BOLT_MODEL_LOAD_ATTEMPTED = True

    if not USE_TRAINED_BOLT_MODEL:
        _BOLT_MODEL_ERROR = "Model usage disabled."
        return None

    if YOLO is None:
        _BOLT_MODEL_ERROR = "ultralytics is not installed."
        return None

    if not MODEL_PATH.exists():
        _BOLT_MODEL_ERROR = f"Model not found at {MODEL_PATH}"
        return None

    try:
        _BOLT_MODEL = YOLO(str(MODEL_PATH))
    except Exception as exc:
        _BOLT_MODEL_ERROR = str(exc)
        _BOLT_MODEL = None

    return _BOLT_MODEL


def candidate_matches_previous(candidate, previous_bolt):
    """Check whether a candidate is close enough to the previous bolt."""
    if previous_bolt is None:
        return True

    center_distance = float(
        np.hypot(
            candidate["center"][0] - previous_bolt["center"][0],
            candidate["center"][1] - previous_bolt["center"][1],
        )
    )
    max_distance = max(
        35.0,
        previous_bolt["length"] * TRACK_MATCH_DISTANCE_RATIO,
    )
    length_delta_ratio = abs(candidate["length"] - previous_bolt["length"]) / max(
        previous_bolt["length"], 1.0
    )

    return (
        center_distance <= max_distance
        and length_delta_ratio <= TRACK_MAX_LENGTH_DELTA_RATIO
    )


def tracking_score_bonus(candidate, previous_bolt):
    """Prefer candidates that stay near the previous bolt."""
    if previous_bolt is None:
        return 0.0

    center_distance = float(
        np.hypot(
            candidate["center"][0] - previous_bolt["center"][0],
            candidate["center"][1] - previous_bolt["center"][1],
        )
    )
    max_distance = max(
        35.0,
        previous_bolt["length"] * TRACK_MATCH_DISTANCE_RATIO,
    )
    center_bonus = max(0.0, 1.0 - (center_distance / max(max_distance, 1.0)))

    length_delta_ratio = abs(candidate["length"] - previous_bolt["length"]) / max(
        previous_bolt["length"], 1.0
    )
    size_bonus = max(0.0, 1.0 - length_delta_ratio)

    return (center_bonus * TRACK_CENTER_BONUS) + (size_bonus * TRACK_SIZE_BONUS)


def smooth_bolt(current_bolt, previous_bolt):
    """Blend consecutive detections so the box is visually stable."""
    if previous_bolt is None:
        return current_bolt

    alpha = float(np.clip(TRACK_SMOOTHING, 0.0, 0.95))
    beta = 1.0 - alpha

    smoothed_box = (
        (previous_bolt["box"].astype(np.float32) * alpha)
        + (current_bolt["box"].astype(np.float32) * beta)
    ).astype(np.float32)

    smoothed = current_bolt.copy()
    smoothed["box"] = smoothed_box
    smoothed["center"] = tuple(np.mean(smoothed_box, axis=0))
    smoothed["length"] = (previous_bolt["length"] * alpha) + (current_bolt["length"] * beta)
    return smoothed


def select_best_candidate(candidates, previous_bolt=None):
    """Pick the most stable candidate, preferring continuity when possible."""
    if not candidates:
        return None

    if previous_bolt is not None:
        matching_candidates = [
            candidate
            for candidate in candidates
            if candidate_matches_previous(candidate, previous_bolt)
        ]
        if matching_candidates:
            return max(matching_candidates, key=lambda candidate: candidate["tracking_score"])

    return max(candidates, key=lambda candidate: candidate["score"])


def build_axis_aligned_candidate(x1, y1, x2, y2, frame_shape, confidence, previous_bolt=None):
    """Build a bolt candidate directly from a model bounding box."""
    frame_h, frame_w = frame_shape[:2]

    x1 = float(np.clip(x1, 0, frame_w - 1))
    y1 = float(np.clip(y1, 0, frame_h - 1))
    x2 = float(np.clip(x2, 0, frame_w - 1))
    y2 = float(np.clip(y2, 0, frame_h - 1))

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    area = width * height
    if area < MIN_BOLT_AREA:
        return None

    length = max(width, height)
    aspect_ratio = length / max(min(width, height), 1.0)
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    center_bias = 1.0 - (
        np.hypot(center[0] - frame_w / 2.0, center[1] - frame_h / 2.0)
        / np.hypot(frame_w / 2.0, frame_h / 2.0)
    )

    box = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    score = (
        (confidence * 400.0)
        + (aspect_ratio * 40.0)
        + (center_bias * 10.0)
        + (area / 2500.0)
    )

    candidate = {
        "box": box,
        "center": center,
        "length": float(length),
        "search_zone": (int(x1), int(y1), int(width), int(height)),
        "confidence": float(confidence),
        "score": float(score),
    }
    candidate["tracking_score"] = candidate["score"] + tracking_score_bonus(
        candidate, previous_bolt
    )
    return candidate


def detect_bolt_with_model(frame, gray, previous_bolt=None):
    """Use the trained YOLO model to detect the bolt."""
    model = load_bolt_model()
    if model is None or frame is None or gray is None:
        return None

    detection_gray, scale = prepare_detection_gray(gray)
    detection_frame = prepare_detection_frame(frame, scale)
    detection_previous_bolt = scale_bolt_geometry(previous_bolt, scale, scale)

    if USE_INSPECTION_ZONE:
        _, (zone_x, zone_y, zone_w, zone_h) = get_inspection_zone(detection_gray)
        model_frame = detection_frame[zone_y:zone_y + zone_h, zone_x:zone_x + zone_w]
    else:
        zone_x = 0
        zone_y = 0
        zone_h, zone_w = detection_gray.shape[:2]
        model_frame = detection_frame

    if model_frame.size == 0:
        return None

    try:
        result = model.predict(
            source=model_frame,
            conf=MODEL_CONFIDENCE,
            iou=MODEL_IOU,
            imgsz=MODEL_INPUT_SIZE,
            verbose=False,
        )[0]
    except Exception as exc:
        global _BOLT_MODEL_ERROR
        _BOLT_MODEL_ERROR = str(exc)
        return None

    if result.boxes is None or len(result.boxes) == 0:
        return None

    candidates = []
    for box in result.boxes:
        if box.xyxy is None or len(box.xyxy) == 0:
            continue

        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        class_id = int(box.cls[0]) if box.cls is not None else 0
        if class_id != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 += zone_x
        y1 += zone_y
        x2 += zone_x
        y2 += zone_y

        axis_candidate = build_axis_aligned_candidate(
            x1,
            y1,
            x2,
            y2,
            detection_gray.shape,
            confidence,
            previous_bolt=detection_previous_bolt,
        )
        if axis_candidate is not None:
            candidates.append(axis_candidate)

    detected_bolt = select_best_candidate(candidates, detection_previous_bolt)
    if detected_bolt is None:
        return None

    if scale != 1.0:
        inverse_scale = 1.0 / scale
        detected_bolt = scale_bolt_geometry(
            detected_bolt,
            inverse_scale,
            inverse_scale,
        )

    return detected_bolt
# ================= BOLT DETECTION =================
def detect_bolt(frame, gray, previous_bolt=None):
    """Detect bolt using the trained model only."""
    if frame is None or gray is None:
        return None

    return detect_bolt_with_model(frame, gray, previous_bolt)

# ================= DRAWING =================
def draw_measurements(frame, bolt, fps):
    """Overlay bolt geometry and basic measurement text."""
    result = frame.copy()

    box = bolt["box"].astype(np.int32)
    status_color = (0, 255, 0)
    status_text = "BOLT DETECTED"

    if "search_zone" in bolt and USE_INSPECTION_ZONE:
        zone_x, zone_y, zone_w, zone_h = bolt["search_zone"]
        cv2.rectangle(
            result,
            (zone_x, zone_y),
            (zone_x + zone_w, zone_y + zone_h),
            (120, 120, 120),
            1,
        )

    cv2.drawContours(result, [box], -1, status_color, 2)

    center = tuple(np.int32(bolt["center"]))
    confidence_text = f"Confidence: {float(bolt.get('confidence', 0.0) or 0.0) * 100.0:.1f}%"

    length_mm = bolt["length"] * MM_PER_PIXEL
    length_text = f"Length: {length_mm:.2f} mm"

    cv2.putText(
        result,
        confidence_text,
        (center[0] - 100, center[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        length_text,
        (center[0] - 100, center[1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result

def draw_no_detection(frame, fps):
    """Create result view when no bolt is detected."""
    result = frame.copy()

    if USE_INSPECTION_ZONE:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, (zone_x, zone_y, zone_w, zone_h) = get_inspection_zone(gray)
        cv2.rectangle(
            result,
            (zone_x, zone_y),
            (zone_x + zone_w, zone_y + zone_h),
            (120, 120, 120),
            1,
        )

    cv2.putText(
        result,
        "NO BOLT DETECTED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        result,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


# ================= MAIN APPLICATION =================
def main():
    cap = open_camera(CAMERA_INDEX)
    previous_time = time.time()
    tracked_bolt = None
    lost_frames = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from webcam.")
            break

        if DISPLAY_SCALE != 1.0:
            frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

        raw_view = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        detected_bolt = detect_bolt(frame, gray, tracked_bolt)
        if detected_bolt is not None:
            bolt = smooth_bolt(detected_bolt, tracked_bolt)
            tracked_bolt = bolt
            lost_frames = 0
        elif tracked_bolt is not None and lost_frames < TRACK_HOLD_FRAMES:
            lost_frames += 1
            bolt = tracked_bolt
        else:
            tracked_bolt = None
            lost_frames = 0
            bolt = None

        result_view = draw_no_detection(frame, fps)

        if bolt is not None:
            result_view = draw_measurements(frame, bolt, fps)

        cv2.imshow("Raw Camera View", raw_view)
        cv2.imshow("Final Inspection Result", result_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
