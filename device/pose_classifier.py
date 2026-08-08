import os
import math
import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mediapipe_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
except Exception as exc:
    mp = None
    mediapipe_python = None
    vision = None
    PoseLandmarker = None
    PoseLandmarkerOptions = None
    RunningMode = None
    mp_import_error = exc


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _classify_status_from_landmarks(landmarks):
    if not landmarks:
        return None

    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    # Crossed arms partially occlude wrists, so use a lower wrist threshold
    # while keeping the head/shoulder/elbow anchors reliable.
    anchors = (nose, left_shoulder, right_shoulder, left_elbow, right_elbow)
    if min(getattr(point, "visibility", 1.0) for point in anchors) < 0.5:
        return None
    if min(getattr(left_wrist, "visibility", 1.0), getattr(right_wrist, "visibility", 1.0)) < 0.25:
        return None

    both_hands_over_head = (
        left_wrist.y < nose.y and
        right_wrist.y < nose.y and
        left_elbow.y < left_shoulder.y and
        right_elbow.y < right_shoulder.y
    )

    # NO accepts an X across the chest or waist. These relative tests are
    # mirror-safe and scale with the person's size in the frame.
    torso_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_width = max(_distance(left_shoulder, right_shoulder), 0.1)
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    wrists_cross_torso = (
        (left_wrist.x - torso_center_x) * (left_shoulder.x - torso_center_x) < 0 and
        (right_wrist.x - torso_center_x) * (right_shoulder.x - torso_center_x) < 0
    )
    wrist_order_crossed = (
        (left_wrist.x - right_wrist.x) * (left_elbow.x - right_elbow.x) < 0
    )
    wrists_near_opposite_arms = (
        _distance(left_wrist, right_elbow) < shoulder_width * 0.75 and
        _distance(right_wrist, left_elbow) < shoulder_width * 0.75
    )
    wrists_in_torso_zone = (
        min(left_wrist.y, right_wrist.y) > shoulder_y - shoulder_width * 0.35 and
        max(left_wrist.y, right_wrist.y) < shoulder_y + shoulder_width * 1.6
    )
    arms_crossed = (
        (wrists_cross_torso or wrist_order_crossed or wrists_near_opposite_arms) and
        wrists_in_torso_zone and
        abs(left_wrist.y - right_wrist.y) < 0.25 and
        abs(left_wrist.x - right_wrist.x) < shoulder_width * 1.1
    )

    left_hand_up = left_wrist.y < left_shoulder.y - 0.05
    right_hand_up = right_wrist.y < right_shoulder.y - 0.05

    if both_hands_over_head:
        return "ok"
    if arms_crossed:
        return "no"
    if left_hand_up != right_hand_up:
        return "resource"
    return None


def _find_model_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_root, "pose_landmarker_lite.task"),
        os.path.join(os.getcwd(), "pose_landmarker_lite.task"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


class PoseClassifier:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if PoseLandmarker is None or PoseLandmarkerOptions is None or RunningMode is None or mp is None:
            raise RuntimeError(f"Mediapipe import failed: {mp_import_error}")

        model_path = _find_model_path()
        if not model_path:
            raise FileNotFoundError("pose_landmarker_lite.task not found. Download it into the project root first.")

        options = PoseLandmarkerOptions(
            base_options=mediapipe_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        self.pose = PoseLandmarker.create_from_options(options)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def classify(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.pose.detect(image)
        if not result.pose_landmarks:
            return None

        pose_landmarks = result.pose_landmarks[0]
        landmarks = getattr(pose_landmarks, "landmark", pose_landmarks)
        return _classify_status_from_landmarks(landmarks)

    def close(self):
        self.pose.close()
