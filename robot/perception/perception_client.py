"""Client for perception model interaction."""

from __future__ import annotations

import logging
from typing import Any

from ..config import MOCK_PERCEPTION, YOLO_MODEL_PATH


class _TeammatePerceptionAdapter:
    """Wrap the teammate AI implementation behind a stable robot interface."""

    def __init__(self, model_name: str) -> None:
        from device.controller import PersonDetector

        self.detector = PersonDetector(model_name=model_name)
        self._gesture_classifier = None

    def detect_person(self, frame: Any) -> tuple[int, int, int, int] | None:
        return self.detector.find_person(frame)

    def classify_gesture(self, frame: Any) -> str | None:
        if self._gesture_classifier is None:
            from device.pose_classifier import PoseClassifier

            self._gesture_classifier = PoseClassifier()
        return self._gesture_classifier.classify(frame)

    def close(self) -> None:
        self.detector.close()
        if self._gesture_classifier is not None:
            self._gesture_classifier.close()


class PerceptionClient:
    """Perception client that preserves a stable robot-facing API."""

    def __init__(
        self,
        use_mock: bool | None = None,
        model_name: str | None = None,
        adapter: _TeammatePerceptionAdapter | None = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = MOCK_PERCEPTION if use_mock is None else use_mock
        self.adapter = None

        if self.use_mock:
            self.logger.info("Mock perception client active; returning placeholder results.")
        else:
            self.adapter = adapter or _TeammatePerceptionAdapter(model_name or YOLO_MODEL_PATH)
            self.logger.info("Real perception adapter initialized with %s.", model_name or YOLO_MODEL_PATH)

    def detect_person(self, frame: Any) -> bool:
        """Detect whether a person is present in the given frame."""
        return self.locate_person(frame) is not None

    def locate_person(self, frame: Any) -> dict[str, Any] | None:
        """Return the most confident person detection using a stable dict format."""
        if self.use_mock:
            if not self._mock_person_detected(frame):
                return None
            return {
                "bbox": (120, 60, 520, 420),
                "confidence": 1.0,
                "frame_width": 640,
                "frame_height": 480,
            }

        if self.adapter is None:
            raise RuntimeError("Perception adapter is not initialized.")

        self._validate_real_frame(frame)
        detection = self.adapter.detect_person(frame)
        if not detection:
            return None

        x1, y1, x2, y2 = detection
        frame_height, frame_width = frame.shape[:2]
        return {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "confidence": 1.0,
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
        }

    def classify_gesture(self, frame: Any) -> dict[str, Any]:
        """Classify a gesture while hiding the underlying ML framework."""
        if self.use_mock:
            self.logger.debug("Mock classify_gesture called.")
            return self._mock_gesture_result(frame)

        if self.adapter is None:
            raise RuntimeError("Perception adapter is not initialized.")

        self._validate_real_frame(frame)
        gesture = self.adapter.classify_gesture(frame)
        return {
            "gesture": gesture,
            "confidence": 1.0 if gesture else 0.0,
            "detected": gesture is not None,
            "notes": "device.pose_classifier adapter output",
        }

    def _validate_real_frame(self, frame: Any) -> None:
        if not hasattr(frame, "shape"):
            raise TypeError(
                "Real perception expects an image frame with shape metadata; "
                f"received {type(frame).__name__}."
            )

    def _mock_person_detected(self, frame: Any) -> bool:
        return isinstance(frame, bytes) and frame == b"MOCK_CAMERA_FRAME"

    def _mock_gesture_result(self, frame: Any) -> dict[str, Any]:
        del frame
        return {
            "gesture": "wave",
            "confidence": 0.75,
            "detected": True,
            "notes": "mock gesture output",
        }

    def close(self) -> None:
        if self.adapter is not None:
            self.adapter.close()
