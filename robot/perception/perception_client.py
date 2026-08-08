"""Client for perception model interaction."""

import logging
from typing import Any


class PerceptionClient:
    """Mock perception client for robot integration.

    This interface is intentionally framework-agnostic and only provides
    placeholder behavior until the AI/perception team supplies real models.
    """

    def __init__(self, use_mock: bool = True) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = use_mock

        if self.use_mock:
            self.logger.info("Mock perception client active; returning placeholder results.")
        else:
            self.logger.warning(
                "Real perception mode requested, but no real model is integrated yet. "
                "Use mock behavior until the AI teammate implements inference."
            )
            self.use_mock = True

    def detect_person(self, frame: Any) -> bool:
        """Detect whether a person is present in the given frame.

        Args:
            frame: A camera frame or placeholder payload.

        Returns:
            True when a person is mocked as detected, False otherwise.
        """
        self.logger.debug("Mock detect_person called.")
        return self._mock_person_detected(frame)

    def classify_gesture(self, frame: Any) -> dict:
        """Classify a gesture from the given frame.

        Args:
            frame: A camera frame or placeholder payload.

        Returns:
            A simple mock gesture classification result.
        """
        self.logger.debug("Mock classify_gesture called.")
        return self._mock_gesture_result(frame)

    def _mock_person_detected(self, frame: Any) -> bool:
        if isinstance(frame, bytes) and frame == b"MOCK_CAMERA_FRAME":
            return True
        return False

    def _mock_gesture_result(self, frame: Any) -> dict:
        return {
            "gesture": "wave",
            "confidence": 0.75,
            "notes": "mock gesture output"
        }

    # AI teammate integration point
    # Replace _mock_person_detected and _mock_gesture_result with real model inference.
    # Example:
    #   def detect_person(self, frame: Any) -> bool:
    #       return self.model.predict_person(frame)
    #   def classify_gesture(self, frame: Any) -> dict:
    #       return self.model.predict_gesture(frame)
