"""Camera abstraction for robot vision capture."""

from __future__ import annotations

import logging
from typing import Any, Optional

from device.camera import open_camera

from ..config import CAMERA_INDEX, MOCK_CAMERA


class Camera:
    """Camera abstraction that defaults to the local webcam."""

    def __init__(self, use_mock: Optional[bool] = None, camera_index: int | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = MOCK_CAMERA if use_mock is None else use_mock
        self.camera_index = CAMERA_INDEX if camera_index is None else camera_index
        self.cap = None

        if self.use_mock:
            self.logger.info("Mock camera mode active; using placeholder frame source.")
        else:
            self.cap = open_camera(self.camera_index)
            self.logger.info("Opened camera index %s for robot vision.", self.camera_index)

    def get_frame(self) -> Any:
        """Return a single camera frame for perception.

        In mock mode this returns a placeholder byte payload for testing.
        """
        if self.use_mock:
            return self._get_mock_frame()
        return self._get_real_frame()

    def _get_mock_frame(self) -> bytes:
        self.logger.debug("Returning mock camera frame payload.")
        return b"MOCK_CAMERA_FRAME"

    def _get_real_frame(self):
        if self.cap is None:
            raise RuntimeError("Camera handle is not initialized.")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Unable to read frame from camera.")
        return frame

    def adjust_view(self, direction: str) -> None:
        """Rotate the camera view when pan/tilt hardware becomes available."""
        self.logger.info("Adjusting camera view: %s", direction)

    def encode_frame(self, frame: Any, image_format: str = ".jpg", quality: int = 85) -> bytes:
        """Encode a frame for backend upload."""
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(f"OpenCV unavailable for frame encoding: {exc}") from exc

        params = []
        if image_format.lower() in {".jpg", ".jpeg"}:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, encoded = cv2.imencode(image_format, frame, params)
        if not ok:
            raise RuntimeError("Failed to encode frame for upload.")
        return encoded.tobytes()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera = Camera()
    frame = camera.get_frame()
    print(f"Frame type: {type(frame).__name__}")
