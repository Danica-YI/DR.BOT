"""Camera abstraction for robot vision capture."""

import logging
from typing import Optional

from ..config import MOCK_CAMERA


class Camera:
    """Basic camera abstraction with mock mode for development."""

    def __init__(self, use_mock: Optional[bool] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = MOCK_CAMERA if use_mock is None else use_mock

        if self.use_mock:
            self.logger.info("Mock camera mode active; using placeholder frame source.")
        else:
            self.logger.warning(
                "Real camera mode requested but no real camera implementation is available. "
                "Falling back to mock mode."
            )
            self.use_mock = True

    def get_frame(self) -> bytes:
        """Return a single camera frame for perception.

        In mock mode this returns a placeholder byte payload that
        downstream consumers can safely inspect.
        """
        if self.use_mock:
            return self._get_mock_frame()
        return self._get_real_frame()

    def _get_mock_frame(self) -> bytes:
        self.logger.debug("Returning mock camera frame payload.")
        return b"MOCK_CAMERA_FRAME"

    def _get_real_frame(self) -> bytes:
        raise NotImplementedError(
            "Real Raspberry Pi camera support is not implemented yet."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera = Camera()
    frame = camera.get_frame()
    print(f"Mock frame length: {len(frame)}")
