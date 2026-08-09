"""Motor control abstractions."""

import logging
from typing import Callable

from ..config import MOCK_MOTORS


class MotorController:
    """Safe motor control interface for robot movement."""

    def __init__(self, executor: Callable[[str], None] | None = None, use_mock: bool | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = MOCK_MOTORS if use_mock is None else use_mock
        self.executor = executor or self._mock_execute
        if self.use_mock:
            self.logger.info("Motor controller initialized in mock mode.")
        else:
            self.logger.warning("Motor controller is set to real mode, but no hardware executor is configured.")

    def move_forward(self) -> None:
        self._perform_action("Moving forward")

    def move_backward(self) -> None:
        self._perform_action("Moving backward")

    def turn_left(self) -> None:
        self._perform_action("Turning left")

    def turn_right(self) -> None:
        self._perform_action("Turning right")

    def stop_robot(self) -> None:
        self._perform_action("STOP", safe=True)

    def _perform_action(self, action: str, safe: bool = False) -> None:
        try:
            self.logger.info("[MOTOR] %s", action)
            self.executor(action)
        except Exception as exc:
            self.logger.error("Motor action failed: %s", exc, exc_info=True)
            if not safe:
                self.logger.info("Falling back to safe stop_robot() due to motor error.")
                self._perform_action("STOP", safe=True)

    def _mock_execute(self, action: str) -> None:
        if self.use_mock:
            self.logger.debug("Mock motor execute: %s", action)
            return
        raise RuntimeError("No real motor executor configured for action: %s" % action)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    controller = MotorController()
    controller.move_forward()
    controller.turn_left()
    controller.stop_robot()
