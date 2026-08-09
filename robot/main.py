"""Robot application entrypoint for DR-01.

This module is intended to be executed from the package root with:

    python -m robot.main

It also supports direct script execution when the root directory is on PYTHONPATH.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from robot.config import ROBOT_ID, SOFTWARE_VERSION
from robot.controller.state_machine import StateMachine

CYCLE_DELAY_SECONDS = 1.0


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run(
    heartbeat_interval_seconds: int | None = None,
    periodic_test_report_interval_seconds: int | None = None,
    static_location: dict[str, float] | None = None,
) -> None:
    """Load configuration, initialize components, and start robot execution."""
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting robot %s version %s", ROBOT_ID, SOFTWARE_VERSION)
    state_machine = StateMachine(
        heartbeat_interval=heartbeat_interval_seconds,
        periodic_test_report_interval=periodic_test_report_interval_seconds,
        static_location=static_location,
    )

    try:
        while True:
            state_machine.run_once()
            time.sleep(CYCLE_DELAY_SECONDS)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received; shutting down robot.")
    except Exception as exc:
        logger.exception("Unexpected error in robot main loop: %s", exc)
    finally:
        logger.info("Executing safe shutdown sequence.")
        try:
            state_machine.motors.stop_robot()
        except Exception as exc:
            logger.exception("Failed to stop motors during shutdown: %s", exc)
        try:
            state_machine.camera.release()
        except Exception as exc:
            logger.exception("Failed to release camera during shutdown: %s", exc)
        try:
            state_machine.perception.close()
        except Exception as exc:
            logger.exception("Failed to close perception during shutdown: %s", exc)
        try:
            state_machine.triage.close()
        except Exception as exc:
            logger.exception("Failed to close triage during shutdown: %s", exc)
        logger.info("Robot shutdown complete.")


if __name__ == "__main__":
    run()
