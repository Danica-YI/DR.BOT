"""Heartbeat-only entrypoint that reuses the main robot runner."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("MOCK_CAMERA", "true")
os.environ.setdefault("MOCK_PERCEPTION", "true")
os.environ.setdefault("MOCK_TRIAGE", "true")

from robot.config import BACKEND_URL, HEARTBEAT_INTERVAL_SECONDS, ROBOT_ID, SOFTWARE_VERSION
from robot.main import run as run_main

PERIODIC_TEST_REPORT_INTERVAL_SECONDS = 8 * 60
DEFAULT_TEST_LOCATION = {"latitude": -27.4698, "longitude": 153.0251}


def run() -> None:
    """Run the main robot loop with heartbeat timing overridden for testing."""
    print(
        f"Starting heartbeat-only runner for {ROBOT_ID} version {SOFTWARE_VERSION} "
        f"-> {BACKEND_URL} with heartbeat every {HEARTBEAT_INTERVAL_SECONDS} seconds "
        f"and report every {PERIODIC_TEST_REPORT_INTERVAL_SECONDS} seconds "
        f"from fixed test location {DEFAULT_TEST_LOCATION['latitude']}, "
        f"{DEFAULT_TEST_LOCATION['longitude']}"
    )
    run_main(
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL_SECONDS,
        periodic_test_report_interval_seconds=PERIODIC_TEST_REPORT_INTERVAL_SECONDS,
        static_location=DEFAULT_TEST_LOCATION,
    )


if __name__ == "__main__":
    run()
