"""Shared robot configuration for the DR-01 rescue robot."""

import os
import time

# Record the program startup time for heartbeat interval calculations
PROGRAM_STARTUP_TIME = time.time()


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


ROBOT_ID = os.getenv("ROBOT_ID", "DR-01")
ROBOT_NAME = os.getenv("ROBOT_NAME", "RescueBot-01")
ROBOT_TYPE = os.getenv("ROBOT_TYPE", "ground_robot")
SOFTWARE_VERSION = os.getenv("SOFTWARE_VERSION", "0.1.0")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

MOCK_CAMERA = _bool_from_env("MOCK_CAMERA", True)
MOCK_PERCEPTION = _bool_from_env("MOCK_PERCEPTION", True)
MOCK_MOTORS = _bool_from_env("MOCK_MOTORS", True)
MOCK_TRIAGE = _bool_from_env("MOCK_TRIAGE", True)

SYNC_RETRY_INTERVAL = _int_from_env("SYNC_RETRY_INTERVAL", 10)
HEARTBEAT_INTERVAL_SECONDS = _int_from_env("HEARTBEAT_INTERVAL_SECONDS", 300)

REPORT_BATCH_SIZE = _int_from_env("REPORT_BATCH_SIZE", 10)
