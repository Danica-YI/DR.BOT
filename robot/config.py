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
YOLO_CONFIG_DIR = os.getenv("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), "..", ".cache"))
os.environ.setdefault("YOLO_CONFIG_DIR", YOLO_CONFIG_DIR)

MOCK_CAMERA = _bool_from_env("MOCK_CAMERA", False)
MOCK_PERCEPTION = _bool_from_env("MOCK_PERCEPTION", False)
MOCK_MOTORS = _bool_from_env("MOCK_MOTORS", True)
MOCK_TRIAGE = _bool_from_env("MOCK_TRIAGE", False)
ROBOT_SHOW_PREVIEW = _bool_from_env("ROBOT_SHOW_PREVIEW", False)

SYNC_RETRY_INTERVAL = _int_from_env("SYNC_RETRY_INTERVAL", 10)
HEARTBEAT_INTERVAL_SECONDS = _int_from_env("HEARTBEAT_INTERVAL_SECONDS", 300)

REPORT_BATCH_SIZE = _int_from_env("REPORT_BATCH_SIZE", 10)
CAMERA_INDEX = _int_from_env("CAMERA_INDEX", 0)
YOLO_INTERVAL = _int_from_env("YOLO_INTERVAL", 3)
PERSON_STABLE_FRAMES = _int_from_env("PERSON_STABLE_FRAMES", 3)
TARGET_LOCK_FRAMES = _int_from_env("TARGET_LOCK_FRAMES", 2)
TARGET_CAPTURE_ENABLED = _bool_from_env("TARGET_CAPTURE_ENABLED", True)
TARGET_CAPTURE_APPROACH_RATIO = float(os.getenv("TARGET_CAPTURE_APPROACH_RATIO", "0.22"))
TARGET_CAPTURE_DISTANCE_METERS = float(os.getenv("TARGET_CAPTURE_DISTANCE_METERS", "5.0"))
TARGET_CAPTURE_JPEG_QUALITY = _int_from_env("TARGET_CAPTURE_JPEG_QUALITY", 85)
VOICE_TIMEOUT_SECONDS = _int_from_env("VOICE_TIMEOUT_SECONDS", 4)
GESTURE_TIMEOUT_SECONDS = _int_from_env("GESTURE_TIMEOUT_SECONDS", 5)
CLOUD_TRIAGE_ENABLED = _bool_from_env("CLOUD_TRIAGE_ENABLED", False)
CLOUD_TRIAGE_RECORD_SECONDS = _int_from_env("CLOUD_TRIAGE_RECORD_SECONDS", 5)
CLOUD_TRIAGE_MAX_CONSECUTIVE_FAILURES = _int_from_env("CLOUD_TRIAGE_MAX_CONSECUTIVE_FAILURES", 2)
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
SUMMARY_MAX_WORDS = _int_from_env("SUMMARY_MAX_WORDS", 100)
SUMMARY_REQUEST_TIMEOUT_SECONDS = _int_from_env("SUMMARY_REQUEST_TIMEOUT_SECONDS", 30)
CLOUD_CHAT_MAX_RETRIES = _int_from_env("CLOUD_CHAT_MAX_RETRIES", 2)
CLOUD_SUMMARY_ENABLED = _bool_from_env("CLOUD_SUMMARY_ENABLED", False)
CLOUD_SUMMARY_BASE_URL = os.getenv(
    "CLOUD_SUMMARY_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
CLOUD_SUMMARY_MODEL = os.getenv("CLOUD_SUMMARY_MODEL", "gemini-3.6-flash")
CLOUD_SUMMARY_API_KEY = (
    os.getenv("CLOUD_SUMMARY_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
EDGE_SUMMARY_ENABLED = _bool_from_env("EDGE_SUMMARY_ENABLED", True)
EDGE_SUMMARY_MODEL_PATH = os.getenv("EDGE_SUMMARY_MODEL_PATH", "google/flan-t5-small")
EDGE_SUMMARY_MODEL_TYPE = os.getenv("EDGE_SUMMARY_MODEL_TYPE", "text2text-generation")
