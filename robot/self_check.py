"""Runtime self-check for real robot integrations."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".cache"))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        detail = fn()
        return CheckResult(name, True, detail)
    except Exception as exc:
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


def check_python_modules() -> list[CheckResult]:
    modules = [
        "flask",
        "requests",
        "cv2",
        "ultralytics",
        "mediapipe",
        "sounddevice",
        "pyttsx3",
        "vosk",
    ]
    results: list[CheckResult] = []
    for module_name in modules:
        def _import_module(name: str = module_name) -> str:
            module = importlib.import_module(name)
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, "__version__", "unknown")
            return f"installed ({version})"

        results.append(_run_check(f"module:{module_name}", _import_module))
    return results


def check_files() -> list[CheckResult]:
    required = [
        PROJECT_ROOT / "yolov8n.pt",
        PROJECT_ROOT / "pose_landmarker_lite.task",
        PROJECT_ROOT / "vosk-model-small-en-us-0.15",
    ]
    results: list[CheckResult] = []
    for path in required:
        def _check_path(target: Path = path) -> str:
            if not target.exists():
                raise FileNotFoundError(target)
            if target.is_dir():
                return "present (directory)"
            return "present (file)"

        results.append(_run_check(f"path:{path.name}", _check_path))
    return results


def check_backend_port() -> CheckResult:
    from robot.config import BACKEND_URL
    from urllib.parse import urlparse

    parsed = urlparse(BACKEND_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    def _connect() -> str:
        with socket.create_connection((host, port), timeout=2):
            return f"reachable at {host}:{port}"

    return _run_check("backend", _connect)


def check_camera() -> CheckResult:
    def _open_camera() -> str:
        from robot.camera.camera import Camera

        camera = Camera()
        try:
            frame = camera.get_frame()
            shape = getattr(frame, "shape", None)
            return f"frame captured shape={shape}"
        finally:
            camera.release()

    return _run_check("camera", _open_camera)


def check_speaker() -> CheckResult:
    def _speaker() -> str:
        from robot.hardware.speaker import Speaker

        speaker = Speaker()
        ok = speaker.say("Robot self check speaker test.")
        if not ok:
            raise RuntimeError("local text to speech unavailable")
        return "local TTS spoke test phrase"

    return _run_check("speaker", _speaker)


def check_microphone() -> CheckResult:
    def _mic() -> str:
        from device.voice import OfflineVoice

        voice = OfflineVoice(enabled=True)
        if not voice.recognition_available:
            raise RuntimeError(voice.error or "speech recognition unavailable")
        import sounddevice as sd

        device = sd.query_devices(None, "input")
        return f"input device ready ({device['name']})"

    return _run_check("microphone", _mic)


def check_perception() -> CheckResult:
    def _perception() -> str:
        from robot.camera.camera import Camera
        from robot.perception.perception_client import PerceptionClient

        camera = Camera()
        perception = PerceptionClient()
        try:
            frame = camera.get_frame()
            detection = perception.locate_person(frame)
            return "initialized; detection=" + ("person" if detection else "none")
        finally:
            camera.release()
            perception.close()

    return _run_check("perception", _perception)


def check_pose_classifier() -> CheckResult:
    def _pose() -> str:
        from robot.camera.camera import Camera
        from device.pose_classifier import PoseClassifier

        camera = Camera()
        classifier = PoseClassifier()
        try:
            frame = camera.get_frame()
            result = classifier.classify(frame)
            return f"initialized; classification={result}"
        finally:
            camera.release()
            classifier.close()

    return _run_check("pose_classifier", _pose)


def check_cloud_summary() -> CheckResult:
    def _cloud() -> str:
        from robot.config import (
            CLOUD_SUMMARY_API_KEY,
            CLOUD_SUMMARY_BASE_URL,
            CLOUD_SUMMARY_ENABLED,
            CLOUD_SUMMARY_MODEL,
        )

        if not CLOUD_SUMMARY_ENABLED:
            raise RuntimeError("cloud summary disabled")
        if not CLOUD_SUMMARY_API_KEY:
            raise RuntimeError("cloud API key not configured")
        return f"configured base_url={CLOUD_SUMMARY_BASE_URL} model={CLOUD_SUMMARY_MODEL}"

    return _run_check("cloud_summary", _cloud)


def main() -> int:
    checks: list[CheckResult] = []
    checks.extend(check_python_modules())
    checks.extend(check_files())
    checks.append(check_backend_port())
    checks.append(check_camera())
    checks.append(check_speaker())
    checks.append(check_microphone())
    checks.append(check_perception())
    checks.append(check_pose_classifier())
    checks.append(check_cloud_summary())

    failures = [item for item in checks if not item.ok]
    for item in checks:
        status = "PASS" if item.ok else "FAIL"
        print(f"[{status}] {item.name}: {item.detail}")

    print()
    print(f"Summary: {len(checks) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
