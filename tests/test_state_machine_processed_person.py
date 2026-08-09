import unittest
from unittest.mock import patch

import numpy as np

from robot.controller.state_machine import (
    TARGET_LOCK_FRAMES,
    PersonFingerprint,
    RobotState,
    StateMachine,
    build_person_fingerprint,
)


class _FakeCamera:
    use_mock = False

    def __init__(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame.copy()

    def encode_frame(self, frame, image_format=".jpg", quality=85):
        del frame, image_format, quality
        return b"encoded-frame"

    def adjust_view(self, direction):
        del direction

    def release(self):
        return None


class _FakePerception:
    use_mock = False

    def __init__(self, detection):
        self.detection = detection

    def locate_person(self, frame):
        del frame
        return dict(self.detection)

    def close(self):
        return None


class _FakeMotors:
    def move_forward(self):
        return None

    def turn_left(self):
        return None

    def turn_right(self):
        return None

    def stop_robot(self):
        return None


class _FakeTriage:
    use_mock = False

    def __init__(self, assessment=None, exc=None):
        self.assessment = assessment or {
            "priority": "medical",
            "timestamp": "2026-08-09T12:00:00+10:00",
        }
        self.exc = exc
        self.calls = 0

    def run_triage(self, frame_provider=None):
        del frame_provider
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return dict(self.assessment)

    def close(self):
        return None


class _FakeApiClient:
    base_url = "http://127.0.0.1:5000"

    def send_heartbeat(self, heartbeat):
        del heartbeat
        return {"success": True}


class ProcessedPersonStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame[60:420, 160:480] = (40, 120, 210)
        self.detection = {
            "bbox": (160, 60, 480, 420),
            "confidence": 1.0,
            "frame_width": 640,
            "frame_height": 480,
        }
        self.location = {"latitude": -27.4698, "longitude": 153.0251}

    def _build_machine(self, initial_state, triage=None):
        return StateMachine(
            camera=_FakeCamera(self.frame),
            perception=_FakePerception(self.detection),
            motors=_FakeMotors(),
            triage=triage or _FakeTriage(),
            api_client=_FakeApiClient(),
            initial_state=initial_state,
            heartbeat_interval=999999,
            static_location=self.location,
        )

    def test_aligned_person_builds_runtime_fingerprint_before_interaction(self):
        machine = self._build_machine(RobotState.PERSON_DETECTED)
        machine.target_lock_frames = TARGET_LOCK_FRAMES - 1

        with patch("robot.controller.state_machine.post_target_frame", return_value=(False, "disabled")):
            machine._handle_person_detected()

        self.assertEqual(machine.current_state, RobotState.STOPPED)
        self.assertIsInstance(machine.current_person_fingerprint, PersonFingerprint)

    def test_processed_person_is_skipped_before_existing_interaction(self):
        machine = self._build_machine(RobotState.STOPPED)
        fingerprint = build_person_fingerprint(self.frame, self.detection)
        machine.current_person_fingerprint = fingerprint
        machine.processed_person_registry.append(fingerprint)

        with patch.object(machine, "_perform_interaction") as mocked_interaction:
            machine._prepare_interaction()

        mocked_interaction.assert_not_called()
        self.assertEqual(machine.current_state, RobotState.SEARCHING)
        self.assertIsNone(machine.current_person_fingerprint)

    def test_new_person_still_enters_existing_interaction_flow(self):
        machine = self._build_machine(RobotState.STOPPED)
        machine.current_person_fingerprint = build_person_fingerprint(self.frame, self.detection)

        with patch.object(machine, "_perform_interaction") as mocked_interaction:
            machine._prepare_interaction()

        mocked_interaction.assert_called_once()
        self.assertEqual(machine.current_state, RobotState.INTERACTING)

    def test_person_is_marked_only_after_report_completion(self):
        machine = self._build_machine(RobotState.REPORTING)
        fingerprint = build_person_fingerprint(self.frame, self.detection)
        machine.current_person_fingerprint = fingerprint
        machine.assessment = {
            "priority": "medical",
            "timestamp": "2026-08-09T12:00:00+10:00",
        }

        with patch("robot.controller.state_machine.flush_triage_queue", return_value=(True, {"queued": 0})):
            with patch("robot.controller.state_machine.post_triage_assessment", return_value=(True, {"success": True})):
                machine._report()

        self.assertTrue(machine._is_person_processed(fingerprint))
        self.assertEqual(machine.current_state, RobotState.SEARCHING)

    def test_failed_interaction_does_not_mark_person_processed(self):
        machine = self._build_machine(
            RobotState.INTERACTING,
            triage=_FakeTriage(exc=RuntimeError("triage failed")),
        )
        machine.current_person_fingerprint = build_person_fingerprint(self.frame, self.detection)

        machine._perform_interaction()

        self.assertEqual(machine.processed_person_registry, [])


if __name__ == "__main__":
    unittest.main()
