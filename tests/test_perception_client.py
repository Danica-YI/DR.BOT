import unittest

import numpy as np

from robot.perception.perception_client import PerceptionClient


class _AdapterStub:
    def __init__(self, detection=None, gesture=None):
        self.detection = detection
        self.gesture = gesture
        self.closed = False

    def detect_person(self, frame):
        del frame
        return self.detection

    def classify_gesture(self, frame):
        del frame
        return self.gesture

    def close(self):
        self.closed = True


class PerceptionClientTests(unittest.TestCase):
    def test_mock_mode_returns_placeholder_detection(self):
        client = PerceptionClient(use_mock=True)

        detection = client.locate_person(b"MOCK_CAMERA_FRAME")

        self.assertEqual(detection["bbox"], (120, 60, 520, 420))
        self.assertEqual(detection["frame_width"], 640)
        self.assertEqual(detection["frame_height"], 480)

    def test_real_mode_adapts_teammate_detection_to_robot_format(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        client = PerceptionClient(use_mock=False, adapter=_AdapterStub(detection=(10, 20, 110, 220)))

        detection = client.locate_person(frame)

        self.assertEqual(
            detection,
            {
                "bbox": (10, 20, 110, 220),
                "confidence": 1.0,
                "frame_width": 640,
                "frame_height": 480,
            },
        )

    def test_real_mode_converts_gesture_to_dict_contract(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        client = PerceptionClient(use_mock=False, adapter=_AdapterStub(gesture="resource"))

        result = client.classify_gesture(frame)

        self.assertEqual(result["gesture"], "resource")
        self.assertEqual(result["confidence"], 1.0)
        self.assertTrue(result["detected"])

    def test_real_mode_rejects_non_image_frames(self):
        client = PerceptionClient(use_mock=False, adapter=_AdapterStub(detection=(1, 2, 3, 4)))

        with self.assertRaises(TypeError):
            client.locate_person(b"MOCK_CAMERA_FRAME")

    def test_close_forwards_to_adapter(self):
        adapter = _AdapterStub()
        client = PerceptionClient(use_mock=False, adapter=adapter)

        client.close()

        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
