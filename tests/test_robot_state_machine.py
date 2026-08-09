import unittest

from robot.controller.state_machine import build_target_frame_metadata, plan_target_motion


class RobotTargetingTests(unittest.TestCase):
    def test_turns_left_when_person_is_left_of_centre(self):
        motion = plan_target_motion(
            {
                "bbox": (10, 40, 150, 260),
                "frame_width": 640,
                "frame_height": 480,
            }
        )
        self.assertEqual(motion["action"], "turn_left")

    def test_moves_forward_when_centered_but_too_far(self):
        motion = plan_target_motion(
            {
                "bbox": (240, 100, 400, 260),
                "frame_width": 640,
                "frame_height": 480,
            }
        )
        self.assertEqual(motion["action"], "move_forward")

    def test_holds_when_centered_and_upper_body_fills_frame(self):
        motion = plan_target_motion(
            {
                "bbox": (200, 20, 460, 330),
                "frame_width": 640,
                "frame_height": 480,
            }
        )
        self.assertEqual(motion["action"], "hold")

    def test_target_frame_metadata_uses_bbox_proxy_distance(self):
        metadata = build_target_frame_metadata(
            {
                "bbox": (200, 20, 460, 330),
                "confidence": 0.88,
                "frame_width": 640,
                "frame_height": 480,
            },
            estimated_distance_m=5.0,
        )
        self.assertEqual(metadata["distance_mode"], "bbox_proxy")
        self.assertEqual(metadata["estimated_distance_m"], 5.0)
        self.assertEqual(metadata["bbox"], [200, 20, 460, 330])


if __name__ == "__main__":
    unittest.main()
