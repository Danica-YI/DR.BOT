import os
import tempfile
import unittest
from types import SimpleNamespace

from device.pose_classifier import _classify_status_from_landmarks
from device.camera import status_color
from device.main import initial_response_route, second_response_route
from device.triage_flow import OfflineTriageStore, build_triage_assessment
from device.response import GestureStabilizer
from device.voice import _answer_from_text


class TriageFlowTests(unittest.TestCase):
    @staticmethod
    def _pose_landmarks(**points):
        landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
        indexes = {
            "nose": 0, "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14,
            "left_wrist": 15, "right_wrist": 16,
        }
        for name, coordinates in points.items():
            landmarks[indexes[name]] = SimpleNamespace(x=coordinates[0], y=coordinates[1], visibility=1.0)
        return landmarks

    def test_build_triage_assessment_marks_medical_priority(self):
        assessment = build_triage_assessment(
            initial_status="resource",
            response_type="responding",
            can_walk=False,
            heavy_bleeding=True,
            breathing_difficulty=False,
            trapped=True,
        )

        self.assertEqual(assessment["priority"], "medical")
        self.assertIn("heavy bleeding", assessment["reasons"][0])
        self.assertGreaterEqual(assessment["confidence"], 0.8)

    def test_offline_store_enqueues_and_drain_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "offline.db")
            store = OfflineTriageStore(db_path)
            assessment = build_triage_assessment(initial_status="ok", response_type="responding", can_walk=True)
            store.enqueue(assessment)

            self.assertEqual(store.pending_count(), 1)
            pending = store.get_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0]["payload"]["priority"], "ok")

    def test_gesture_requires_multiple_consistent_frames(self):
        stabilizer = GestureStabilizer(window_size=3, required_matches=2)
        self.assertIsNone(stabilizer.add("no"))
        result = stabilizer.add("no")
        self.assertEqual(result.answer, "NO")
        self.assertEqual(result.source, "gesture")

    def test_stale_history_does_not_accept_missing_current_gesture(self):
        stabilizer = GestureStabilizer(window_size=3, required_matches=2)
        self.assertIsNone(stabilizer.add("ok"))
        self.assertEqual(stabilizer.add("ok").answer, "YES")
        self.assertIsNone(stabilizer.add(None))

    def test_resource_gesture_maps_to_help(self):
        stabilizer = GestureStabilizer()
        self.assertIsNone(stabilizer.add("resource"))
        result = stabilizer.add("resource")
        self.assertEqual(result.answer, "HELP")

    def test_offline_voice_vocabulary_maps_to_answers(self):
        self.assertEqual(_answer_from_text("yeah"), "YES")
        self.assertEqual(_answer_from_text("nope"), "NO")
        self.assertEqual(_answer_from_text("help"), "HELP")
        self.assertIsNone(_answer_from_text("maybe"))
        self.assertEqual(_answer_from_text("I can hear you"), "YES")
        self.assertEqual(_answer_from_text("I can't"), "NO")

    def test_chinese_voice_vocabulary_maps_to_answers(self):
        self.assertEqual(_answer_from_text("可以", "zh"), "YES")
        self.assertEqual(_answer_from_text("不能走路", "zh"), "NO")
        self.assertEqual(_answer_from_text("我沒有", "zh"), "NO")
        self.assertEqual(_answer_from_text("我需要帮助", "zh"), "HELP")
        self.assertIsNone(_answer_from_text("不知道", "zh"))
        self.assertEqual(_answer_from_text("是", "zh"), "YES")
        self.assertEqual(_answer_from_text("听不到", "zh"), "NO")

    def test_initial_contact_routes(self):
        self.assertEqual(initial_response_route("YES"), "VOICE_MODE")
        self.assertEqual(initial_response_route("NO"), "COMPLETE")
        self.assertEqual(initial_response_route(None), "SECOND_PROMPT")

    def test_second_window_routes(self):
        self.assertEqual(second_response_route(voice_answer="YES"), "VOICE_MODE")
        self.assertEqual(second_response_route(gesture_answer="resource"), "GESTURE_MODE")
        self.assertEqual(second_response_route(expired=True), "NO_RESPONSE")

    def test_required_status_colours(self):
        self.assertEqual(status_color("yes"), (0, 200, 0))
        self.assertEqual(status_color("no"), (0, 0, 255))
        self.assertEqual(status_color("no_response"), (0, 0, 255))
        self.assertEqual(status_color("resource"), (0, 165, 255))
        self.assertEqual(status_color("result"), (0, 165, 255))

    def test_english_and_chinese_answers_produce_same_medical_state(self):
        english_can_walk = _answer_from_text("no", "en") == "YES"
        chinese_can_walk = _answer_from_text("不能走路", "zh") == "YES"
        self.assertEqual(english_can_walk, chinese_can_walk)

        english = build_triage_assessment("ok", "responding", english_can_walk)
        chinese = build_triage_assessment("ok", "responding", chinese_can_walk)
        self.assertEqual(english["priority"], "medical")
        self.assertEqual(chinese["priority"], "medical")

    def test_english_and_chinese_safe_answers_produce_ok_state(self):
        english = build_triage_assessment(
            "ok", "responding", _answer_from_text("yes", "en") == "YES"
        )
        chinese = build_triage_assessment(
            "ok", "responding", _answer_from_text("可以走路", "zh") == "YES"
        )
        self.assertEqual(english["priority"], "ok")
        self.assertEqual(chinese["priority"], "ok")

    def test_hands_crossed_to_opposite_shoulders_are_no(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.4), right_shoulder=(0.3, 0.4),
            left_elbow=(0.65, 0.55), right_elbow=(0.35, 0.55),
            left_wrist=(0.35, 0.48), right_wrist=(0.65, 0.48),
        )
        self.assertEqual(_classify_status_from_landmarks(landmarks), "no")

    def test_hands_crossed_at_waist_are_no(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.4), right_shoulder=(0.3, 0.4),
            left_elbow=(0.65, 0.58), right_elbow=(0.35, 0.58),
            left_wrist=(0.35, 0.75), right_wrist=(0.65, 0.75),
        )
        self.assertEqual(_classify_status_from_landmarks(landmarks), "no")

    def test_crossed_wrist_order_is_no_without_elbows(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.4), right_shoulder=(0.3, 0.4),
            left_elbow=(0.68, 0.58), right_elbow=(0.32, 0.58),
            left_wrist=(0.46, 0.62), right_wrist=(0.54, 0.68),
        )
        self.assertEqual(_classify_status_from_landmarks(landmarks), "no")

    def test_two_hands_above_head_are_yes(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.45), right_shoulder=(0.3, 0.45),
            left_elbow=(0.65, 0.35), right_elbow=(0.35, 0.35),
            left_wrist=(0.7, 0.1), right_wrist=(0.3, 0.1),
        )
        self.assertEqual(_classify_status_from_landmarks(landmarks), "ok")

    def test_one_hand_above_shoulder_requests_resource(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.45), right_shoulder=(0.3, 0.45),
            left_elbow=(0.65, 0.35), right_elbow=(0.35, 0.55),
            left_wrist=(0.7, 0.3), right_wrist=(0.3, 0.65),
        )
        self.assertEqual(_classify_status_from_landmarks(landmarks), "resource")

    def test_right_hand_alone_requests_resource(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.45), right_shoulder=(0.3, 0.45),
            left_elbow=(0.65, 0.55), right_elbow=(0.35, 0.35),
            left_wrist=(0.7, 0.65), right_wrist=(0.3, 0.3),
        )
        self.assertEqual(_classify_status_from_landmarks(landmarks), "resource")

    def test_neutral_pose_is_unknown(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.4), right_shoulder=(0.3, 0.4),
            left_elbow=(0.7, 0.55), right_elbow=(0.3, 0.55),
            left_wrist=(0.7, 0.7), right_wrist=(0.3, 0.7),
        )
        self.assertIsNone(_classify_status_from_landmarks(landmarks))

    def test_low_visibility_pose_is_unknown(self):
        landmarks = self._pose_landmarks(
            nose=(0.5, 0.2),
            left_shoulder=(0.7, 0.45), right_shoulder=(0.3, 0.45),
            left_wrist=(0.7, 0.1), right_wrist=(0.3, 0.1),
        )
        landmarks[15].visibility = 0.2
        self.assertIsNone(_classify_status_from_landmarks(landmarks))


if __name__ == "__main__":
    unittest.main()
