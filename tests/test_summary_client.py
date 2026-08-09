import unittest
from unittest.mock import patch

from robot.summarization.summary_client import (
    SummaryClient,
    SummaryResult,
    build_assessment_summary_input,
    trim_summary,
)
from robot.triage.interaction import TriageInteraction
from robot.cloud_triage_voice import CloudTriageUnavailable


class FakeSummarizer:
    def __init__(self, result: SummaryResult):
        self.result = result

    def summarize(self, assessment):
        return self.result


class SummaryClientTests(unittest.TestCase):
    def test_trim_summary_enforces_word_limit(self):
        text = "one two three four five"
        self.assertEqual(trim_summary(text, max_words=3), "one two three")

    def test_summary_input_contains_answers_and_reasons(self):
        payload = build_assessment_summary_input(
            {
                "priority": "medical",
                "reasons": ["heavy bleeding"],
                "answers": [
                    {
                        "question": "heavy_bleeding",
                        "answer": "YES",
                        "source": "voice",
                        "channel_status": "RECOGNIZED",
                        "confidence": 0.9,
                    }
                ],
            }
        )
        self.assertIn("heavy bleeding", payload)
        self.assertIn("heavy_bleeding", payload)

    def test_client_prefers_cloud_when_available(self):
        client = SummaryClient(
            cloud=FakeSummarizer(SummaryResult("cloud summary", "cloud", True)),
            edge=FakeSummarizer(SummaryResult("edge summary", "edge", True)),
        )
        result = client.summarize_assessment({"priority": "ok"})
        self.assertEqual(result["summary"], "cloud summary")
        self.assertEqual(result["summary_mode"], "online")

    def test_client_falls_back_to_edge(self):
        client = SummaryClient(
            cloud=FakeSummarizer(SummaryResult(None, "cloud", False, "timeout")),
            edge=FakeSummarizer(SummaryResult("edge summary", "edge", True)),
        )
        result = client.summarize_assessment({"priority": "ok"})
        self.assertEqual(result["summary"], "edge summary")
        self.assertEqual(result["summary_mode"], "offline")
        self.assertEqual(result["summary_fallback_reason"], "timeout")

    @patch("robot.triage.interaction.CLOUD_TRIAGE_ENABLED", True)
    @patch("robot.triage.interaction.CLOUD_SUMMARY_API_KEY", "test-key")
    @patch("robot.triage.interaction.CloudTriageInterviewer")
    def test_triage_interaction_uses_cloud_mode_when_enabled(self, interviewer_cls):
        interviewer = interviewer_cls.return_value
        interviewer.run_once.return_value = ({"priority": "medical", "interaction_mode": "cloud_voice"}, None, None)

        triage = TriageInteraction(use_mock=False)
        result = triage.run_triage(frame_provider=None)

        self.assertEqual(result["interaction_mode"], "cloud_voice")
        interviewer.run_once.assert_called_once_with(submit_report=False)

    @patch("robot.triage.interaction.CLOUD_TRIAGE_ENABLED", True)
    @patch("robot.triage.interaction.CLOUD_SUMMARY_API_KEY", "test-key")
    @patch("robot.triage.interaction.CloudTriageInterviewer")
    @patch("robot.triage.interaction.SummaryClient")
    @patch("robot.triage.interaction.GestureStabilizer")
    @patch("robot.triage.interaction.PoseClassifier")
    @patch("robot.triage.interaction.OfflineVoice")
    def test_triage_interaction_falls_back_to_offline_when_cloud_unavailable(
        self,
        offline_voice_cls,
        pose_cls,
        _gesture_cls,
        summary_client_cls,
        interviewer_cls,
    ):
        interviewer_cls.return_value.run_once.side_effect = CloudTriageUnavailable("timeout")
        voice = offline_voice_cls.return_value
        voice.listen_yes_no.side_effect = [
            type("VoiceResult", (), {"answer": "YES", "confidence": 0.9, "status": "RECOGNIZED"})(),
            type("VoiceResult", (), {"answer": "YES", "confidence": 0.9, "status": "RECOGNIZED"})(),
            type("VoiceResult", (), {"answer": "NO", "confidence": 0.9, "status": "RECOGNIZED"})(),
            type("VoiceResult", (), {"answer": "NO", "confidence": 0.9, "status": "RECOGNIZED"})(),
            type("VoiceResult", (), {"answer": "NO", "confidence": 0.9, "status": "RECOGNIZED"})(),
        ]
        summary_client_cls.return_value.summarize_assessment.return_value = {
            "summary": "offline summary",
            "summary_provider": "cloud",
            "summary_mode": "online",
        }

        triage = TriageInteraction(use_mock=False)
        result = triage.run_triage(frame_provider=lambda: object())

        self.assertEqual(result["interaction_mode"], "speech")
        self.assertEqual(result["summary"], "offline summary")
        interviewer_cls.return_value.run_once.assert_called_once_with(submit_report=False)


if __name__ == "__main__":
    unittest.main()
