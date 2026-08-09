import unittest
from unittest.mock import patch

from robot.cloud_chat_client import CloudChatResponse
from robot.cloud_triage_voice import (
    CloudTriageUnavailable,
    CloudTriageInterviewer,
    _extract_json_object,
    _is_meaningful_language_signal,
    _normalize_language,
    _parse_interpreted_answer,
    _should_repeat_in_detected_language,
)


class CloudTriageVoiceTests(unittest.TestCase):
    def test_extract_json_object_from_fenced_block(self):
        payload = _extract_json_object(
            "```json\n{\"language\":\"en-US\",\"answer\":\"YES\"}\n```"
        )
        self.assertEqual(payload["language"], "en-US")
        self.assertEqual(payload["answer"], "YES")

    def test_parse_interpreted_answer_normalizes_unknown_to_none(self):
        interpreted = _parse_interpreted_answer(
            "can_walk",
            CloudChatResponse(
                text='{"language":"zh","answer":"UNKNOWN","transcript":"听不清","confidence":0.1,"needs_help":false}',
                success=True,
            ),
        )
        self.assertIsNone(interpreted.answer)
        self.assertEqual(interpreted.language, "zh-CN")
        self.assertEqual(interpreted.transcript, "听不清")

    def test_normalize_language_defaults_to_english(self):
        self.assertEqual(_normalize_language(""), "en-US")
        self.assertEqual(_normalize_language("en"), "en-US")
        self.assertEqual(_normalize_language("zh"), "zh-CN")

    def test_build_responsive_assessment_marks_review_on_missing_answer(self):
        interviewer = CloudTriageInterviewer()
        interviewer.active_language = "es-ES"
        assessment = interviewer._build_responsive_assessment(
            answer_details=[
                {"question": "initial_response", "channel_status": "RECOGNIZED"},
                {"question": "can_walk", "channel_status": "NO_RESPONSE"},
            ],
            answers={"can_walk": None, "heavy_bleeding": False, "breathing_difficulty": False, "trapped": False},
            resource_requested=False,
        )
        self.assertEqual(assessment["conversation_language"], "es-ES")
        self.assertEqual(assessment["review_status"], "PRIORITY_REVIEW")
        self.assertIn("no response to can_walk", assessment["reasons"])

    def test_should_repeat_when_detected_language_differs(self):
        interpreted = _parse_interpreted_answer(
            "initial_response",
            CloudChatResponse(
                text='{"language":"zh-CN","answer":"YES","transcript":"我能听见","confidence":0.9,"needs_help":false}',
                success=True,
            ),
        )
        self.assertTrue(_should_repeat_in_detected_language("en-US", interpreted))
        self.assertFalse(_should_repeat_in_detected_language("zh-CN", interpreted))

    def test_meaningful_language_signal_locks_on_non_empty_transcript(self):
        interpreted = _parse_interpreted_answer(
            "initial_response",
            CloudChatResponse(
                text='{"language":"zh-CN","answer":"YES","transcript":"我能听见","confidence":0.9,"needs_help":false}',
                success=True,
            ),
        )
        self.assertTrue(_is_meaningful_language_signal(interpreted))

    @patch("robot.cloud_triage_voice.record_wav", return_value=b"wav")
    def test_question_repeats_once_in_detected_language(self, _record_wav):
        interviewer = CloudTriageInterviewer()
        interviewer.voice = type("VoiceStub", (), {"say": lambda self, message, language=None: None})()
        prompts = []
        interviewer._translate_prompt = lambda english_prompt, language: prompts.append(language) or f"{language}:{english_prompt}"
        responses = iter(
            [
                CloudChatResponse(
                    text='{"language":"zh-CN","answer":"YES","transcript":"我能听见","confidence":0.9,"needs_help":false}',
                    success=True,
                ),
                CloudChatResponse(
                    text='{"language":"zh-CN","answer":"YES","transcript":"可以","confidence":0.95,"needs_help":false}',
                    success=True,
                ),
            ]
        )
        interviewer.chat = type("ChatStub", (), {"ask_audio": lambda self, *args, **kwargs: next(responses)})()

        interpreted = interviewer._ask_question("initial_response", "Please raise a hand or wave if you can hear me.")

        self.assertEqual(prompts, ["en-US", "zh-CN"])
        self.assertEqual(interpreted.language, "zh-CN")
        self.assertEqual(interviewer.active_language, "zh-CN")
        self.assertEqual(interviewer.locked_language, "zh-CN")

    @patch("robot.cloud_triage_voice.record_wav", return_value=b"wav")
    def test_locked_language_prevents_later_short_answer_from_switching_back(self, _record_wav):
        interviewer = CloudTriageInterviewer()
        interviewer.locked_language = "zh-CN"
        interviewer.active_language = "zh-CN"
        interviewer.voice = type("VoiceStub", (), {"say": lambda self, message, language=None: None})()
        prompts = []
        interviewer._translate_prompt = lambda english_prompt, language: prompts.append(language) or f"{language}:{english_prompt}"
        interviewer.chat = type(
            "ChatStub",
            (),
            {
                "ask_audio": lambda self, *args, **kwargs: CloudChatResponse(
                    text='{"language":"en-US","answer":"YES","transcript":"yes","confidence":0.95,"needs_help":false}',
                    success=True,
                )
            },
        )()

        interpreted = interviewer._ask_question("can_walk", "Can you walk?")

        self.assertEqual(prompts, ["zh-CN"])
        self.assertEqual(interpreted.language, "en-US")
        self.assertEqual(interviewer.active_language, "zh-CN")
        self.assertEqual(interviewer.locked_language, "zh-CN")

    def test_interpret_audio_retries_current_question_before_failing(self):
        interviewer = CloudTriageInterviewer(max_consecutive_failures=2)
        responses = iter(
            [
                CloudChatResponse(text=None, success=False, error="timeout"),
                CloudChatResponse(
                    text='{"language":"en-US","answer":"YES","transcript":"yes","confidence":0.95,"needs_help":false}',
                    success=True,
                ),
            ]
        )
        interviewer.chat = type("ChatStub", (), {"ask_audio": lambda self, *args, **kwargs: next(responses)})()

        interpreted = interviewer._interpret_audio_with_retry("can_walk", "Can you walk?", b"wav")

        self.assertEqual(interpreted.answer, "YES")
        self.assertEqual(interviewer.consecutive_failures, 0)

    def test_interpret_audio_raises_after_consecutive_failures_threshold(self):
        interviewer = CloudTriageInterviewer(max_consecutive_failures=2)
        responses = iter(
            [
                CloudChatResponse(text=None, success=False, error="timeout"),
                CloudChatResponse(text=None, success=False, error="timeout"),
            ]
        )
        interviewer.chat = type("ChatStub", (), {"ask_audio": lambda self, *args, **kwargs: next(responses)})()

        with self.assertRaises(CloudTriageUnavailable):
            interviewer._interpret_audio_with_retry("can_walk", "Can you walk?", b"wav")


if __name__ == "__main__":
    unittest.main()
