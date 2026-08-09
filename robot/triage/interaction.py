"""Triage interaction helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from device.response import GestureStabilizer
from device.triage_flow import build_triage_assessment
from device.voice import OfflineVoice
from device.pose_classifier import PoseClassifier

from ..cloud_triage_voice import CloudTriageInterviewer, CloudTriageUnavailable
from ..config import (
    CLOUD_SUMMARY_API_KEY,
    CLOUD_TRIAGE_ENABLED,
    CLOUD_TRIAGE_RECORD_SECONDS,
    GESTURE_TIMEOUT_SECONDS,
    MOCK_TRIAGE,
    VOICE_TIMEOUT_SECONDS,
)
from ..summarization import SummaryClient


class TriageInteraction:
    """Mock triage interaction interface for victim assessment."""

    def __init__(self, use_mock: bool | None = None, vosk_model: str | None = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.use_mock = MOCK_TRIAGE if use_mock is None else use_mock
        self.use_cloud_triage = CLOUD_TRIAGE_ENABLED and not self.use_mock and bool(CLOUD_SUMMARY_API_KEY)
        self.questions = [
            ("can_walk", "Can you walk?"),
            ("heavy_bleeding", "Are you bleeding heavily?"),
            ("breathing_difficulty", "Are you having trouble breathing?"),
            ("trapped", "Are you trapped?"),
        ]
        self.voice_timeout = VOICE_TIMEOUT_SECONDS
        self.gesture_timeout = GESTURE_TIMEOUT_SECONDS

        if self.use_mock:
            self.logger.info("Mock triage interaction active.")
        elif self.use_cloud_triage:
            self.cloud_interviewer = CloudTriageInterviewer(record_seconds=CLOUD_TRIAGE_RECORD_SECONDS)
            self.logger.info("Real triage interaction initialized with cloud multilingual voice mode.")
        else:
            self._init_offline_components(vosk_model)
            self.logger.info("Real triage interaction initialized with offline voice and pose recognition.")

    def run_triage(self, frame_provider: Callable[[], Any] | None = None, context: Any | None = None) -> dict:
        """Run a triage interaction and return a dashboard assessment.

        Args:
            frame_provider: Function returning the latest camera frame for gesture recognition.
            context: Optional contextual data from vision or previous state.

        Returns:
            Structured triage assessment suitable for dashboard reporting.
        """
        del context
        if self.use_mock:
            self.logger.debug("Running mock triage interaction.")
            return self._mock_response()
        if self.use_cloud_triage:
            try:
                assessment, _, _ = self.cloud_interviewer.run_once(submit_report=False)
                return assessment
            except CloudTriageUnavailable as exc:
                self.logger.warning("Cloud triage unavailable, falling back to offline mode: %s", exc)
                self._init_offline_components()

        if frame_provider is None:
            raise ValueError("frame_provider is required for offline triage interaction")

        answer_details: list[dict[str, Any]] = []
        answers: dict[str, bool | None] = {}
        resource_requested = False
        current_question = ("initial_response", "Please raise a hand or wave if you can hear me.")

        initial_answer = self._ask_question(current_question, frame_provider, interaction_mode=None)
        answer_details.append(initial_answer)

        if initial_answer["answer"] is None:
            assessment = build_triage_assessment(
                initial_status="ok",
                response_type="unresponsive",
                can_walk=True,
            )
            assessment["response_detected"] = False
            assessment["interaction_mode"] = None
            assessment["answers"] = answer_details
            assessment["no_response_questions"] = ["initial_response"]
            assessment["review_status"] = "PRIORITY_REVIEW"
            assessment["review_reason"] = "NO_RESPONSE"
            if "no response to initial_response" not in assessment["reasons"]:
                assessment["reasons"].append("no response to initial_response")
            assessment.update(self.summary_client.summarize_assessment(assessment))
            return assessment

        interaction_mode = "speech" if initial_answer["source"] == "voice" else "gesture"
        if initial_answer["answer"] == "HELP":
            resource_requested = True

        for question in self.questions:
            result = self._ask_question(question, frame_provider, interaction_mode=interaction_mode)
            answer_details.append(result)
            if result["answer"] == "HELP":
                resource_requested = True
            if question[0] != "initial_response":
                answers[question[0]] = result["answer"] == "YES" if result["answer"] in ("YES", "NO") else None

        can_walk = answers.get("can_walk")
        no_response_questions = [
            item["question"]
            for item in answer_details
            if item.get("channel_status") == "NO_RESPONSE"
        ]
        assessment = build_triage_assessment(
            initial_status="resource" if resource_requested else "ok",
            response_type="responding",
            can_walk=can_walk is not False,
            heavy_bleeding=answers.get("heavy_bleeding") is True,
            breathing_difficulty=answers.get("breathing_difficulty") is True,
            trapped=answers.get("trapped") is True,
        )
        assessment["response_detected"] = True
        assessment["interaction_mode"] = interaction_mode
        assessment["answers"] = answer_details
        assessment["can_walk"] = can_walk
        assessment["no_response_questions"] = no_response_questions
        assessment["review_status"] = "PRIORITY_REVIEW" if no_response_questions else "COMPLETE"
        if no_response_questions:
            assessment["review_reason"] = "NO_RESPONSE"
            for question in no_response_questions:
                reason = f"no response to {question}"
                if reason not in assessment["reasons"]:
                    assessment["reasons"].append(reason)
        assessment.update(self.summary_client.summarize_assessment(assessment))
        return assessment

    def _ask_question(
        self,
        question: tuple[str, str],
        frame_provider: Callable[[], Any],
        interaction_mode: str | None,
    ) -> dict[str, Any]:
        key, prompt = question
        voice_result = None
        if interaction_mode != "gesture":
            self.voice.say(prompt)
            voice_result = self.voice.listen_yes_no(self.voice_timeout)
            if voice_result.answer:
                return {
                    "question": key,
                    "answer": voice_result.answer,
                    "source": "voice",
                    "confidence": voice_result.confidence,
                    "channel_status": voice_result.status,
                }
            if key != "initial_response" and interaction_mode == "speech":
                return {
                    "question": key,
                    "answer": None,
                    "source": "none",
                    "confidence": 0.0,
                    "channel_status": "NO_RESPONSE",
                }
            if key == "initial_response":
                self.voice.say("If you can hear me, please wave your hand.")

        self.gesture_stabilizer.reset()
        deadline = time.monotonic() + self.gesture_timeout
        while time.monotonic() < deadline:
            frame = frame_provider()
            status = self.classifier.classify(frame)
            stable = self.gesture_stabilizer.add(status)
            if stable:
                return {
                    "question": key,
                    "answer": stable.answer,
                    "source": stable.source,
                    "confidence": stable.confidence,
                    "channel_status": "STABLE",
                }

        return {
            "question": key,
            "answer": None,
            "source": "none",
            "confidence": 0.0,
            "channel_status": "NO_RESPONSE" if interaction_mode == "speech" or key != "initial_response" else "TIMEOUT",
        }

    def _mock_response(self) -> dict:
        assessment = build_triage_assessment(
            initial_status="ok",
            response_type="responding",
            can_walk=False,
            heavy_bleeding=True,
        )
        assessment["response_detected"] = True
        assessment["interaction_mode"] = "mock"
        assessment["answers"] = []
        assessment["no_response_questions"] = []
        assessment["review_status"] = "COMPLETE"
        assessment["summary"] = "Mock triage indicates a responsive victim with heavy bleeding and limited mobility. Priority medical."
        assessment["summary_provider"] = "mock"
        assessment["summary_mode"] = "offline"
        return assessment

    def _init_offline_components(self, vosk_model: str | None = None) -> None:
        if hasattr(self, "voice") and hasattr(self, "classifier") and hasattr(self, "gesture_stabilizer") and hasattr(self, "summary_client"):
            return
        self.voice = OfflineVoice(model_path=vosk_model, enabled=True)
        self.classifier = PoseClassifier()
        self.gesture_stabilizer = GestureStabilizer()
        self.summary_client = SummaryClient()

    def close(self) -> None:
        if not self.use_mock and hasattr(self, "classifier"):
            self.classifier.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    triage = TriageInteraction()
    result = triage.run_triage()
    print(result)
