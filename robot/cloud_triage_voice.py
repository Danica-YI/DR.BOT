"""Cloud LLM triage interview with multilingual voice prompts and backend reporting."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any

from device.api_client import post_triage_assessment
from device.triage_flow import build_triage_assessment
from device.voice import OfflineVoice, record_wav

from .cloud_chat_client import CloudChatClient, CloudChatResponse
from .config import BACKEND_URL, CLOUD_TRIAGE_MAX_CONSECUTIVE_FAILURES, ROBOT_ID
from .summarization import SummaryClient


QUESTION_SET: list[tuple[str, str]] = [
    ("initial_response", "Please raise a hand or wave if you can hear me."),
    ("can_walk", "Can you walk?"),
    ("heavy_bleeding", "Are you bleeding heavily?"),
    ("breathing_difficulty", "Are you having trouble breathing?"),
    ("trapped", "Are you trapped?"),
]
DEFAULT_LANGUAGE = "en-US"


class CloudTriageUnavailable(RuntimeError):
    """Raised when the cloud triage path cannot complete and should fall back offline."""


@dataclass(frozen=True)
class InterpretedAnswer:
    question: str
    answer: str | None
    language: str
    transcript: str
    confidence: float
    needs_help: bool
    raw_response: str


class CloudTriageInterviewer:
    """Ask fixed triage questions, interpret spoken answers with a cloud model, and post the report."""

    def __init__(
        self,
        base_url: str = BACKEND_URL,
        device_id: str = ROBOT_ID,
        lat: float = -27.4698,
        lon: float = 153.0251,
        audio_device: int | None = None,
        record_seconds: float = 5.0,
        max_consecutive_failures: int = CLOUD_TRIAGE_MAX_CONSECUTIVE_FAILURES,
    ) -> None:
        self.base_url = base_url
        self.device_id = device_id
        self.lat = lat
        self.lon = lon
        self.audio_device = audio_device
        self.record_seconds = record_seconds
        self.max_consecutive_failures = max(1, max_consecutive_failures)
        self.voice = OfflineVoice(enabled=True, audio_device=audio_device)
        self.chat = CloudChatClient()
        self.summary_client = SummaryClient()
        self.logger = logging.getLogger(__name__)
        self._prompt_cache: dict[tuple[str, str], str] = {}
        self.active_language = DEFAULT_LANGUAGE
        self.locked_language: str | None = None
        self.consecutive_failures = 0

    def run_once(self, submit_report: bool = True) -> tuple[dict[str, Any], bool | None, Any]:
        answer_details: list[dict[str, Any]] = []
        answers: dict[str, bool | None] = {}
        resource_requested = False

        initial = self._ask_question(*QUESTION_SET[0])
        answer_details.append(self._to_answer_detail(initial))
        self._apply_language(initial.language)
        if initial.answer == "HELP":
            resource_requested = True

        if initial.answer is None:
            assessment = self._build_unresponsive_assessment(answer_details)
        else:
            for key, prompt in QUESTION_SET[1:]:
                interpreted = self._ask_question(key, prompt)
                answer_details.append(self._to_answer_detail(interpreted))
                self._apply_language(interpreted.language)
                if interpreted.answer == "HELP":
                    resource_requested = True
                answers[key] = interpreted.answer == "YES" if interpreted.answer in ("YES", "NO") else None

            assessment = self._build_responsive_assessment(answer_details, answers, resource_requested)

        assessment.update(self.summary_client.summarize_assessment(assessment))
        success = None
        response: Any = None
        if submit_report:
            success, response = post_triage_assessment(
                self.base_url,
                self.device_id,
                assessment,
                self.lat,
                self.lon,
            )
        return assessment, success, response

    def _ask_question(self, question_key: str, english_prompt: str, allow_language_retry: bool = True) -> InterpretedAnswer:
        prompt_language = _normalize_language(self.locked_language or self.active_language)
        spoken_prompt = self._translate_prompt(english_prompt, prompt_language)
        print(f"[cloud-triage] ask question={question_key} language={prompt_language} retry={not allow_language_retry}")
        self.voice.say(spoken_prompt, language=self.active_language)
        audio_bytes = record_wav(duration=self.record_seconds, audio_device=self.audio_device)
        interpreted = self._interpret_audio_with_retry(question_key, english_prompt, audio_bytes)
        self.logger.info(
            "Question %s -> answer=%s language=%s transcript=%s",
            question_key,
            interpreted.answer,
            interpreted.language,
            interpreted.transcript,
        )
        print(
            "[cloud-triage] heard "
            f"question={question_key} detected_language={interpreted.language} "
            f"answer={interpreted.answer or 'UNKNOWN'} transcript={interpreted.transcript or '<empty>'}"
        )
        target_language = self._select_target_language(interpreted)
        if allow_language_retry and target_language != prompt_language and _should_repeat_in_detected_language(prompt_language, interpreted):
            self.logger.info(
                "Repeating question %s in detected language %s after mismatch from %s.",
                question_key,
                target_language,
                prompt_language,
            )
            print(
                "[cloud-triage] repeat-current-question "
                f"question={question_key} from_language={prompt_language} to_language={target_language}"
            )
            self.active_language = target_language
            if self.locked_language is None and _is_meaningful_language_signal(interpreted):
                self.locked_language = target_language
            return self._ask_question(question_key, english_prompt, allow_language_retry=False)
        if self.locked_language is None and _is_meaningful_language_signal(interpreted):
            self.locked_language = target_language
            self.active_language = target_language
        return interpreted

    def _interpret_audio_with_retry(self, question_key: str, english_prompt: str, audio_bytes: bytes) -> InterpretedAnswer:
        last_error: str | None = None
        for attempt in range(self.max_consecutive_failures):
            response = self.chat.ask_audio(
                audio_bytes,
                instruction=_build_audio_interpretation_instruction(question_key, english_prompt),
                system_prompt=(
                    "You are an emergency triage interpreter. "
                    "Return only one compact JSON object and do not add markdown."
                ),
            )
            try:
                interpreted = _parse_interpreted_answer(question_key, response)
                self.consecutive_failures = 0
                return interpreted
            except CloudTriageUnavailable as exc:
                last_error = str(exc)
                self.consecutive_failures += 1
                self.logger.warning(
                    "Cloud triage attempt %s/%s failed for question %s: %s",
                    attempt + 1,
                    self.max_consecutive_failures,
                    question_key,
                    exc,
                )
                if self.consecutive_failures >= self.max_consecutive_failures:
                    raise
        raise CloudTriageUnavailable(last_error or "cloud triage request failed")

    def _translate_prompt(self, english_prompt: str, language: str) -> str:
        normalized_language = _normalize_language(language)
        if normalized_language.startswith("en"):
            return english_prompt

        cache_key = (english_prompt, normalized_language)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        translation = self.chat.ask_text(
            (
                "Translate the following rescue-robot question into natural spoken "
                f"{normalized_language}. Return only the translated question.\n\n"
                f"Question: {english_prompt}"
            ),
            system_prompt="You are a concise translator for emergency voice prompts.",
        )
        translated = translation.text.strip() if translation.success and translation.text else english_prompt
        self._prompt_cache[cache_key] = translated
        return translated

    def _apply_language(self, language: str) -> None:
        if self.locked_language:
            self.active_language = self.locked_language
            return
        normalized = _normalize_language(language)
        if normalized:
            self.active_language = normalized

    def _select_target_language(self, interpreted: InterpretedAnswer) -> str:
        if self.locked_language:
            return self.locked_language
        return _normalize_language(interpreted.language)

    def _build_unresponsive_assessment(self, answer_details: list[dict[str, Any]]) -> dict[str, Any]:
        assessment = build_triage_assessment(
            response_type="unresponsive",
            can_walk=True,
        )
        assessment["response_detected"] = False
        assessment["interaction_mode"] = "cloud_voice"
        assessment["answers"] = answer_details
        assessment["conversation_language"] = self.active_language
        assessment["no_response_questions"] = ["initial_response"]
        assessment["review_status"] = "PRIORITY_REVIEW"
        assessment["review_reason"] = "NO_RESPONSE"
        if "no response to initial_response" not in assessment["reasons"]:
            assessment["reasons"].append("no response to initial_response")
        return assessment

    def _build_responsive_assessment(
        self,
        answer_details: list[dict[str, Any]],
        answers: dict[str, bool | None],
        resource_requested: bool,
    ) -> dict[str, Any]:
        can_walk = answers.get("can_walk")
        assessment = build_triage_assessment(
            response_type="responding",
            can_walk=can_walk is not False,
            heavy_bleeding=answers.get("heavy_bleeding") is True,
            breathing_difficulty=answers.get("breathing_difficulty") is True,
            trapped=answers.get("trapped") is True,
            needs_supply=resource_requested,
        )
        no_response_questions = [
            item["question"]
            for item in answer_details
            if item["channel_status"] == "NO_RESPONSE"
        ]
        assessment["response_detected"] = True
        assessment["interaction_mode"] = "cloud_voice"
        assessment["conversation_language"] = self.active_language
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
        return assessment

    def _to_answer_detail(self, interpreted: InterpretedAnswer) -> dict[str, Any]:
        return {
            "question": interpreted.question,
            "answer": interpreted.answer,
            "source": "cloud_voice",
            "confidence": interpreted.confidence,
            "channel_status": "RECOGNIZED" if interpreted.answer else "NO_RESPONSE",
            "transcript": interpreted.transcript,
            "language": interpreted.language,
            "raw_response": interpreted.raw_response,
        }


def _build_audio_interpretation_instruction(question_key: str, english_prompt: str) -> str:
    return (
        "You are interpreting one victim answer in an emergency triage interview. "
        "The spoken question was: "
        f"'{english_prompt}'. "
        "Return JSON only with these keys: "
        "language, answer, transcript, confidence, needs_help. "
        "language must be a BCP-47 style tag like en-US or zh-CN. "
        "answer must be one of YES, NO, HELP, UNKNOWN. "
        "If the speech is missing or unclear, use UNKNOWN and keep transcript empty if needed. "
        "needs_help must be true only if the person is explicitly asking for help or resources. "
        f"Question key: {question_key}."
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in model response")
    return json.loads(stripped[start:end + 1])


def _parse_interpreted_answer(question_key: str, response: CloudChatResponse) -> InterpretedAnswer:
    if not response.success or not response.text:
        raise CloudTriageUnavailable(response.error or "cloud triage request failed")

    try:
        payload = _extract_json_object(response.text)
    except Exception as exc:
        raise CloudTriageUnavailable(f"invalid cloud triage response: {exc}") from exc
    answer = str(payload.get("answer") or "").strip().upper()
    if answer not in {"YES", "NO", "HELP"}:
        answer = None

    return InterpretedAnswer(
        question=question_key,
        answer=answer,
        language=_normalize_language(str(payload.get("language") or DEFAULT_LANGUAGE)),
        transcript=str(payload.get("transcript") or "").strip(),
        confidence=float(payload.get("confidence") or 0.0),
        needs_help=bool(payload.get("needs_help")),
        raw_response=response.text,
    )


def _normalize_language(language: str) -> str:
    normalized = language.strip().replace("_", "-")
    if not normalized:
        return DEFAULT_LANGUAGE
    if normalized.lower() == "zh":
        return "zh-CN"
    if "-" not in normalized:
        if normalized.lower() == "en":
            return "en-US"
        return normalized
    return normalized


def _should_repeat_in_detected_language(prompt_language: str, interpreted: InterpretedAnswer) -> bool:
    detected_language = _normalize_language(interpreted.language)
    asked_language = _normalize_language(prompt_language)
    if detected_language == asked_language:
        return False
    if not interpreted.transcript and interpreted.answer is None:
        return False
    return True


def _is_meaningful_language_signal(interpreted: InterpretedAnswer) -> bool:
    transcript = interpreted.transcript.strip()
    if not transcript:
        return False
    if len(transcript) >= 2:
        return True
    return interpreted.answer == "HELP"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cloud triage voice interview and post the report.")
    parser.add_argument("--base-url", default=BACKEND_URL, help="Backend API base URL")
    parser.add_argument("--device-id", default=ROBOT_ID, help="Device ID to report")
    parser.add_argument("--lat", type=float, default=-27.4698, help="Report latitude")
    parser.add_argument("--lon", type=float, default=153.0251, help="Report longitude")
    parser.add_argument("--audio-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to record each spoken answer")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()
    interviewer = CloudTriageInterviewer(
        base_url=args.base_url,
        device_id=args.device_id,
        lat=args.lat,
        lon=args.lon,
        audio_device=args.audio_device,
        record_seconds=args.duration,
    )
    assessment, success, response = interviewer.run_once()
    print(json.dumps({"success": success, "response": response, "assessment": assessment}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
