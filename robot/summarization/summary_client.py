"""Cloud and edge AI summarization for triage assessments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import (
    CLOUD_SUMMARY_API_KEY,
    CLOUD_SUMMARY_BASE_URL,
    CLOUD_SUMMARY_ENABLED,
    CLOUD_SUMMARY_MODEL,
    EDGE_SUMMARY_ENABLED,
    EDGE_SUMMARY_MODEL_PATH,
    EDGE_SUMMARY_MODEL_TYPE,
    SUMMARY_MAX_WORDS,
    SUMMARY_REQUEST_TIMEOUT_SECONDS,
)


def trim_summary(text: str, max_words: int = SUMMARY_MAX_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_assessment_summary_input(assessment: dict[str, Any]) -> str:
    """Flatten the triage result into concise structured input for AI summarization."""
    answers = assessment.get("answers") or []
    answer_lines = []
    for item in answers:
        answer_lines.append(
            f"{item.get('question')}: answer={item.get('answer')}, "
            f"source={item.get('source')}, status={item.get('channel_status')}, "
            f"confidence={item.get('confidence')}"
        )

    payload = {
        "priority": assessment.get("priority"),
        "response_type": assessment.get("response_type"),
        "interaction_mode": assessment.get("interaction_mode"),
        "response_detected": assessment.get("response_detected"),
        "can_walk": assessment.get("can_walk"),
        "heavy_bleeding": assessment.get("heavy_bleeding"),
        "breathing_difficulty": assessment.get("breathing_difficulty"),
        "trapped": assessment.get("trapped"),
        "review_status": assessment.get("review_status"),
        "review_reason": assessment.get("review_reason"),
        "reasons": assessment.get("reasons") or [],
        "no_response_questions": assessment.get("no_response_questions") or [],
        "answers": answer_lines,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_summary_prompt(assessment: dict[str, Any], max_words: int = SUMMARY_MAX_WORDS) -> str:
    """Construct a prompt for either cloud or edge LLM summarization."""
    return (
        "Summarize the emergency triage result in plain English. "
        f"Use no more than {max_words} words. "
        "Mention the victim response mode, important symptoms or risks, mobility, "
        "resource requests, and why the current priority was assigned. "
        "Do not invent facts. "
        f"Structured triage data: {build_assessment_summary_input(assessment)}"
    )


@dataclass(frozen=True)
class SummaryResult:
    text: str | None
    provider: str
    success: bool
    error: str | None = None


class Summarizer(Protocol):
    def summarize(self, assessment: dict[str, Any]) -> SummaryResult:
        ...


class CloudSummarizer:
    """Call a cloud chat-completions endpoint when network and credentials exist."""

    def __init__(
        self,
        enabled: bool = CLOUD_SUMMARY_ENABLED,
        base_url: str = CLOUD_SUMMARY_BASE_URL,
        api_key: str = CLOUD_SUMMARY_API_KEY,
        model: str = CLOUD_SUMMARY_MODEL,
        max_words: int = SUMMARY_MAX_WORDS,
        timeout_seconds: int = SUMMARY_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_words = max_words
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)

    def summarize(self, assessment: dict[str, Any]) -> SummaryResult:
        if not self.enabled:
            return SummaryResult(None, "cloud", False, "cloud summary disabled")
        if not self.api_key:
            return SummaryResult(None, "cloud", False, "cloud API key not configured")
        if not self.model:
            return SummaryResult(None, "cloud", False, "cloud model not configured")

        try:
            import requests
        except Exception as exc:
            return SummaryResult(None, "cloud", False, f"requests unavailable: {exc}")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an emergency-response summarizer. "
                        f"Return one factual paragraph of at most {self.max_words} words."
                    ),
                },
                {
                    "role": "user",
                    "content": build_summary_prompt(assessment, self.max_words),
                },
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            last_error = None
            for _ in range(2):
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if not response.ok:
                    last_error = f"HTTP {response.status_code}: {response.text.strip()}"
                    continue
                body = response.json()
                text = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if not text:
                    last_error = "cloud response missing content"
                    continue
                return SummaryResult(trim_summary(text, self.max_words), "cloud", True)
            return SummaryResult(None, "cloud", False, last_error or "cloud request failed")
        except Exception as exc:
            self.logger.warning("Cloud summary failed: %s", exc)
            return SummaryResult(None, "cloud", False, str(exc))


class EdgeSummarizer:
    """Run a local text generation model with transformers in local-files-only mode."""

    def __init__(
        self,
        enabled: bool = EDGE_SUMMARY_ENABLED,
        model_path: str = EDGE_SUMMARY_MODEL_PATH,
        model_type: str = EDGE_SUMMARY_MODEL_TYPE,
        max_words: int = SUMMARY_MAX_WORDS,
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.model_type = model_type
        self.max_words = max_words
        self.logger = logging.getLogger(__name__)
        self._pipeline = None

    def summarize(self, assessment: dict[str, Any]) -> SummaryResult:
        if not self.enabled:
            return SummaryResult(None, "edge", False, "edge summary disabled")
        if not self.model_path:
            return SummaryResult(None, "edge", False, "edge model path not configured")

        try:
            generator = self._get_pipeline()
        except Exception as exc:
            return SummaryResult(None, "edge", False, str(exc))

        prompt = build_summary_prompt(assessment, self.max_words)
        try:
            outputs = generator(
                prompt,
                max_new_tokens=96,
                do_sample=False,
                truncation=True,
            )
            text = self._extract_text(outputs, prompt)
            if not text:
                return SummaryResult(None, "edge", False, "edge model returned empty content")
            return SummaryResult(trim_summary(text, self.max_words), "edge", True)
        except Exception as exc:
            self.logger.warning("Edge summary failed: %s", exc)
            return SummaryResult(None, "edge", False, str(exc))

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(f"transformers unavailable: {exc}") from exc

        self._pipeline = pipeline(
            self.model_type,
            model=self.model_path,
            tokenizer=self.model_path,
            local_files_only=True,
        )
        return self._pipeline

    def _extract_text(self, outputs: Any, prompt: str) -> str:
        if not outputs:
            return ""
        first = outputs[0]
        if isinstance(first, dict):
            if "summary_text" in first:
                return str(first["summary_text"]).strip()
            generated = str(first.get("generated_text", "")).strip()
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated
        return str(first).strip()


class SummaryClient:
    """Try cloud AI first, then edge AI when cloud is unavailable or offline."""

    def __init__(
        self,
        cloud: Summarizer | None = None,
        edge: Summarizer | None = None,
        max_words: int = SUMMARY_MAX_WORDS,
    ) -> None:
        self.cloud = cloud or CloudSummarizer(max_words=max_words)
        self.edge = edge or EdgeSummarizer(max_words=max_words)
        self.max_words = max_words
        self.logger = logging.getLogger(__name__)

    def summarize_assessment(self, assessment: dict[str, Any]) -> dict[str, Any]:
        cloud_result = self.cloud.summarize(assessment)
        if cloud_result.success and cloud_result.text:
            return {
                "summary": cloud_result.text,
                "summary_provider": cloud_result.provider,
                "summary_mode": "online",
            }

        edge_result = self.edge.summarize(assessment)
        if edge_result.success and edge_result.text:
            return {
                "summary": edge_result.text,
                "summary_provider": edge_result.provider,
                "summary_mode": "offline",
                "summary_fallback_reason": cloud_result.error,
            }

        self.logger.warning(
            "AI summarization unavailable. cloud=%s edge=%s",
            cloud_result.error,
            edge_result.error,
        )
        return {
            "summary": None,
            "summary_provider": None,
            "summary_mode": None,
            "summary_error": {
                "cloud": cloud_result.error,
                "edge": edge_result.error,
            },
        }
