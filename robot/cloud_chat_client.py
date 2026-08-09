"""Online chat client for cloud LLM text and audio questions."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import requests

from .config import (
    CLOUD_CHAT_MAX_RETRIES,
    CLOUD_SUMMARY_API_KEY,
    CLOUD_SUMMARY_BASE_URL,
    CLOUD_SUMMARY_ENABLED,
    CLOUD_SUMMARY_MODEL,
    SUMMARY_REQUEST_TIMEOUT_SECONDS,
)


@dataclass(frozen=True)
class CloudChatResponse:
    text: str | None
    success: bool
    error: str | None = None
    raw: dict[str, Any] | None = None


class CloudChatClient:
    """Send free-form text or audio questions to the configured cloud model."""

    def __init__(
        self,
        enabled: bool = CLOUD_SUMMARY_ENABLED,
        base_url: str = CLOUD_SUMMARY_BASE_URL,
        api_key: str | None = CLOUD_SUMMARY_API_KEY,
        model: str = CLOUD_SUMMARY_MODEL,
        timeout_seconds: int = SUMMARY_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = CLOUD_CHAT_MAX_RETRIES,
        session: requests.Session | None = None,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, max_retries)
        self.session = session or requests.Session()

    def ask_text(self, prompt: str, system_prompt: str | None = None) -> CloudChatResponse:
        if not prompt.strip():
            return CloudChatResponse(None, False, "prompt is empty")
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._create_completion(messages)

    def ask_audio(
        self,
        wav_bytes: bytes,
        instruction: str = "Listen to the user's audio question and answer it directly.",
        system_prompt: str | None = None,
    ) -> CloudChatResponse:
        if not wav_bytes:
            return CloudChatResponse(None, False, "audio payload is empty")
        messages = self._build_audio_messages(wav_bytes, instruction, system_prompt)
        return self._create_completion(messages)

    def _build_audio_messages(
        self,
        wav_bytes: bytes,
        instruction: str,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                ],
            }
        )
        return messages

    def _create_completion(self, messages: list[dict[str, Any]]) -> CloudChatResponse:
        if not self.enabled:
            return CloudChatResponse(None, False, "cloud chat disabled")
        if not self.api_key:
            return CloudChatResponse(None, False, "cloud API key not configured")
        if not self.model:
            return CloudChatResponse(None, False, "cloud model not configured")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: str | None = None
        for _ in range(self.max_retries):
            try:
                response = self.session.post(
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
                return CloudChatResponse(text, True, raw=body)
            except Exception as exc:
                last_error = str(exc)
        return CloudChatResponse(None, False, last_error or "cloud request failed")
