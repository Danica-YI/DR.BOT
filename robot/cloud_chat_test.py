"""Minimal online chat-completions smoke test."""

from __future__ import annotations

import argparse
import json
import sys

import requests

from robot.config import (
    CLOUD_SUMMARY_API_KEY,
    CLOUD_SUMMARY_BASE_URL,
    CLOUD_SUMMARY_MODEL,
)


def run_chat_test(prompt: str, timeout: int = 20) -> dict[str, object]:
    if not CLOUD_SUMMARY_API_KEY:
        raise RuntimeError("CLOUD_SUMMARY_API_KEY or OPENAI_API_KEY is not configured")

    url = f"{CLOUD_SUMMARY_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": CLOUD_SUMMARY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise assistant. Answer the user directly in one short paragraph.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {CLOUD_SUMMARY_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if not response.ok:
        detail = response.text.strip()
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    body = response.json()
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    return {
        "url": url,
        "model": CLOUD_SUMMARY_MODEL,
        "content": content,
        "id": body.get("id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an online cloud LLM chat smoke test.")
    parser.add_argument(
        "--prompt",
        default="Reply with the exact text: CLOUD_LLM_OK",
        help="Prompt to send to the cloud model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds.",
    )
    args = parser.parse_args()

    result = run_chat_test(args.prompt, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
