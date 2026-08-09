"""Microphone -> cloud LLM -> speaker voice chat loop."""

from __future__ import annotations

import argparse
import logging

from device.voice import OfflineVoice, record_wav

from .cloud_chat_client import CloudChatClient


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise voice assistant for a rescue robot. "
    "Answer clearly and briefly in spoken English."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a microphone-to-cloud voice chat loop.")
    parser.add_argument("--audio-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--duration", type=float, default=6.0, help="Recording duration in seconds")
    parser.add_argument(
        "--instruction",
        default="Transcribe the user's audio question and answer it directly.",
        help="Instruction sent alongside the recorded audio.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the cloud model.",
    )
    parser.add_argument("--once", action="store_true", help="Run a single question/answer turn and exit")
    return parser


def run_voice_chat(audio_device: int | None, duration: float, instruction: str, system_prompt: str, once: bool) -> None:
    logger = logging.getLogger(__name__)
    speaker = OfflineVoice(enabled=True, audio_device=audio_device)
    client = CloudChatClient()

    while True:
        prompt = "Please ask your question after the beep."
        print(prompt)
        speaker.say(prompt)
        audio_bytes = record_wav(duration=duration, audio_device=audio_device)
        logger.info("Recorded %.2f seconds of microphone audio.", duration)

        response = client.ask_audio(
            audio_bytes,
            instruction=instruction,
            system_prompt=system_prompt,
        )
        if not response.success or not response.text:
            error_text = f"Cloud chat failed: {response.error}"
            print(error_text)
            speaker.say("Cloud chat failed. Please check the network or API key.")
        else:
            print(f"AI: {response.text}")
            speaker.say(response.text)

        if once:
            return


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()
    run_voice_chat(
        audio_device=args.audio_device,
        duration=args.duration,
        instruction=args.instruction,
        system_prompt=args.system_prompt,
        once=args.once,
    )


if __name__ == "__main__":
    main()
