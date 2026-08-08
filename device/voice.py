"""Offline text-to-speech and constrained YES/NO speech recognition."""

from __future__ import annotations

import json
import os
import queue
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceAnswer:
    answer: str | None
    confidence: float
    status: str
    transcript: str = ""


class OfflineVoice:
    """Optional offline voice channel.

    Missing audio dependencies or a missing Vosk model are reported as
    UNAVAILABLE so the caller can immediately fall back to gestures.
    """

    def __init__(self, model_path: str | None = None, enabled: bool = True, audio_device: int | None = None):
        self.enabled = enabled
        self.audio_device = audio_device
        self.model_path = model_path or os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
        self._tts = None
        self._tts_module = None
        self._model = None
        self.error: str | None = None

        if not enabled:
            self.error = "voice disabled"
            return
        try:
            if sys.platform == "win32":
                import win32com.client

                self._tts_module = win32com.client
            else:
                import pyttsx3

                self._tts_module = pyttsx3
        except Exception as exc:
            self.error = f"TTS unavailable: {exc}"

        try:
            from vosk import Model

            if os.path.isdir(self.model_path):
                self._model = Model(self.model_path)
            else:
                self.error = f"Vosk model not found: {self.model_path}"
        except Exception as exc:
            self.error = f"speech recognition unavailable: {exc}"

    @property
    def recognition_available(self) -> bool:
        return self._model is not None

    def say(self, message: str) -> bool:
        print(f"VOICE: {message}")
        if self._tts_module is None:
            return False
        try:
            if self._tts is None:
                if sys.platform == "win32":
                    self._tts = self._tts_module.Dispatch("SAPI.SpVoice")
                else:
                    self._tts = self._tts_module.init()
            if sys.platform == "win32":
                self._tts.Speak(message)
            else:
                self._tts.say(message)
                self._tts.runAndWait()
            return True
        except Exception as exc:
            self.error = f"TTS failed: {exc}"
            return False

    def listen_yes_no(self, timeout: float = 4.0) -> VoiceAnswer:
        if self._model is None:
            return VoiceAnswer(None, 0.0, "UNAVAILABLE")

        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer

            device_info = sd.query_devices(self.audio_device, "input")
            sample_rate = int(device_info.get("default_samplerate") or 16000)
            recognizer = KaldiRecognizer(
                self._model,
                sample_rate,
                json.dumps(["yes", "yeah", "yep", "no", "nope", "help", "[unk]"]),
            )
            chunks: queue.Queue[bytes] = queue.Queue()

            def callback(indata, frames, callback_time, status):
                del frames, callback_time, status
                chunks.put(bytes(indata))

            deadline = time.monotonic() + timeout
            transcript = ""
            with sd.RawInputStream(
                device=self.audio_device,
                samplerate=sample_rate,
                blocksize=4000,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                while time.monotonic() < deadline:
                    try:
                        data = chunks.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(data):
                        text = json.loads(recognizer.Result()).get("text", "").lower()
                        transcript = f"{transcript} {text}".strip()
                        answer = _answer_from_text(transcript)
                        if answer:
                            return VoiceAnswer(answer, 0.9, "RECOGNIZED", transcript)

            final_text = json.loads(recognizer.FinalResult()).get("text", "").lower()
            transcript = f"{transcript} {final_text}".strip()
            answer = _answer_from_text(transcript)
            if answer:
                return VoiceAnswer(answer, 0.8, "RECOGNIZED", transcript)
            return VoiceAnswer(None, 0.0, "TIMEOUT" if not transcript else "UNRECOGNIZED", transcript)
        except Exception as exc:
            self.error = f"speech recognition failed: {exc}"
            return VoiceAnswer(None, 0.0, "UNAVAILABLE")


def _answer_from_text(text: str) -> str | None:
    words = set(text.lower().split())
    if words & {"yes", "yeah", "yep"}:
        return "YES"
    if words & {"no", "nope"}:
        return "NO"
    if "help" in words:
        return "HELP"
    return None
