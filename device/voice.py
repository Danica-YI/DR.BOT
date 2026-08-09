"""Offline text-to-speech and constrained YES/NO speech recognition."""

from __future__ import annotations

import json
import os
import queue
import sys
import time
import wave
from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceAnswer:
    answer: str | None
    confidence: float
    status: str
    transcript: str = ""
    language: str | None = None


class OfflineVoice:
    """Optional offline voice channel.

    Missing audio dependencies or a missing Vosk model are reported as
    UNAVAILABLE so the caller can immediately fall back to gestures.
    """

    def __init__(
        self,
        model_path: str | None = None,
        enabled: bool = True,
        audio_device: int | None = None,
        whisper_model: str | None = None,
    ):
        self.enabled = enabled
        self.audio_device = audio_device
        self.whisper_model_path = whisper_model
        self.model_path = model_path or os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
        self._tts = None
        self._tts_module = None
        self._model = None
        self._whisper = None
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

        if self.whisper_model_path:
            try:
                from faster_whisper import WhisperModel

                print("Loading local Whisper model...")
                self._whisper = WhisperModel(
                    self.whisper_model_path,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
                print("Local Whisper model ready.")
            except Exception as exc:
                self.error = f"Whisper unavailable: {exc}"

    @property
    def recognition_available(self) -> bool:
        return self._model is not None

    def say(self, message: str, language: str | None = None) -> bool:
        del language
        print(f"VOICE: {message}")
        if self._tts_module is None:
            return False
        com_initialized = False
        try:
            if sys.platform == "win32":
                import pythoncom

                # Voice calls run on a worker thread. Each call needs its own
                # COM apartment and SAPI object to avoid stale/hung playback.
                pythoncom.CoInitialize()
                com_initialized = True
                speaker = self._tts_module.Dispatch("SAPI.SpVoice")
                speaker.Speak(message)
                speaker = None
            else:
                if self._tts is None:
                    self._tts = self._tts_module.init()
                self._tts.say(message)
                self._tts.runAndWait()
            return True
        except Exception as exc:
            self.error = f"TTS failed: {exc}"
            print(self.error)
            return False
        finally:
            if com_initialized:
                pythoncom.CoUninitialize()

    def listen_yes_no(self, timeout: float = 4.0, stop_event=None) -> VoiceAnswer:
        return self.listen_yes_no_in_language(timeout, "en", stop_event=stop_event)

    def listen_initial(self, timeout: float = 4.0, stop_event=None) -> VoiceAnswer:
        """Detect language once on the first response when Whisper is configured."""
        if self.whisper_model_path:
            result = self._listen_whisper(timeout, language=None, stop_event=stop_event)
            if result.status != "UNAVAILABLE":
                return result
        return self.listen_yes_no_in_language(timeout, "en", stop_event=stop_event)

    def listen_yes_no_in_language(self, timeout: float, language: str, stop_event=None) -> VoiceAnswer:
        if language == "zh" and self.whisper_model_path:
            return self._listen_whisper(timeout, language="zh", stop_event=stop_event)
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
                    if stop_event is not None and stop_event.is_set():
                        return VoiceAnswer(None, 0.0, "CANCELLED")
                    try:
                        data = chunks.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(data):
                        text = json.loads(recognizer.Result()).get("text", "").lower()
                        transcript = f"{transcript} {text}".strip()
                        answer = _answer_from_text(transcript)
                        if answer:
                            return VoiceAnswer(answer, 0.9, "RECOGNIZED", transcript, "en")

            final_text = json.loads(recognizer.FinalResult()).get("text", "").lower()
            transcript = f"{transcript} {final_text}".strip()
            answer = _answer_from_text(transcript)
            if answer:
                return VoiceAnswer(answer, 0.8, "RECOGNIZED", transcript, "en")
            return VoiceAnswer(None, 0.0, "TIMEOUT" if not transcript else "UNRECOGNIZED", transcript, "en")
        except Exception as exc:
            self.error = f"speech recognition failed: {exc}"
            return VoiceAnswer(None, 0.0, "UNAVAILABLE")

    def _listen_whisper(self, timeout: float, language: str | None, stop_event=None) -> VoiceAnswer:
        try:
            import numpy as np
            import sounddevice as sd
            if self._whisper is None:
                return VoiceAnswer(None, 0.0, "UNAVAILABLE")

            device_info = sd.query_devices(self.audio_device, "input")
            source_rate = int(device_info.get("default_samplerate") or 16000)
            block_duration = 0.1
            block_size = max(1, int(source_rate * block_duration))
            deadline = time.monotonic() + timeout
            chunks = []
            speech_started = False
            silence_duration = 0.0
            with sd.InputStream(
                samplerate=source_rate,
                channels=1,
                dtype="float32",
                device=self.audio_device,
                blocksize=block_size,
            ) as stream:
                while time.monotonic() < deadline:
                    if stop_event is not None and stop_event.is_set():
                        return VoiceAnswer(None, 0.0, "CANCELLED")
                    chunk, _ = stream.read(block_size)
                    chunk = chunk.reshape(-1).copy()
                    chunks.append(chunk)
                    peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
                    if peak >= 0.001:
                        speech_started = True
                        silence_duration = 0.0
                    elif speech_started:
                        silence_duration += block_duration
                        if silence_duration >= 0.8:
                            break
            audio = np.concatenate(chunks) if chunks else np.empty(0, dtype="float32")
            if stop_event is not None and stop_event.is_set():
                return VoiceAnswer(None, 0.0, "CANCELLED")
            if source_rate != 16000 and len(audio):
                target_length = max(1, round(len(audio) * 16000 / source_rate))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_length),
                    np.arange(len(audio)),
                    audio,
                ).astype("float32")

            segments, info = self._whisper.transcribe(
                audio,
                language=language,
                beam_size=1,
                best_of=1,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            detected = language or getattr(info, "language", None)
            probability = float(getattr(info, "language_probability", 0.8) or 0.8)

            # Very short Chinese replies such as "可以" are difficult to
            # language-detect. If the one-time automatic pass is inconclusive,
            # reuse the same audio for one fixed-Chinese verification pass.
            # This does not record again or repeat language detection later.
            if language is None and (detected != "zh" or not _answer_from_text(transcript, detected or "en")):
                zh_segments, _ = self._whisper.transcribe(
                    audio,
                    language="zh",
                    beam_size=1,
                    best_of=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt="请回答可以、不可以或需要帮助。",
                )
                zh_transcript = " ".join(segment.text.strip() for segment in zh_segments).strip()
                zh_answer = _answer_from_text(zh_transcript, "zh")
                if zh_answer:
                    return VoiceAnswer(zh_answer, max(round(probability, 2), 0.7), "RECOGNIZED", zh_transcript, "zh")

            if detected not in ("en", "zh"):
                return VoiceAnswer(None, 0.0, "UNSUPPORTED_LANGUAGE", transcript, detected)
            answer = _answer_from_text(transcript, detected)
            if answer:
                return VoiceAnswer(answer, round(probability, 2), "RECOGNIZED", transcript, detected)
            return VoiceAnswer(
                None,
                round(probability, 2) if transcript else 0.0,
                "UNRECOGNIZED" if transcript else "TIMEOUT",
                transcript,
                detected,
            )
        except Exception as exc:
            self.error = f"Whisper unavailable: {exc}"
            return VoiceAnswer(None, 0.0, "UNAVAILABLE")


def _answer_from_text(text: str, language: str = "en") -> str | None:
    compact = "".join(text.lower().split())
    if language == "zh":
        # Whisper can emit Traditional Chinese even when the prompt uses
        # Simplified Chinese. Normalize the answer-critical characters first.
        compact = compact.translate(str.maketrans({"沒": "没", "聽": "听", "說": "说"}))
        if any(token in compact for token in ("需要帮助", "救命", "帮帮我", "帮助")):
            return "HELP"
        if compact == "不" or any(token in compact for token in ("听不到", "不是", "不能", "不可以", "没有", "不行", "否")):
            return "NO"
        if any(token in compact for token in ("听得到", "可以听到", "能听到", "可以", "能", "是的", "是", "有", "行", "好")):
            return "YES"
        return None
    words = set(text.lower().split())
    lowered = text.lower().strip()
    if words & {"no", "nope", "cannot", "can't"} or "can not" in lowered:
        return "NO"
    if words & {"yes", "yeah", "yep"} or "i can" in lowered:
        return "YES"
    if "help" in words:
        return "HELP"
    return None


def record_wav(duration: float = 5.0, audio_device: int | None = None, sample_rate: int = 16000) -> bytes:
    """Record a mono WAV clip and return it as bytes.

    This keeps cloud triage compatible with the older helper-based audio path.
    If recording is unavailable, callers can treat the raised exception as a
    signal to fall back to gesture or offline handling.
    """
    try:
        import io
        import sounddevice as sd

        frames = max(1, int(duration * sample_rate))
        audio = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=audio_device,
        )
        sd.wait()

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        return buffer.getvalue()
    except Exception as exc:
        raise RuntimeError(f"audio recording unavailable: {exc}") from exc
