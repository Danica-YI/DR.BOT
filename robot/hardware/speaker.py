"""Robot speaker abstraction backed by local text-to-speech."""

from __future__ import annotations

from device.voice import OfflineVoice


class Speaker:
    """Speak text using the local machine speaker when available."""

    def __init__(self) -> None:
        self.voice = OfflineVoice(enabled=True)

    def say(self, message: str) -> bool:
        """Speak a text message through the local audio output device."""
        return self.voice.say(message)
