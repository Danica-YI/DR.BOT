"""Shared answer types and multi-frame gesture stabilization."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass



@dataclass(frozen=True)
class DetectedAnswer:
    answer: str
    source: str
    confidence: float


class GestureStabilizer:
    def __init__(self, window_size: int = 3, required_matches: int = 2):
        if required_matches > window_size:
            raise ValueError("required_matches cannot exceed window_size")
        self.samples = deque(maxlen=window_size)
        self.required_matches = required_matches

    def reset(self) -> None:
        self.samples.clear()

    def add(self, classifier_status: str | None) -> DetectedAnswer | None:
        answer = classifier_status
        self.samples.append(answer)
        valid = [sample for sample in self.samples if sample]
        if len(valid) < self.required_matches:
            return None
        candidate, count = Counter(valid).most_common(1)[0]
        if count < self.required_matches or answer != candidate:
            return None
        return DetectedAnswer(candidate, "gesture", round(count / len(self.samples), 2))
