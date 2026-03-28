from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(slots=True)
class AudioFrame:
    pcm: bytes
    sample_rate: int
    channels: int
    language_hint: str
    direction: str
    source: str
    timestamp: float = field(default_factory=time)


@dataclass(slots=True)
class SpeechSegment:
    pcm: bytes
    sample_rate: int
    channels: int
    language: str
    direction: str
    source: str
    started_at: float
    ended_at: float

    @property
    def duration_s(self) -> float:
        frame_width = 2 * self.channels
        if frame_width == 0 or self.sample_rate == 0:
            return 0.0
        return len(self.pcm) / frame_width / self.sample_rate


@dataclass(slots=True)
class TranscriptSegment:
    text: str
    language: str
    direction: str
    source: str
    started_at: float
    ended_at: float
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationSegment:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    direction: str
    source: str
    started_at: float
    ended_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SynthesizedAudio:
    pcm: bytes
    sample_rate: int
    channels: int
    text: str
    language: str
    direction: str
    source: str
    started_at: float
    ended_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubtitleEvent:
    direction: str
    speaker: str
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    timestamp: float = field(default_factory=time)
