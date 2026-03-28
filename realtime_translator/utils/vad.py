from __future__ import annotations

import audioop
import logging
from collections import deque
from typing import Iterable

from realtime_translator.config.models import VadSettings
from realtime_translator.models.events import AudioFrame, SpeechSegment

logger = logging.getLogger(__name__)


class SpeechSegmenter:
    def __init__(self, settings: VadSettings) -> None:
        self.settings = settings
        self._vad = None
        self._frame_duration_ms: int | None = None
        self._triggered = False
        self._speech_frames: list[bytes] = []
        self._speech_started_at: float | None = None
        self._pre_buffer: deque[AudioFrame] = deque()
        self._silence_frames = 0
        self._speech_frame_count = 0

        if settings.enabled:
            try:
                import webrtcvad  # type: ignore

                self._vad = webrtcvad.Vad(settings.aggressiveness)
            except Exception as exc:  # pragma: no cover - optional dependency at runtime
                logger.warning("WebRTC VAD unavailable, fallback energy-based VAD is used: %s", exc)

    def process_frame(self, frame: AudioFrame) -> list[SpeechSegment]:
        frame_duration_ms = int(len(frame.pcm) / (2 * frame.channels) / frame.sample_rate * 1000)
        if frame_duration_ms <= 0:
            return []
        self._frame_duration_ms = frame_duration_ms

        mono_pcm = self._ensure_mono(frame.pcm, frame.channels)
        speech = self._is_speech(mono_pcm, frame.sample_rate)
        segments: list[SpeechSegment] = []

        if not self._triggered:
            self._pre_buffer.append(frame)
            while len(self._pre_buffer) * frame_duration_ms > self.settings.pad_ms:
                self._pre_buffer.popleft()

            if speech:
                self._speech_frame_count += 1
            else:
                self._speech_frame_count = 0

            if self._speech_frame_count * frame_duration_ms >= self.settings.min_speech_ms:
                self._triggered = True
                self._speech_started_at = self._pre_buffer[0].timestamp if self._pre_buffer else frame.timestamp
                self._speech_frames = [item.pcm for item in self._pre_buffer]
                self._pre_buffer.clear()
                self._silence_frames = 0
        else:
            self._speech_frames.append(frame.pcm)
            if speech:
                self._silence_frames = 0
            else:
                self._silence_frames += 1

            speech_ms = len(self._speech_frames) * frame_duration_ms
            silence_ms = self._silence_frames * frame_duration_ms
            should_flush = silence_ms >= self.settings.silence_ms or speech_ms >= self.settings.max_segment_ms
            if should_flush:
                segments.append(
                    SpeechSegment(
                        pcm=b"".join(self._speech_frames),
                        sample_rate=frame.sample_rate,
                        channels=frame.channels,
                        language=frame.language_hint,
                        direction=frame.direction,
                        source=frame.source,
                        started_at=self._speech_started_at or frame.timestamp,
                        ended_at=frame.timestamp,
                    )
                )
                self._reset()

        return segments

    def flush(self) -> Iterable[SpeechSegment]:
        if self._triggered and self._speech_frames and self._frame_duration_ms:
            ended_at = (self._speech_started_at or 0.0) + (
                len(self._speech_frames) * self._frame_duration_ms / 1000
            )
            yield SpeechSegment(
                pcm=b"".join(self._speech_frames),
                sample_rate=16000,
                channels=1,
                language="unknown",
                direction="unknown",
                source="flush",
                started_at=self._speech_started_at or 0.0,
                ended_at=ended_at,
            )
        self._reset()

    def _reset(self) -> None:
        self._triggered = False
        self._speech_frames = []
        self._speech_started_at = None
        self._silence_frames = 0
        self._speech_frame_count = 0
        self._pre_buffer.clear()

    def _is_speech(self, pcm: bytes, sample_rate: int) -> bool:
        if self._vad is not None:
            try:
                return bool(self._vad.is_speech(pcm, sample_rate))
            except Exception:
                return self._energy_gate(pcm)
        return self._energy_gate(pcm)

    @staticmethod
    def _ensure_mono(pcm: bytes, channels: int) -> bytes:
        if channels == 1:
            return pcm
        return audioop.tomono(pcm, 2, 0.5, 0.5)

    @staticmethod
    def _energy_gate(pcm: bytes) -> bool:
        return audioop.rms(pcm, 2) > 450
