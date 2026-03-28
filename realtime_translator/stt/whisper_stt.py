from __future__ import annotations

import asyncio
import io
import logging
import wave

from realtime_translator.config.models import ModelProviderSettings
from realtime_translator.models.events import SpeechSegment, TranscriptSegment
from realtime_translator.stt.base import BaseSpeechRecognizer

logger = logging.getLogger(__name__)


class FasterWhisperSTT(BaseSpeechRecognizer):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings
        self._model = None
        self._lock = asyncio.Lock()

    async def transcribe(self, segment: SpeechSegment) -> TranscriptSegment | None:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_blocking, segment)

    def _transcribe_blocking(self, segment: SpeechSegment) -> TranscriptSegment | None:
        model = self._get_model()
        buffer = self._to_wav_buffer(segment)
        segments, info = model.transcribe(
            buffer,
            language=segment.language,
            vad_filter=False,
            beam_size=int(self.settings.options.get("beam_size", 1)),
            best_of=int(self.settings.options.get("best_of", 1)),
            condition_on_previous_text=bool(
                self.settings.options.get("condition_on_previous_text", False)
            ),
        )
        text = " ".join(item.text.strip() for item in segments).strip()
        if not text:
            return None
        return TranscriptSegment(
            text=text,
            language=segment.language,
            direction=segment.direction,
            source=segment.source,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
            confidence=getattr(info, "language_probability", None),
            metadata={"detected_language": getattr(info, "language", segment.language)},
        )

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # type: ignore

            device = self.settings.device
            if device == "auto":
                device = "cuda"
            try:
                self._model = WhisperModel(
                    self.settings.model or "small",
                    device=device,
                    compute_type=self.settings.compute_type or "int8",
                )
            except Exception:
                logger.warning("CUDA Whisper init failed, falling back to CPU")
                self._model = WhisperModel(
                    self.settings.model or "small",
                    device="cpu",
                    compute_type="int8",
                )
        return self._model

    @staticmethod
    def _to_wav_buffer(segment: SpeechSegment) -> io.BytesIO:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(segment.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(segment.sample_rate)
            wav_file.writeframes(segment.pcm)
        buffer.seek(0)
        return buffer
