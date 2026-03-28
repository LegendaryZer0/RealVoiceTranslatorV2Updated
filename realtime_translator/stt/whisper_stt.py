from __future__ import annotations

import asyncio
import ctypes.util
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
        self._resolved_device: str | None = None

    async def transcribe(self, segment: SpeechSegment) -> TranscriptSegment | None:
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_blocking, segment)

    def _transcribe_blocking(self, segment: SpeechSegment) -> TranscriptSegment | None:
        try:
            return self._transcribe_once(segment)
        except RuntimeError as exc:
            if not self._is_gpu_runtime_error(exc):
                raise
            logger.warning(
                "Whisper CUDA runtime is unavailable (%s). Falling back to CPU for subsequent STT.",
                exc,
            )
            self._model = None
            self._resolved_device = "cpu"
            return self._transcribe_once(segment)

    def _transcribe_once(self, segment: SpeechSegment) -> TranscriptSegment | None:
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

            device = self._resolved_device or self._select_device()
            try:
                self._model = WhisperModel(
                    self.settings.model or "small",
                    device=device,
                    compute_type=self._compute_type_for(device),
                )
                self._resolved_device = device
            except Exception:
                logger.warning("Whisper init for device=%s failed, falling back to CPU", device)
                self._model = WhisperModel(
                    self.settings.model or "small",
                    device="cpu",
                    compute_type="int8",
                )
                self._resolved_device = "cpu"
        return self._model

    def _select_device(self) -> str:
        configured = (self.settings.device or "auto").lower()
        if configured == "cpu":
            return "cpu"
        if configured == "cuda":
            return "cuda"
        if self._cuda_runtime_available():
            return "cuda"
        return "cpu"

    def _compute_type_for(self, device: str) -> str:
        if device == "cpu":
            return "int8"
        return self.settings.compute_type or "int8_float16"

    @staticmethod
    def _cuda_runtime_available() -> bool:
        return bool(ctypes.util.find_library("cublas64_12") or ctypes.util.find_library("cublas"))

    @staticmethod
    def _is_gpu_runtime_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return "cublas" in message or "cuda" in message

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
