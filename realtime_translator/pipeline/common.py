from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from realtime_translator.config.models import PipelineDirectionSettings
from realtime_translator.models.events import (
    AudioFrame,
    SpeechSegment,
    SubtitleEvent,
    SynthesizedAudio,
    TranscriptSegment,
    TranslationSegment,
)
from realtime_translator.runtime.subtitles import BaseSubtitleSink
from realtime_translator.stt.base import BaseSpeechRecognizer
from realtime_translator.translation.base import BaseTranslator
from realtime_translator.tts.base import BaseTextToSpeech
from realtime_translator.utils.vad import SpeechSegmenter
from realtime_translator.voice.base import BaseVoiceConverter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineDependencies:
    capture_service: object
    segmenter: SpeechSegmenter
    stt: BaseSpeechRecognizer
    translator: BaseTranslator
    tts: BaseTextToSpeech
    voice_converter: BaseVoiceConverter
    output_func: Callable[[SynthesizedAudio], Awaitable[None]]
    subtitle_sink: BaseSubtitleSink | None = None


class StreamingDirectionPipeline:
    def __init__(self, settings: PipelineDirectionSettings, deps: PipelineDependencies) -> None:
        self.settings = settings
        self.deps = deps
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        self._frame_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(
            maxsize=settings.input_queue_maxsize
        )
        self._segment_queue: asyncio.Queue[SpeechSegment] = asyncio.Queue(
            maxsize=settings.segment_queue_maxsize
        )
        self._transcript_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue(
            maxsize=settings.transcript_queue_maxsize
        )
        self._translation_queue: asyncio.Queue[TranslationSegment] = asyncio.Queue(
            maxsize=settings.translation_queue_maxsize
        )
        self._synthesis_queue: asyncio.Queue[SynthesizedAudio] = asyncio.Queue(
            maxsize=settings.synthesis_queue_maxsize
        )
        self._output_queue: asyncio.Queue[SynthesizedAudio] = asyncio.Queue(
            maxsize=settings.synthesis_queue_maxsize
        )

    async def run(self) -> None:
        await self.deps.capture_service.start(self._frame_queue)
        self._tasks = [
            asyncio.create_task(self._segment_worker(), name=f"{self.settings.direction}-segment"),
            asyncio.create_task(self._stt_worker(), name=f"{self.settings.direction}-stt"),
            asyncio.create_task(self._translation_worker(), name=f"{self.settings.direction}-translate"),
            asyncio.create_task(self._tts_worker(), name=f"{self.settings.direction}-tts"),
            asyncio.create_task(self._voice_worker(), name=f"{self.settings.direction}-voice"),
            asyncio.create_task(self._output_worker(), name=f"{self.settings.direction}-output"),
        ]
        logger.info("Pipeline %s started", self.settings.direction)
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stop_event.is_set():
            pass
        else:
            self._stop_event.set()
        await self.deps.capture_service.stop()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def _segment_worker(self) -> None:
        while True:
            frame = await self._frame_queue.get()
            try:
                for segment in self.deps.segmenter.process_frame(frame):
                    await self._segment_queue.put(segment)
            except Exception:
                logger.exception("VAD/segment stage failed for %s", self.settings.direction)

    async def _stt_worker(self) -> None:
        while True:
            segment = await self._segment_queue.get()
            try:
                transcript = await self.deps.stt.transcribe(segment)
                if transcript and transcript.text.strip():
                    await self._transcript_queue.put(transcript)
            except Exception:
                logger.exception("STT stage failed for %s", self.settings.direction)

    async def _translation_worker(self) -> None:
        while True:
            transcript = await self._transcript_queue.get()
            try:
                translation = await self.deps.translator.translate(
                    transcript,
                    target_language=self.settings.output_language,
                )
                if translation and translation.translated_text.strip():
                    await self._translation_queue.put(translation)
                    await self._publish_subtitle(translation)
            except Exception:
                logger.exception("Translation stage failed for %s", self.settings.direction)

    async def _tts_worker(self) -> None:
        while True:
            translation = await self._translation_queue.get()
            try:
                synthesized = await self.deps.tts.synthesize(translation)
                if synthesized:
                    await self._synthesis_queue.put(synthesized)
            except Exception:
                logger.exception("TTS stage failed for %s", self.settings.direction)

    async def _voice_worker(self) -> None:
        while True:
            synthesized = await self._synthesis_queue.get()
            try:
                converted = await self.deps.voice_converter.convert(synthesized)
                await self._output_queue.put(converted)
            except Exception:
                logger.exception("Voice conversion/output pre-stage failed for %s", self.settings.direction)

    async def _output_worker(self) -> None:
        while True:
            converted = await self._output_queue.get()
            try:
                await self.deps.output_func(converted)
            except Exception:
                logger.exception("Audio output stage failed for %s", self.settings.direction)

    async def _publish_subtitle(self, translation: TranslationSegment) -> None:
        if not self.deps.subtitle_sink:
            return
        await self.deps.subtitle_sink.publish(
            SubtitleEvent(
                direction=self.settings.direction,
                speaker=self.settings.speaker_label,
                original_text=translation.original_text,
                translated_text=translation.translated_text,
                source_language=translation.source_language,
                target_language=translation.target_language,
            )
        )
