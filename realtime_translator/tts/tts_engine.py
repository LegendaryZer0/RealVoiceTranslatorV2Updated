from __future__ import annotations

import asyncio
import logging
import tempfile
import wave
from pathlib import Path

from realtime_translator.config.models import ModelProviderSettings
from realtime_translator.models.events import SynthesizedAudio, TranslationSegment
from realtime_translator.tts.base import BaseTextToSpeech

logger = logging.getLogger(__name__)


class Pyttsx3TTSEngine(BaseTextToSpeech):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings
        self._engine = None
        self._lock = asyncio.Lock()

    async def synthesize(self, segment: TranslationSegment) -> SynthesizedAudio | None:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_blocking, segment)

    def _synthesize_blocking(self, segment: TranslationSegment) -> SynthesizedAudio | None:
        engine = self._get_engine()
        self._apply_language_voice(engine, segment.target_language)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)

        engine.save_to_file(segment.translated_text, str(output_path))
        engine.runAndWait()

        with wave.open(str(output_path), "rb") as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
        output_path.unlink(missing_ok=True)

        return SynthesizedAudio(
            pcm=pcm,
            sample_rate=sample_rate,
            channels=channels,
            text=segment.translated_text,
            language=segment.target_language,
            direction=segment.direction,
            source=segment.source,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
        )

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3  # type: ignore

            self._engine = pyttsx3.init()
            if self.settings.voice:
                for voice in self._engine.getProperty("voices"):
                    if self.settings.voice.lower() in str(voice.id).lower():
                        self._engine.setProperty("voice", voice.id)
                        break
            if rate := self.settings.options.get("rate"):
                self._engine.setProperty("rate", int(rate))
        return self._engine

    def _apply_language_voice(self, engine, language: str) -> None:
        configured_voices = self.settings.options.get("voices", {})
        if not isinstance(configured_voices, dict):
            return
        voice_hint = configured_voices.get(language)
        if not voice_hint:
            return
        for voice in engine.getProperty("voices"):
            if str(voice_hint).lower() in str(voice.id).lower():
                engine.setProperty("voice", voice.id)
                return


class CoquiTTSEngine(BaseTextToSpeech):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings
        self._tts = None
        self._lock = asyncio.Lock()

    async def synthesize(self, segment: TranslationSegment) -> SynthesizedAudio | None:
        async with self._lock:
            return await asyncio.to_thread(self._synthesize_blocking, segment)

    def _synthesize_blocking(self, segment: TranslationSegment) -> SynthesizedAudio | None:
        tts = self._get_tts()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output_path = Path(handle.name)
        tts.tts_to_file(text=segment.translated_text, file_path=str(output_path))
        with wave.open(str(output_path), "rb") as wav_file:
            pcm = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
        output_path.unlink(missing_ok=True)
        return SynthesizedAudio(
            pcm=pcm,
            sample_rate=sample_rate,
            channels=channels,
            text=segment.translated_text,
            language=segment.target_language,
            direction=segment.direction,
            source=segment.source,
            started_at=segment.started_at,
            ended_at=segment.ended_at,
        )

    def _get_tts(self):
        if self._tts is None:
            from TTS.api import TTS  # type: ignore

            self._tts = TTS(model_name=self.settings.model, gpu=self.settings.device == "cuda")
        return self._tts
