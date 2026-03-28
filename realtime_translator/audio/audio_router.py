from __future__ import annotations

import asyncio
import logging

import numpy as np

from realtime_translator.config.models import AudioSettings
from realtime_translator.models.events import SynthesizedAudio

logger = logging.getLogger(__name__)


class AudioRouter:
    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings

    async def play_to_virtual_microphone(self, audio: SynthesizedAudio) -> None:
        if not self.settings.virtual_microphone_output_device:
            raise RuntimeError("virtual_microphone_output_device is not configured")
        await asyncio.to_thread(
            self._play_pcm,
            audio.pcm,
            audio.sample_rate,
            audio.channels,
            self.settings.virtual_microphone_output_device,
        )

    async def play_to_headphones(self, audio: SynthesizedAudio) -> None:
        if not self.settings.headphones_output_device:
            raise RuntimeError("headphones_output_device is not configured")
        await asyncio.to_thread(
            self._play_pcm,
            audio.pcm,
            audio.sample_rate,
            audio.channels,
            self.settings.headphones_output_device,
        )

    @staticmethod
    def list_devices() -> list[str]:
        import sounddevice as sd  # type: ignore

        return [str(item["name"]) for item in sd.query_devices()]

    @staticmethod
    def _play_pcm(pcm: bytes, sample_rate: int, channels: int, device_name: str) -> None:
        import sounddevice as sd  # type: ignore

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=sample_rate, device=device_name, blocking=True)
        logger.debug("Playback complete on device=%s", device_name)
