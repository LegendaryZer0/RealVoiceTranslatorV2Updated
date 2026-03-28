from __future__ import annotations

import asyncio
import logging

from realtime_translator.config.models import ModelProviderSettings
from realtime_translator.models.events import SynthesizedAudio
from realtime_translator.voice.base import BaseVoiceConverter

logger = logging.getLogger(__name__)


class NoOpVoiceConverter(BaseVoiceConverter):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings

    async def convert(self, audio: SynthesizedAudio) -> SynthesizedAudio:
        return audio


class RvcVoiceConverter(BaseVoiceConverter):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings

    async def convert(self, audio: SynthesizedAudio) -> SynthesizedAudio:
        return await asyncio.to_thread(self._convert_blocking, audio)

    def _convert_blocking(self, audio: SynthesizedAudio) -> SynthesizedAudio:
        logger.warning(
            "RVC provider is configured but concrete RVC integration is environment-specific. "
            "Returning synthesized audio unchanged."
        )
        return audio
