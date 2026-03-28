from __future__ import annotations

from abc import ABC, abstractmethod

from realtime_translator.models.events import SynthesizedAudio


class BaseVoiceConverter(ABC):
    @abstractmethod
    async def convert(self, audio: SynthesizedAudio) -> SynthesizedAudio:
        raise NotImplementedError
