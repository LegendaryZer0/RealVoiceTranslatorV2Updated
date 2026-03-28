from __future__ import annotations

from abc import ABC, abstractmethod

from realtime_translator.models.events import SynthesizedAudio, TranslationSegment


class BaseTextToSpeech(ABC):
    @abstractmethod
    async def synthesize(self, segment: TranslationSegment) -> SynthesizedAudio | None:
        raise NotImplementedError
