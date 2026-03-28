from __future__ import annotations

from abc import ABC, abstractmethod

from realtime_translator.models.events import TranscriptSegment, TranslationSegment


class BaseTranslator(ABC):
    @abstractmethod
    async def translate(
        self,
        transcript: TranscriptSegment,
        target_language: str,
    ) -> TranslationSegment | None:
        raise NotImplementedError
