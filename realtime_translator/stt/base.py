from __future__ import annotations

from abc import ABC, abstractmethod

from realtime_translator.models.events import SpeechSegment, TranscriptSegment


class BaseSpeechRecognizer(ABC):
    @abstractmethod
    async def transcribe(self, segment: SpeechSegment) -> TranscriptSegment | None:
        raise NotImplementedError
