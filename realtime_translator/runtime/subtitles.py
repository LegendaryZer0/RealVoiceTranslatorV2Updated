from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import asdict

from realtime_translator.models.events import SubtitleEvent

logger = logging.getLogger(__name__)


class BaseSubtitleSink(abc.ABC):
    @abc.abstractmethod
    async def publish(self, event: SubtitleEvent) -> None:
        raise NotImplementedError


class ConsoleSubtitleSink(BaseSubtitleSink):
    async def publish(self, event: SubtitleEvent) -> None:
        logger.info(
            "[%s] %s | %s -> %s",
            event.direction,
            event.speaker,
            event.original_text,
            event.translated_text,
        )


class FanoutSubtitleSink(BaseSubtitleSink):
    def __init__(self, sinks: list[BaseSubtitleSink]) -> None:
        self.sinks = sinks

    async def publish(self, event: SubtitleEvent) -> None:
        await asyncio.gather(*(sink.publish(event) for sink in self.sinks))


class MemorySubtitleSink(BaseSubtitleSink):
    def __init__(self, max_items: int = 200) -> None:
        self.max_items = max_items
        self.items: list[str] = []

    async def publish(self, event: SubtitleEvent) -> None:
        payload = str(asdict(event))
        self.items.append(payload)
        if len(self.items) > self.max_items:
            self.items.pop(0)
