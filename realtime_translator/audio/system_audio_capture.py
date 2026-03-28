from __future__ import annotations

import asyncio
import logging
import threading
import time

import numpy as np

from realtime_translator.config.models import CaptureSettings
from realtime_translator.models.events import AudioFrame

logger = logging.getLogger(__name__)


class SystemAudioCaptureService:
    def __init__(self, settings: CaptureSettings, direction: str, language_hint: str) -> None:
        self.settings = settings
        self.direction = direction
        self.language_hint = language_hint
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    async def start(self, queue: asyncio.Queue[AudioFrame]) -> None:
        loop = asyncio.get_running_loop()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(loop, queue),
            name="system-audio-capture",
            daemon=True,
        )
        self._thread.start()
        logger.info("System loopback capture started, device=%s", self.settings.device_name or "default speaker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def _capture_loop(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[AudioFrame]) -> None:
        frames_per_chunk = int(self.settings.sample_rate * self.settings.chunk_ms / 1000)

        import soundcard as sc  # type: ignore

        speaker = sc.get_speaker(self.settings.device_name) if self.settings.device_name else sc.default_speaker()
        with speaker.recorder(
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            blocksize=frames_per_chunk,
        ) as recorder:
            while not self._stop_event.is_set():
                data = recorder.record(numframes=frames_per_chunk)
                pcm = self._float_to_pcm16(data)
                frame = AudioFrame(
                    pcm=pcm,
                    sample_rate=self.settings.sample_rate,
                    channels=self.settings.channels,
                    language_hint=self.language_hint,
                    direction=self.direction,
                    source="system_loopback",
                    timestamp=time.time(),
                )
                loop.call_soon_threadsafe(self._put_nowait, queue, frame)

    @staticmethod
    def _float_to_pcm16(data: np.ndarray) -> bytes:
        clipped = np.clip(data, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16).tobytes()

    @staticmethod
    def _put_nowait(queue: asyncio.Queue[AudioFrame], frame: AudioFrame) -> None:
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("System audio frame dropped because input queue is full")
