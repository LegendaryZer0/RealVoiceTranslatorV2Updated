from __future__ import annotations

import asyncio
import logging
import threading

from realtime_translator.audio.device_resolver import resolve_sounddevice_device
from realtime_translator.config.models import CaptureSettings
from realtime_translator.models.events import AudioFrame

logger = logging.getLogger(__name__)


class MicrophoneCaptureService:
    def __init__(self, settings: CaptureSettings, direction: str, language_hint: str) -> None:
        self.settings = settings
        self.direction = direction
        self.language_hint = language_hint
        self._stream = None
        self._stop_event = threading.Event()

    async def start(self, queue: asyncio.Queue[AudioFrame]) -> None:
        self._stop_event.clear()
        loop = asyncio.get_running_loop()
        blocksize = int(self.settings.sample_rate * self.settings.chunk_ms / 1000)

        def callback(indata, frames, time_info, status) -> None:  # pragma: no cover - callback
            if status:
                logger.debug("Microphone callback status: %s", status)
            if self._stop_event.is_set():
                return
            frame = AudioFrame(
                pcm=bytes(indata),
                sample_rate=self.settings.sample_rate,
                channels=self.settings.channels,
                language_hint=self.language_hint,
                direction=self.direction,
                source="microphone",
            )
            loop.call_soon_threadsafe(self._put_nowait, queue, frame)

        import sounddevice as sd  # type: ignore

        resolved_device = resolve_sounddevice_device(self.settings.device_name, kind="input")

        self._stream = sd.RawInputStream(
            samplerate=self.settings.sample_rate,
            blocksize=blocksize,
            channels=self.settings.channels,
            dtype="int16",
            device=resolved_device,
            callback=callback,
        )
        self._stream.start()
        logger.info(
            "Microphone capture started, configured_device=%s, resolved_device=%s",
            self.settings.device_name or "default",
            resolved_device,
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def _put_nowait(queue: asyncio.Queue[AudioFrame], frame: AudioFrame) -> None:
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Microphone frame dropped because input queue is full")
