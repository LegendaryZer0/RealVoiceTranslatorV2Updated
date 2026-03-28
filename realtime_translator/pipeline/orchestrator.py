from __future__ import annotations

import asyncio
import contextlib
import logging

from realtime_translator.audio.audio_router import AudioRouter
from realtime_translator.config.models import AppConfig
from realtime_translator.factories import build_stt, build_translator, build_tts, build_voice_converter
from realtime_translator.pipeline.pipeline_incoming import build_incoming_pipeline
from realtime_translator.pipeline.pipeline_outgoing import build_outgoing_pipeline
from realtime_translator.runtime.subtitles import BaseSubtitleSink

logger = logging.getLogger(__name__)


class ApplicationOrchestrator:
    def __init__(self, config: AppConfig, subtitle_sink: BaseSubtitleSink | None = None) -> None:
        self.config = config
        self.subtitle_sink = subtitle_sink
        self.router = AudioRouter(config.audio)
        self.stt = build_stt(config)
        self.translator = build_translator(config)
        self.tts = build_tts(config)
        self.voice_converter = build_voice_converter(config)
        self._pipelines = []

    async def run(self) -> None:
        if self.config.outgoing.enabled:
            self._pipelines.append(
                build_outgoing_pipeline(
                    self.config,
                    self.router,
                    self.stt,
                    self.translator,
                    self.tts,
                    self.voice_converter,
                    self.subtitle_sink,
                )
            )
        if self.config.incoming.enabled:
            self._pipelines.append(
                build_incoming_pipeline(
                    self.config,
                    self.router,
                    self.stt,
                    self.translator,
                    self.tts,
                    self.voice_converter,
                    self.subtitle_sink,
                )
            )
        if not self._pipelines:
            raise RuntimeError("No pipelines are enabled")

        tasks = [asyncio.create_task(pipeline.run()) for pipeline in self._pipelines]
        logger.info("Application orchestrator started with %s pipeline(s)", len(tasks))
        try:
            await asyncio.gather(*tasks)
        finally:
            await self.stop()

    async def stop(self) -> None:
        for pipeline in self._pipelines:
            with contextlib.suppress(Exception):
                await pipeline.stop()
