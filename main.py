from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from realtime_translator.audio.audio_router import AudioRouter
from realtime_translator.config.manager import ConfigManager
from realtime_translator.pipeline.orchestrator import ApplicationOrchestrator
from realtime_translator.runtime.subtitles import ConsoleSubtitleSink, FanoutSubtitleSink
from realtime_translator.runtime.websocket_server import WebSocketSubtitleServer
from realtime_translator.utils.logger import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime bidirectional voice translator")
    parser.add_argument(
        "--config",
        default=str(Path("config") / "config.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Load config, print summary and exit",
    )
    return parser.parse_args()


async def run_application(config_path: str) -> None:
    config = ConfigManager.load(config_path)
    setup_logging(config.logging)

    subtitle_sinks = []
    websocket_server = None
    if config.subtitles.enabled and config.subtitles.print_to_console:
        subtitle_sinks.append(ConsoleSubtitleSink())
    if config.runtime.websocket_enabled:
        websocket_server = WebSocketSubtitleServer(
            config.runtime.websocket_host,
            config.runtime.websocket_port,
        )
        await websocket_server.start()
        subtitle_sinks.append(websocket_server)

    subtitle_sink = FanoutSubtitleSink(subtitle_sinks) if subtitle_sinks else None
    orchestrator = ApplicationOrchestrator(config, subtitle_sink)

    try:
        await orchestrator.run()
    finally:
        if websocket_server is not None:
            await websocket_server.stop()


def validate_config(config_path: str) -> None:
    config = ConfigManager.load(config_path)
    setup_logging(config.logging)
    logger = logging.getLogger("config")
    logger.info("Configuration loaded successfully from %s", config_path)
    logger.info(
        "Pipelines enabled | outgoing=%s incoming=%s | STT=%s | Translation=%s | TTS=%s",
        config.outgoing.enabled,
        config.incoming.enabled,
        config.stt.provider,
        config.translation.provider,
        config.tts.provider,
    )


def list_devices() -> None:
    for index, device in enumerate(AudioRouter.list_devices(), start=1):
        print(f"{index:02d}. {device}")


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.validate_config:
        validate_config(args.config)
        return

    try:
        asyncio.run(run_application(args.config))
    except KeyboardInterrupt:
        logging.getLogger("main").info("Shutdown requested by user")


if __name__ == "__main__":
    main()
