from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict

from realtime_translator.models.events import SubtitleEvent
from realtime_translator.runtime.subtitles import BaseSubtitleSink

logger = logging.getLogger(__name__)


class WebSocketSubtitleServer(BaseSubtitleSink):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server = None
        self._clients: set[object] = set()

    async def start(self) -> None:
        import websockets  # type: ignore

        self._server = await websockets.serve(self._client_handler, self.host, self.port)
        logger.info("Subtitle websocket server listening on ws://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def publish(self, event: SubtitleEvent) -> None:
        if not self._clients:
            return
        message = json.dumps(asdict(event), ensure_ascii=False)
        stale_clients: list[object] = []
        for client in self._clients:
            try:
                await client.send(message)
            except Exception:
                stale_clients.append(client)
        for client in stale_clients:
            self._clients.discard(client)

    async def _client_handler(self, websocket) -> None:  # pragma: no cover - network I/O
        self._clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        finally:
            self._clients.discard(websocket)
