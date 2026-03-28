from __future__ import annotations

import asyncio
import logging
import os

from realtime_translator.config.models import ModelProviderSettings
from realtime_translator.models.events import TranscriptSegment, TranslationSegment
from realtime_translator.translation.base import BaseTranslator

logger = logging.getLogger(__name__)


class TransformersMarianTranslator(BaseTranslator):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings
        self._pipelines: dict[tuple[str, str], object] = {}
        self._lock = asyncio.Lock()

    async def translate(
        self,
        transcript: TranscriptSegment,
        target_language: str,
    ) -> TranslationSegment | None:
        async with self._lock:
            return await asyncio.to_thread(self._translate_blocking, transcript, target_language)

    def _translate_blocking(
        self,
        transcript: TranscriptSegment,
        target_language: str,
    ) -> TranslationSegment | None:
        pipeline = self._get_pipeline(transcript.language, target_language)
        result = pipeline(transcript.text, max_length=int(self.settings.options.get("max_length", 256)))
        translated_text = result[0]["translation_text"].strip()
        if not translated_text:
            return None
        return TranslationSegment(
            original_text=transcript.text,
            translated_text=translated_text,
            source_language=transcript.language,
            target_language=target_language,
            direction=transcript.direction,
            source=transcript.source,
            started_at=transcript.started_at,
            ended_at=transcript.ended_at,
        )

    def _get_pipeline(self, source_language: str, target_language: str):
        key = (source_language, target_language)
        if key not in self._pipelines:
            from transformers import pipeline  # type: ignore

            device = 0 if self.settings.device == "cuda" else -1
            model_name = self._resolve_model_name(source_language, target_language)
            self._pipelines[key] = pipeline(
                task="translation",
                model=model_name,
                device=device,
            )
        return self._pipelines[key]

    def _resolve_model_name(self, source_language: str, target_language: str) -> str:
        model_mapping = self.settings.options.get("models", {})
        pair_key = f"{source_language}-{target_language}"
        if isinstance(model_mapping, dict) and pair_key in model_mapping:
            return str(model_mapping[pair_key])
        if self.settings.model:
            return self.settings.model
        raise ValueError(
            f"No translation model configured for language pair {source_language}->{target_language}"
        )


class OpenAITranslator(BaseTranslator):
    def __init__(self, settings: ModelProviderSettings) -> None:
        self.settings = settings
        self._lock = asyncio.Lock()

    async def translate(
        self,
        transcript: TranscriptSegment,
        target_language: str,
    ) -> TranslationSegment | None:
        async with self._lock:
            return await asyncio.to_thread(self._translate_blocking, transcript, target_language)

    def _translate_blocking(
        self,
        transcript: TranscriptSegment,
        target_language: str,
    ) -> TranslationSegment | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI translation provider")

        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.settings.model or "gpt-4.1-mini",
            input=(
                f"Translate the following text from {transcript.language} to {target_language}. "
                "Keep it concise and natural.\n\n"
                f"{transcript.text}"
            ),
            temperature=0.2,
        )
        translated_text = response.output_text.strip()
        if not translated_text:
            return None
        return TranslationSegment(
            original_text=transcript.text,
            translated_text=translated_text,
            source_language=transcript.language,
            target_language=target_language,
            direction=transcript.direction,
            source=transcript.source,
            started_at=transcript.started_at,
            ended_at=transcript.ended_at,
        )
