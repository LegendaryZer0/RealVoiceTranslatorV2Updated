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
        self._models: dict[tuple[str, str], object] = {}
        self._tokenizers: dict[tuple[str, str], object] = {}
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
        model, tokenizer = self._get_model_bundle(transcript.language, target_language)
        max_length = int(self.settings.options.get("max_length", 256))
        encoded = tokenizer(
            transcript.text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if self.settings.device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        generated = model.generate(**encoded, max_length=max_length)
        translated_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
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

    def _get_model_bundle(self, source_language: str, target_language: str):
        key = (source_language, target_language)
        if key not in self._models:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore

            model_name = self._resolve_model_name(source_language, target_language)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if self.settings.device == "cuda":
                model = model.to("cuda")
            self._tokenizers[key] = tokenizer
            self._models[key] = model
        return self._models[key], self._tokenizers[key]

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
