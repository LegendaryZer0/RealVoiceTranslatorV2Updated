from __future__ import annotations

from realtime_translator.config.models import AppConfig
from realtime_translator.stt.base import BaseSpeechRecognizer
from realtime_translator.stt.whisper_stt import FasterWhisperSTT
from realtime_translator.translation.base import BaseTranslator
from realtime_translator.translation.translator import OpenAITranslator, TransformersMarianTranslator
from realtime_translator.tts.base import BaseTextToSpeech
from realtime_translator.tts.tts_engine import CoquiTTSEngine, Pyttsx3TTSEngine
from realtime_translator.voice.base import BaseVoiceConverter
from realtime_translator.voice.voice_conversion import NoOpVoiceConverter, RvcVoiceConverter


def build_stt(config: AppConfig) -> BaseSpeechRecognizer:
    provider = config.stt.provider.lower()
    if provider == "faster-whisper":
        return FasterWhisperSTT(config.stt)
    raise ValueError(f"Unsupported STT provider: {config.stt.provider}")


def build_translator(config: AppConfig) -> BaseTranslator:
    provider = config.translation.provider.lower()
    if provider == "transformers-marian":
        return TransformersMarianTranslator(config.translation)
    if provider == "openai":
        return OpenAITranslator(config.translation)
    raise ValueError(f"Unsupported translation provider: {config.translation.provider}")


def build_tts(config: AppConfig) -> BaseTextToSpeech:
    provider = config.tts.provider.lower()
    if provider == "pyttsx3":
        return Pyttsx3TTSEngine(config.tts)
    if provider in {"coqui", "xtts"}:
        return CoquiTTSEngine(config.tts)
    raise ValueError(f"Unsupported TTS provider: {config.tts.provider}")


def build_voice_converter(config: AppConfig) -> BaseVoiceConverter:
    provider = config.voice_conversion.provider.lower()
    if not config.voice_conversion.enabled or provider == "none":
        return NoOpVoiceConverter(config.voice_conversion)
    if provider == "rvc":
        return RvcVoiceConverter(config.voice_conversion)
    raise ValueError(f"Unsupported voice conversion provider: {config.voice_conversion.provider}")
