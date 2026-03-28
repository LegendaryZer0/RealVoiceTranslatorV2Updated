from __future__ import annotations

from realtime_translator.audio.audio_router import AudioRouter
from realtime_translator.audio.microphone_capture import MicrophoneCaptureService
from realtime_translator.config.models import AppConfig
from realtime_translator.pipeline.common import PipelineDependencies, StreamingDirectionPipeline
from realtime_translator.runtime.subtitles import BaseSubtitleSink
from realtime_translator.stt.base import BaseSpeechRecognizer
from realtime_translator.translation.base import BaseTranslator
from realtime_translator.tts.base import BaseTextToSpeech
from realtime_translator.utils.vad import SpeechSegmenter
from realtime_translator.voice.base import BaseVoiceConverter


def build_outgoing_pipeline(
    config: AppConfig,
    router: AudioRouter,
    stt: BaseSpeechRecognizer,
    translator: BaseTranslator,
    tts: BaseTextToSpeech,
    voice_converter: BaseVoiceConverter,
    subtitle_sink: BaseSubtitleSink | None,
) -> StreamingDirectionPipeline:
    capture_service = MicrophoneCaptureService(
        config.audio.microphone,
        direction=config.outgoing.direction,
        language_hint=config.outgoing.input_language,
    )
    deps = PipelineDependencies(
        capture_service=capture_service,
        segmenter=SpeechSegmenter(config.vad),
        stt=stt,
        translator=translator,
        tts=tts,
        voice_converter=voice_converter,
        output_func=router.play_to_virtual_microphone,
        subtitle_sink=subtitle_sink,
    )
    return StreamingDirectionPipeline(config.outgoing, deps)
