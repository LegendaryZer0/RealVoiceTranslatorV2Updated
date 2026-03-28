from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AppSettings:
    name: str = "Realtime Voice Translator"
    version: str = "0.1.0"
    log_level: str = "INFO"
    default_timeout_s: float = 15.0


@dataclass(slots=True)
class CaptureSettings:
    enabled: bool = True
    device_name: str | int | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 20
    queue_maxsize: int = 256


@dataclass(slots=True)
class AudioSettings:
    microphone: CaptureSettings = field(default_factory=CaptureSettings)
    system_loopback: CaptureSettings = field(
        default_factory=lambda: CaptureSettings(channels=2, device_name=None)
    )
    headphones_output_device: str | int | None = None
    virtual_microphone_output_device: str | int | None = None
    target_output_sample_rate: int = 24000


@dataclass(slots=True)
class VadSettings:
    enabled: bool = True
    aggressiveness: int = 2
    min_speech_ms: int = 300
    silence_ms: int = 500
    max_segment_ms: int = 6000
    pad_ms: int = 200


@dataclass(slots=True)
class ModelProviderSettings:
    provider: str
    model: str | None = None
    device: str = "auto"
    compute_type: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    voice: str | None = None
    enabled: bool = True
    options: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineDirectionSettings:
    enabled: bool
    direction: str
    input_language: str
    output_language: str
    speaker_label: str
    source_name: str
    target_name: str
    input_queue_maxsize: int = 256
    segment_queue_maxsize: int = 32
    transcript_queue_maxsize: int = 32
    translation_queue_maxsize: int = 32
    synthesis_queue_maxsize: int = 16


@dataclass(slots=True)
class SubtitleSettings:
    enabled: bool = True
    print_to_console: bool = True


@dataclass(slots=True)
class LoggingSettings:
    level: str = "INFO"
    log_dir: str = "logs"
    json_logs: bool = False


@dataclass(slots=True)
class RuntimeSettings:
    websocket_enabled: bool = False
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 8765
    gpu_enabled: bool = True
    prefer_local_models: bool = True


@dataclass(slots=True)
class AppConfig:
    app: AppSettings = field(default_factory=AppSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    vad: VadSettings = field(default_factory=VadSettings)
    stt: ModelProviderSettings = field(
        default_factory=lambda: ModelProviderSettings(
            provider="faster-whisper",
            model="small",
            compute_type="int8_float16",
        )
    )
    translation: ModelProviderSettings = field(
        default_factory=lambda: ModelProviderSettings(
            provider="transformers-marian",
            model=None,
            options={
                "models": {
                    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
                    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
                }
            },
        )
    )
    tts: ModelProviderSettings = field(
        default_factory=lambda: ModelProviderSettings(
            provider="pyttsx3",
            model=None,
            voice=None,
        )
    )
    voice_conversion: ModelProviderSettings = field(
        default_factory=lambda: ModelProviderSettings(
            provider="none",
            enabled=False,
        )
    )
    outgoing: PipelineDirectionSettings = field(
        default_factory=lambda: PipelineDirectionSettings(
            enabled=True,
            direction="outgoing",
            input_language="ru",
            output_language="en",
            speaker_label="You",
            source_name="microphone",
            target_name="virtual_microphone",
        )
    )
    incoming: PipelineDirectionSettings = field(
        default_factory=lambda: PipelineDirectionSettings(
            enabled=True,
            direction="incoming",
            input_language="en",
            output_language="ru",
            speaker_label="Remote",
            source_name="system_loopback",
            target_name="headphones",
        )
    )
    subtitles: SubtitleSettings = field(default_factory=SubtitleSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
