# Realtime Voice Translator

Production-like modular Python 3.11 project for bidirectional real-time voice translation during browser calls. The current implementation targets Windows first, while keeping services and configuration portable enough to extend to Linux and macOS later.

## Goal

Two independent pipelines run in parallel:

1. `RU microphone -> STT -> RU->EN translation -> EN TTS -> virtual microphone`
2. `EN system loopback -> STT -> EN->RU translation -> RU TTS -> headphones`

This allows browser call applications such as Zoom, Google Meet, Teams or Discord to use translated speech as if it were a normal audio device.

## Architecture

### Modules

- `realtime_translator/audio`
  - `microphone_capture.py`: microphone capture with `sounddevice`
  - `system_audio_capture.py`: Windows loopback capture with `soundcard`
  - `audio_router.py`: playback to headphones or virtual microphone device
- `realtime_translator/stt`
  - `whisper_stt.py`: `faster-whisper` adapter with GPU fallback
- `realtime_translator/translation`
  - `translator.py`: local MarianMT or OpenAI translation provider
- `realtime_translator/tts`
  - `tts_engine.py`: local `pyttsx3` or optional Coqui/XTTS adapter
- `realtime_translator/voice`
  - `voice_conversion.py`: no-op voice conversion by default, RVC hook placeholder
- `realtime_translator/pipeline`
  - `common.py`: async queues and stage workers
  - `pipeline_outgoing.py`: microphone to virtual mic flow
  - `pipeline_incoming.py`: loopback to headphones flow
  - `orchestrator.py`: application-level lifecycle
- `realtime_translator/runtime`
  - `subtitles.py`: subtitle fan-out sinks
  - `websocket_server.py`: optional websocket subtitle stream
- `realtime_translator/config`
  - `models.py`: strongly typed configuration schema
  - `manager.py`: YAML loader
- `realtime_translator/utils`
  - `vad.py`: WebRTC VAD plus energy-based fallback
  - `logger.py`: console/file logging

### Audio pipeline

Outgoing:

```text
Microphone Input
  -> Voice Activity Detection
  -> Chunking / Speech Segmentation
  -> Speech-to-Text (RU)
  -> Translation (RU -> EN)
  -> Text-to-Speech (EN)
  -> Voice Conversion (optional)
  -> Virtual Microphone Output
```

Incoming:

```text
System Audio Loopback
  -> Voice Activity Detection
  -> Chunking / Speech Segmentation
  -> Speech-to-Text (EN)
  -> Translation (EN -> RU)
  -> Text-to-Speech (RU)
  -> Headphones Output
```

### Runtime behavior

- Each direction uses isolated queues for frames, speech segments, transcripts, translations and synthesized audio.
- STT, translation, TTS and voice conversion are provider interfaces, so models can be swapped from config.
- VAD prevents expensive transcription of silence.
- Subtitle events can be printed to console or streamed over websocket.
- Audio routing is device-based, so the browser can use VB-Cable as its microphone.

## Recommended technology choices

### Best low-cost local-first stack

- STT: `faster-whisper`
- Translation: `transformers` + MarianMT models
- TTS: `pyttsx3` for simplest offline baseline, or `Coqui TTS` / `XTTS` for higher quality
- Voice conversion: `RVC` when you are ready to integrate a trained voice model
- Audio routing on Windows: `VB-Cable`

### Higher quality cloud-assisted stack

- STT: Azure Speech or Whisper API
- Translation: OpenAI API or DeepL API
- TTS: Azure TTS or ElevenLabs
- Voice cloning: XTTS or ElevenLabs voice cloning

Local-first is the correct starting point for cost minimization. Cloud providers should be an optional upgrade path, not the baseline.

## Project structure

```text
.
├── config/
│   └── config.yaml
├── realtime_translator/
│   ├── audio/
│   ├── config/
│   ├── models/
│   ├── pipeline/
│   ├── runtime/
│   ├── stt/
│   ├── translation/
│   ├── tts/
│   ├── utils/
│   └── voice/
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt
```

## Installation

### 1. Python

Use Python `3.11.x`.

### 2. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install virtual audio cable on Windows

Recommended:

- VB-Cable: https://vb-audio.com/Cable/

After installation you should see devices similar to:

- `CABLE Input (VB-Audio Virtual Cable)`
- `CABLE Output (VB-Audio Virtual Cable)`

### 4. Inspect device names

```powershell
python main.py --list-devices
```

Copy the exact device names into `config/config.yaml`.

## Windows call setup

1. Set your normal hardware microphone as `audio.microphone.device_name`, or leave `null` for default input.
2. Set `audio.virtual_microphone_output_device` to `CABLE Input (VB-Audio Virtual Cable)`.
3. Set `audio.headphones_output_device` to your real headphones or speakers.
4. In Zoom/Meet/Teams/Discord select `CABLE Output (VB-Audio Virtual Cable)` as the browser/app microphone.
5. Keep your speakers/headphones as the playback device for the call app.
6. Start the translator. Your microphone speech will be synthesized to the virtual cable, and the remote speaker audio will be translated back to your headphones.

## Configuration

Main config file: `config/config.yaml`

Key options:

- `stt.provider`
  - `faster-whisper`
- `translation.provider`
  - `transformers-marian`
  - `openai`
- `tts.provider`
  - `pyttsx3`
  - `coqui`
  - `xtts`
- `voice_conversion.provider`
  - `none`
  - `rvc`

Model switching example:

```yaml
translation:
  provider: "transformers-marian"
  options:
    models:
      ru-en: "Helsinki-NLP/opus-mt-ru-en"
      en-ru: "Helsinki-NLP/opus-mt-en-ru"
```

## Running

Validate config:

```powershell
python main.py --validate-config
```

Run the translator:

```powershell
python main.py --config config/config.yaml
```

## Websocket subtitles

If you want a lightweight subtitle feed for a future GUI or OBS source:

```yaml
runtime:
  websocket_enabled: true
  websocket_host: "127.0.0.1"
  websocket_port: 8765
```

Then connect to:

```text
ws://127.0.0.1:8765
```

Each message is a JSON subtitle event with original and translated text.

## Docker

`Dockerfile` is included for dependency packaging and config validation, but native Windows audio routing and VB-Cable usage should run on the host OS, not inside a container.

```powershell
docker build -t realtime-translator .
docker run --rm realtime-translator
```

## Roadmap

- Replace `pyttsx3` with `XTTS` or `Coqui` for better quality and voice cloning
- Add RVC integration with actual inference service
- Add streaming Whisper partial transcripts
- Add WebRTC bridge instead of audio-device-based routing
- Add GUI for subtitle overlay, device selection and live metrics
- Add noise suppression, echo cancellation and speaker diarization

## Limitations

- Real sub-second latency depends on GPU, model size and audio device stability.
- `pyttsx3` is the simplest local TTS baseline but not the highest quality.
- `system_audio_capture.py` is designed around Windows speaker loopback.
- Voice cloning is prepared as an extension point, not fully wired to a trained model in this repository.
