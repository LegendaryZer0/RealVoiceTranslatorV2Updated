from __future__ import annotations

from typing import Any


def _normalize_device_ref(device: str | int | None) -> str | int | None:
    if isinstance(device, str):
        stripped = device.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return device


def resolve_sounddevice_device(
    device: str | int | None,
    *,
    kind: str,
    preferred_hostapis: tuple[str, ...] = ("Windows WASAPI", "WASAPI"),
) -> str | int | None:
    device = _normalize_device_ref(device)
    if device is None or isinstance(device, int):
        return device

    import sounddevice as sd  # type: ignore

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    capability_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches: list[dict[str, Any]] = []
    for entry in devices:
        if entry[capability_key] <= 0:
            continue
        name = str(entry["name"])
        if device == name or device in name:
            entry = dict(entry)
            entry["hostapi_name"] = hostapis[entry["hostapi"]]["name"]
            matches.append(entry)

    if not matches:
        raise ValueError(f"No {kind} device found for '{device}'")
    if len(matches) == 1:
        return int(matches[0]["index"])

    for hostapi_name in preferred_hostapis:
        preferred = [item for item in matches if item["hostapi_name"] == hostapi_name]
        if len(preferred) == 1:
            return int(preferred[0]["index"])

    exact = [item for item in matches if str(item["name"]) == device]
    if len(exact) == 1:
        return int(exact[0]["index"])

    details = ", ".join(
        f"[{item['index']}] {item['name']} ({item['hostapi_name']})" for item in matches
    )
    raise ValueError(f"Multiple {kind} devices found for '{device}': {details}")


def resolve_soundcard_speaker(device: str | None):
    import soundcard as sc  # type: ignore

    if not device:
        return sc.default_speaker()

    speakers = list(sc.all_speakers())
    exact = [speaker for speaker in speakers if device == speaker.name]
    if len(exact) == 1:
        return exact[0]

    partial = [speaker for speaker in speakers if device in speaker.name]
    if len(partial) == 1:
        return partial[0]

    available = ", ".join(speaker.name for speaker in speakers)
    raise ValueError(f"No unique speaker found for '{device}'. Available speakers: {available}")


def resolve_soundcard_microphone(device: str | None, *, include_loopback: bool = False):
    import soundcard as sc  # type: ignore

    microphones = list(sc.all_microphones(include_loopback=include_loopback))
    if not device:
        if include_loopback:
            default_speaker = sc.default_speaker()
            exact_loopback = [mic for mic in microphones if mic.isloopback and mic.name == default_speaker.name]
            if len(exact_loopback) == 1:
                return exact_loopback[0]
        return sc.default_microphone()

    exact = [mic for mic in microphones if device == mic.name]
    if len(exact) == 1:
        return exact[0]

    partial = [mic for mic in microphones if device in mic.name]
    if len(partial) == 1:
        return partial[0]

    available = ", ".join(mic.name for mic in microphones)
    raise ValueError(f"No unique microphone found for '{device}'. Available microphones: {available}")
