from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

from realtime_translator.config.models import AppConfig

T = TypeVar("T")


def _convert_value(value: Any, type_hint: Any) -> Any:
    origin = get_origin(type_hint)
    if is_dataclass(type_hint):
        return _build_dataclass(type_hint, value or {})
    if origin is None:
        return value
    if origin in (dict,):
        return value or {}
    if origin in (list,):
        inner_type = get_args(type_hint)[0]
        return [_convert_value(item, inner_type) for item in (value or [])]
    if origin in (tuple,):
        inner_types = get_args(type_hint)
        return tuple(
            _convert_value(item, inner_types[min(index, len(inner_types) - 1)])
            for index, item in enumerate(value or [])
        )
    if origin is not None and type(None) in get_args(type_hint):
        non_none = [arg for arg in get_args(type_hint) if arg is not type(None)]
        return None if value is None else _convert_value(value, non_none[0])
    return value


def _build_dataclass(cls: type[T], payload: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        kwargs[field_info.name] = _convert_value(
            payload[field_info.name],
            type_hints.get(field_info.name, field_info.type),
        )
    return cls(**kwargs)


class ConfigManager:
    @staticmethod
    def load(path: str | Path) -> AppConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return _build_dataclass(AppConfig, data)
