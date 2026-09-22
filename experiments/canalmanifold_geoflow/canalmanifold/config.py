"""Small YAML configuration helpers.

Paths remain external to the code so the same package works on Google Drive,
Windows, macOS and a rented GPU pod.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    config["_config_path"] = str(path)
    return config


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value


def expand_path(value: str | Path, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()
