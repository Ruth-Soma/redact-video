"""Strict, sanitized YAML loading for engine configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yaml.events import AliasEvent

from redact_video.config import (
    AppConfig,
    BlurConfig,
    DetectionConfig,
    MediaLimits,
    OcrConfig,
    OutputConfig,
    ProcessingMode,
    RedactionPolicy,
    TrackingConfig,
    VerificationConfig,
)
from redact_video.exceptions import ConfigurationError

MAX_CONFIG_BYTES = 1024 * 1024
MAX_CONFIG_ALIASES = 32
MAX_CONFIG_NODES = 10_000


class _LimitedSafeLoader(yaml.SafeLoader):
    """SafeLoader with bounds against alias and node expansion abuse."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._aliases = 0
        self._nodes = 0

    def compose_node(self, parent: object, index: object) -> yaml.Node:
        if self.check_event(AliasEvent):
            self._aliases += 1
            if self._aliases > MAX_CONFIG_ALIASES:
                raise yaml.YAMLError("configuration alias limit exceeded")
        node = super().compose_node(parent, index)
        self._nodes += 1
        if self._nodes > MAX_CONFIG_NODES:
            raise yaml.YAMLError("configuration node limit exceeded")
        return node


class RedactionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: RedactionPolicy = RedactionPolicy.CARD_DETAILS
    blur: BlurConfig = Field(default_factory=BlurConfig)


class ConfigurationDocument(BaseModel):
    """External YAML shape from section 40 of the engineering specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    redaction: RedactionConfig = Field(default_factory=RedactionConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    media_limits: MediaLimits = Field(default_factory=MediaLimits)

    def to_app_config(self) -> AppConfig:
        defaults = AppConfig()
        return defaults.model_copy(
            update={
                "policy": self.redaction.policy,
                "blur": self.redaction.blur,
                "detection": self.detection,
                "tracking": self.tracking,
                "verification": self.verification,
                "output": self.output,
                "ocr": self.ocr,
                "media_limits": self.media_limits,
            }
        )


def _validation_locations(exc: ValidationError) -> str:
    locations: list[str] = []
    for error in exc.errors(
        include_url=False, include_context=False, include_input=False
    ):
        location = ".".join(str(part) for part in error["loc"]) or "root"
        locations.append(location)
        if len(locations) == 5:
            break
    suffix = "" if len(exc.errors()) <= 5 else ", ..."
    return ", ".join(locations) + suffix


def load_app_config(path: Path) -> AppConfig:
    """Load one YAML document without exposing its submitted values in errors."""

    source = Path(path)
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError("Configuration must use a .yaml or .yml extension")
    if not source.is_file():
        raise ConfigurationError("Configuration path is not a regular file")
    try:
        with source.open("rb") as handle:
            payload = handle.read(MAX_CONFIG_BYTES + 1)
        if len(payload) > MAX_CONFIG_BYTES:
            raise ConfigurationError("Configuration file exceeds the 1 MiB limit")
        text = payload.decode("utf-8")
        raw: Any = yaml.load(text, Loader=_LimitedSafeLoader)
    except ConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ConfigurationError("Configuration YAML could not be parsed") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    try:
        document = ConfigurationDocument.model_validate(raw)
    except ValidationError as exc:
        locations = _validation_locations(exc)
        raise ConfigurationError(
            f"Configuration contains invalid fields at: {locations}"
        ) from None
    return document.to_app_config()


def apply_processing_mode(
    config: AppConfig,
    mode: ProcessingMode,
) -> AppConfig:
    """Apply the shared CLI/API/acceptance scan-rate profile."""

    if mode is ProcessingMode.SAFE:
        detection = config.detection.model_copy(
            update={"scan_fps": config.detection.safe_scan_fps}
        )
        return config.model_copy(update={"detection": detection})
    if mode is ProcessingMode.FAST:
        detection = config.detection.model_copy(
            update={
                "scan_fps": max(1.0, config.detection.scan_fps / 2),
                "safe_scan_fps": max(1.0, config.detection.safe_scan_fps / 2),
            }
        )
        return config.model_copy(update={"detection": detection})
    return config
