"""Sanitized verification and repair result models."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redact_video.models import BoundingBox, SensitiveDetection, SensitiveType


class VerificationLeak(BaseModel):
    """Value-free evidence that sensitive content remains readable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: SensitiveType
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    source: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")

    @classmethod
    def from_detection(cls, detection: SensitiveDetection) -> VerificationLeak:
        return cls(
            type=detection.type,
            frame_index=detection.frame_index,
            timestamp_ms=detection.timestamp_ms,
            bbox=detection.bbox,
            confidence=detection.confidence,
            source=detection.source,
        )


class RepairWindow(BaseModel):
    """Sanitized time interval selected for automatic repair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    types: tuple[SensitiveType, ...]

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_ms < self.start_ms:
            raise ValueError("Repair window end cannot precede its start")
        if not self.types:
            raise ValueError("Repair window must contain at least one sensitive type")
        return self


class VerificationPassResult(BaseModel):
    """Internal result of one complete rendered-output scan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pass_number: int = Field(ge=1)
    frames_scanned: int = Field(gt=0)
    leaks: tuple[VerificationLeak, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.leaks


class VerificationSummary(BaseModel):
    """Public, sanitized verification statistics for a completed job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passes: int = Field(ge=0)
    repairs: int = Field(ge=0)
    remaining_leaks: int = Field(ge=0)
    potential_issues: tuple[RepairWindow, ...] = ()
