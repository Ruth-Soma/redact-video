"""Core models shared by interfaces and processing layers."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SensitiveType(str, Enum):
    CARD_NUMBER = "card_number"
    CARD_EXPIRY = "card_expiry"
    CARD_CVV = "card_cvv"
    SAVED_CARD_NUMBER = "saved_card_number"
    SAVED_CARD_EXPIRY = "saved_card_expiry"


class JobStatus(str, Enum):
    QUEUED = "queued"
    INSPECTING = "inspecting"
    DETECTING = "detecting"
    TRACKING = "tracking"
    RENDERING = "rendering"
    VERIFYING = "verifying"
    PATCHING = "patching"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class ProcessingTimings(BaseModel):
    """Sanitized wall-clock measurements for one successful pipeline run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    inspection_ms: int = Field(ge=0)
    setup_ms: int = Field(ge=0)
    source_analysis_ms: int = Field(ge=0)
    initial_render_ms: int = Field(ge=0)
    initial_validation_ms: int = Field(ge=0)
    verification_passes_ms: tuple[Annotated[int, Field(ge=0)], ...] = ()
    repair_passes_ms: tuple[Annotated[int, Field(ge=0)], ...] = ()
    publication_ms: int = Field(ge=0)
    total_ms: int = Field(ge=0)


class TrackState(str, Enum):
    DETECTED = "detected"
    TRACKING = "tracking"
    STICKY = "sticky"
    LOST = "lost"
    ENDED = "ended"


class BoundingBox(BaseModel):
    """Pixel-space rectangle with a non-empty area."""

    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def expanded(self, x_padding: int, y_padding: int) -> BoundingBox:
        """Return a padded rectangle without allowing negative coordinates."""

        left = max(0, self.x - x_padding)
        top = max(0, self.y - y_padding)
        return BoundingBox(
            x=left,
            y=top,
            width=self.right + x_padding - left,
            height=self.bottom + y_padding - top,
        )

    def clipped(self, frame_width: int, frame_height: int) -> BoundingBox | None:
        """Clip to a frame, returning ``None`` when there is no intersection."""

        left = min(max(self.x, 0), frame_width)
        top = min(max(self.y, 0), frame_height)
        right = min(max(self.right, 0), frame_width)
        bottom = min(max(self.bottom, 0), frame_height)
        if right <= left or bottom <= top:
            return None
        return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


class OcrToken(BaseModel):
    """Transient OCR text and geometry.

    Text is hidden from representations to reduce accidental disclosure through
    debugging or exception logging. OCR tokens must never be persisted.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    text: str = Field(min_length=1, max_length=4096, repr=False, exclude=True)
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class SensitiveDetection(BaseModel):
    """Sanitized sensitive-region evidence with no recognized value field."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    type: SensitiveType
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    source: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")


class SensitiveTrack(BaseModel):
    """Sanitized temporal history for one sensitive screen region."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")
    type: SensitiveType
    start_frame: int = Field(ge=0)
    end_frame: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    regions: dict[int, BoundingBox]
    state: TrackState

    @model_validator(mode="after")
    def validate_timeline(self) -> Self:
        if not self.regions:
            raise ValueError("A sensitive track must contain at least one region")
        if min(self.regions) < self.start_frame:
            raise ValueError("Track regions cannot precede the start frame")
        if self.end_frame is not None:
            if self.end_frame < self.start_frame:
                raise ValueError("Track end cannot precede its start")
            if max(self.regions) > self.end_frame:
                raise ValueError("Track regions cannot follow the end frame")
        return self


class TrackedRegion(BaseModel):
    """Maskable region for one frame, with no recognized-value payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")
    type: SensitiveType
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    state: TrackState


class TrackingFrameResult(BaseModel):
    """Deterministic output and scan decision for a processed frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    regions: tuple[TrackedRegion, ...] = ()
    tracks: tuple[SensitiveTrack, ...] = ()
    scene_changed: bool = False
    ocr_required: bool = False
    ocr_performed: bool = False
