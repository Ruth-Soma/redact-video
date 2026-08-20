"""Deterministic, value-free detection and mask timelines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redact_video.models import (
    BoundingBox,
    SensitiveDetection,
    SensitiveType,
    TrackingFrameResult,
)


class DetectionTimeline(BaseModel):
    """Sanitized detections grouped deterministically by frame index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames: dict[int, tuple[SensitiveDetection, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frames(self) -> Self:
        if any(frame_index < 0 for frame_index in self.frames):
            raise ValueError("Timeline frame indices must be non-negative")
        if any(
            detection.frame_index != frame_index
            for frame_index, detections in self.frames.items()
            for detection in detections
        ):
            raise ValueError("Every detection must match its timeline frame")
        return self

    @classmethod
    def from_detections(
        cls,
        detections: Iterable[SensitiveDetection],
    ) -> DetectionTimeline:
        grouped: dict[int, list[SensitiveDetection]] = {}
        for detection in detections:
            grouped.setdefault(detection.frame_index, []).append(detection)
        return cls(
            frames={
                frame_index: tuple(
                    sorted(
                        grouped[frame_index],
                        key=lambda item: (
                            item.bbox.y,
                            item.bbox.x,
                            item.type.value,
                            -item.confidence,
                        ),
                    )
                )
                for frame_index in sorted(grouped)
            }
        )


class MaskRegion(BaseModel):
    """One value-free region to privacy-mask on one frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.-]+$")
    type: SensitiveType
    bbox: BoundingBox


class MaskTimeline(BaseModel):
    """In-memory frame-to-region map consumed by the renderer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frames: dict[int, tuple[MaskRegion, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frame_indices(self) -> Self:
        if any(frame_index < 0 for frame_index in self.frames):
            raise ValueError("Timeline frame indices must be non-negative")
        return self

    def regions_for(self, frame_index: int) -> tuple[MaskRegion, ...]:
        return self.frames.get(frame_index, ())

    def boxes_for(self, frame_index: int) -> list[BoundingBox]:
        return [region.bbox for region in self.regions_for(frame_index)]


class MaskTimelineBuilder:
    """Incrementally collect ordered tracking results without frame content."""

    def __init__(self, *, detection_pre_roll_frames: int = 0) -> None:
        if detection_pre_roll_frames < 0:
            raise ValueError("Detection pre-roll frames must be non-negative")
        self._detection_pre_roll_frames = detection_pre_roll_frames
        self._frames: dict[int, tuple[MaskRegion, ...]] = {}
        self._seen_frame_indices: list[int] = []
        self._known_track_ids: set[str] = set()
        self._last_frame_index: int | None = None

    @staticmethod
    def _sort_regions(regions: Iterable[MaskRegion]) -> tuple[MaskRegion, ...]:
        return tuple(
            sorted(
                regions,
                key=lambda item: (
                    item.bbox.y,
                    item.bbox.x,
                    item.type.value,
                    item.track_id,
                ),
            )
        )

    def add(self, result: TrackingFrameResult) -> tuple[MaskRegion, ...]:
        if (
            self._last_frame_index is not None
            and result.frame_index <= self._last_frame_index
        ):
            raise ValueError("Tracking results must be added in increasing frame order")
        regions = self._sort_regions(
            MaskRegion(
                track_id=region.track_id,
                type=region.type,
                bbox=region.bbox,
            )
            for region in result.regions
        )
        new_regions = tuple(
            region for region in regions if region.track_id not in self._known_track_ids
        )
        if new_regions and self._detection_pre_roll_frames:
            first_pre_roll_frame = max(
                0,
                result.frame_index - self._detection_pre_roll_frames,
            )
            for frame_index in self._seen_frame_indices:
                if frame_index < first_pre_roll_frame:
                    continue
                existing = self._frames.get(frame_index, ())
                existing_track_ids = {region.track_id for region in existing}
                self._frames[frame_index] = self._sort_regions(
                    (
                        *existing,
                        *(
                            region
                            for region in new_regions
                            if region.track_id not in existing_track_ids
                        ),
                    )
                )
        if regions:
            self._frames[result.frame_index] = regions
        self._known_track_ids.update(region.track_id for region in regions)
        self._seen_frame_indices.append(result.frame_index)
        self._last_frame_index = result.frame_index
        return regions

    def build(self) -> MaskTimeline:
        return MaskTimeline(
            frames={
                frame_index: self._frames[frame_index]
                for frame_index in sorted(self._frames)
            }
        )
