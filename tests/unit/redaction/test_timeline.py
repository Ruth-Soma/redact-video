from __future__ import annotations

import pytest
from pydantic import ValidationError

from redact_video.models import (
    BoundingBox,
    SensitiveDetection,
    SensitiveType,
    TrackedRegion,
    TrackingFrameResult,
    TrackState,
)
from redact_video.redaction.timeline import (
    DetectionTimeline,
    MaskTimelineBuilder,
)


def _detection(frame_index: int, x: int) -> SensitiveDetection:
    return SensitiveDetection(
        type=SensitiveType.CARD_NUMBER,
        frame_index=frame_index,
        timestamp_ms=frame_index * 40,
        bbox=BoundingBox(x=x, y=40, width=80, height=24),
        confidence=0.9,
        source="synthetic",
    )


def test_detection_timeline_groups_and_sorts_sanitized_detections() -> None:
    timeline = DetectionTimeline.from_detections(
        [_detection(2, 80), _detection(1, 90), _detection(2, 20)]
    )

    assert list(timeline.frames) == [1, 2]
    assert [item.bbox.x for item in timeline.frames[2]] == [20, 80]
    assert "value" not in str(timeline.model_dump()).lower()


def test_detection_timeline_rejects_wrong_frame_membership() -> None:
    with pytest.raises(ValidationError, match="match"):
        DetectionTimeline(frames={3: (_detection(2, 20),)})


def test_mask_timeline_builder_keeps_only_masked_frames_in_stable_order() -> None:
    builder = MaskTimelineBuilder()
    builder.add(
        TrackingFrameResult(
            frame_index=0,
            timestamp_ms=0,
            regions=(
                TrackedRegion(
                    track_id="track-000002",
                    type=SensitiveType.CARD_CVV,
                    bbox=BoundingBox(x=180, y=80, width=40, height=24),
                    confidence=0.8,
                    state=TrackState.DETECTED,
                ),
                TrackedRegion(
                    track_id="track-000001",
                    type=SensitiveType.CARD_NUMBER,
                    bbox=BoundingBox(x=30, y=40, width=120, height=24),
                    confidence=0.9,
                    state=TrackState.TRACKING,
                ),
            ),
        )
    )
    builder.add(TrackingFrameResult(frame_index=1, timestamp_ms=40))

    timeline = builder.build()

    assert list(timeline.frames) == [0]
    assert [item.track_id for item in timeline.regions_for(0)] == [
        "track-000001",
        "track-000002",
    ]
    assert timeline.boxes_for(1) == []
    serialized = str(timeline.model_dump()).lower()
    assert "value" not in serialized
    assert "text" not in serialized


def test_mask_timeline_builder_rejects_duplicate_or_reordered_frames() -> None:
    builder = MaskTimelineBuilder()
    builder.add(TrackingFrameResult(frame_index=2, timestamp_ms=80))

    with pytest.raises(ValueError, match="increasing"):
        builder.add(TrackingFrameResult(frame_index=2, timestamp_ms=80))


def test_mask_timeline_builder_backfills_new_tracks_across_seen_frames() -> None:
    builder = MaskTimelineBuilder(detection_pre_roll_frames=2)
    builder.add(TrackingFrameResult(frame_index=0, timestamp_ms=0))
    builder.add(TrackingFrameResult(frame_index=1, timestamp_ms=40))
    builder.add(TrackingFrameResult(frame_index=3, timestamp_ms=120))
    builder.add(
        TrackingFrameResult(
            frame_index=4,
            timestamp_ms=160,
            regions=(
                TrackedRegion(
                    track_id="track-000001",
                    type=SensitiveType.CARD_NUMBER,
                    bbox=BoundingBox(x=30, y=40, width=120, height=24),
                    confidence=0.9,
                    state=TrackState.TRACKING,
                ),
            ),
        )
    )

    timeline = builder.build()

    assert list(timeline.frames) == [3, 4]
    assert timeline.regions_for(3)[0].track_id == "track-000001"
    assert timeline.regions_for(4)[0].track_id == "track-000001"


def test_mask_timeline_builder_backfills_each_track_only_on_first_appearance() -> None:
    builder = MaskTimelineBuilder(detection_pre_roll_frames=2)
    region = TrackedRegion(
        track_id="track-000001",
        type=SensitiveType.CARD_NUMBER,
        bbox=BoundingBox(x=30, y=40, width=120, height=24),
        confidence=0.9,
        state=TrackState.DETECTED,
    )
    builder.add(TrackingFrameResult(frame_index=0, timestamp_ms=0, regions=(region,)))
    builder.add(TrackingFrameResult(frame_index=1, timestamp_ms=40))
    builder.add(TrackingFrameResult(frame_index=2, timestamp_ms=80, regions=(region,)))

    timeline = builder.build()

    assert timeline.regions_for(1) == ()


def test_mask_timeline_builder_rejects_negative_detection_pre_roll() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MaskTimelineBuilder(detection_pre_roll_frames=-1)
