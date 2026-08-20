from __future__ import annotations

from collections.abc import Callable
from threading import Barrier

import cv2
import numpy as np
import pytest

from redact_video.config import TrackingConfig
from redact_video.models import (
    BoundingBox,
    SensitiveDetection,
    SensitiveType,
    TrackState,
)
from redact_video.tracking.optical_flow import MotionEstimate
from redact_video.tracking.tracker import RegionTracker


class _FailedOpticalFlow:
    def estimate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    estimate_gray = estimate


class _FailedTemplateMatcher:
    def match(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    match_gray = match


class _BarrierOpticalFlow:
    def __init__(self, parties: int) -> None:
        self.barrier = Barrier(parties)
        self.calls = 0

    def estimate_gray(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.barrier.wait(timeout=2)
        return MotionEstimate(
            dx=0.0,
            dy=0.0,
            scale=1.0,
            confidence=1.0,
            point_count=8,
        )


def _failure_tracker(config: TrackingConfig | None = None) -> RegionTracker:
    return RegionTracker(
        config,
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
        template_matcher=_FailedTemplateMatcher(),  # type: ignore[arg-type]
    )


def test_region_tracker_builds_only_one_full_resolution_gray_frame_per_input(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_frame, first_box = tracking_frame_factory()
    second_frame, _ = tracking_frame_factory(dy=8)
    converted_shapes: list[tuple[int, int]] = []
    original = cv2.cvtColor

    def counted(*args, **kwargs):  # type: ignore[no-untyped-def]
        converted_shapes.append(args[0].shape[:2])
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "cvtColor", counted)
    tracker = RegionTracker()
    tracker.update(
        first_frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(first_box)],
    )
    tracker.update(second_frame, frame_index=1, timestamp_ms=33)

    full_resolution_shape = first_frame.shape[:2]
    assert converted_shapes.count(full_resolution_shape) == 2
    assert converted_shapes.count((64, 64)) == 2


def test_region_tracker_follows_slow_vertical_motion_with_optical_flow(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    first_frame, first_box = tracking_frame_factory()
    second_frame, second_box = tracking_frame_factory(dy=8)
    tracker = RegionTracker()
    initial = tracker.update(
        first_frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(first_box)],
    )

    result = tracker.update(second_frame, frame_index=1, timestamp_ms=33)

    assert initial.regions[0].state is TrackState.DETECTED
    assert result.regions[0].state is TrackState.TRACKING
    assert abs(result.regions[0].bbox.y - second_box.y) <= 2
    assert result.ocr_required is False


def test_multiple_active_regions_estimate_optical_flow_concurrently(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    frame, box = tracking_frame_factory()
    second_box = BoundingBox(x=245, y=85, width=60, height=42)
    optical_flow = _BarrierOpticalFlow(parties=2)
    tracker = RegionTracker(
        TrackingConfig(
            optical_flow_workers=2,
            parallel_flow_min_frame_pixels=1,
        ),
        optical_flow=optical_flow,  # type: ignore[arg-type]
    )
    tracker.update(
        frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[
            detection_factory(box),
            detection_factory(second_box, sensitive_type=SensitiveType.CARD_CVV),
        ],
    )

    try:
        result = tracker.update(frame, frame_index=1, timestamp_ms=33)
    finally:
        tracker.close()
        tracker.close()

    assert optical_flow.calls == 2
    assert len(result.regions) == 2
    assert all(region.state is TrackState.TRACKING for region in result.regions)


def test_template_fallback_tracks_a_fast_scroll_when_optical_flow_fails(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    first_frame, first_box = tracking_frame_factory()
    second_frame, second_box = tracking_frame_factory(dy=55)
    tracker = RegionTracker(
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
    )
    tracker.update(
        first_frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(first_box)],
    )

    result = tracker.update(second_frame, frame_index=1, timestamp_ms=33)

    assert result.regions[0].state is TrackState.TRACKING
    assert abs(result.regions[0].bbox.y - second_box.y) <= 1
    assert result.ocr_required is False


def test_fast_scroll_outside_bounded_fallback_keeps_a_sticky_mask(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    first_frame, first_box = tracking_frame_factory()
    second_frame, _ = tracking_frame_factory(dy=80)
    config = TrackingConfig(
        template_search_radius_ratio=0.20,
        minimum_template_search_px=10,
    )
    tracker = RegionTracker(
        config,
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
    )
    tracker.update(
        first_frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(first_box)],
    )

    result = tracker.update(second_frame, frame_index=1, timestamp_ms=33)

    assert result.regions[0].state is TrackState.STICKY
    assert result.ocr_required is True


def test_tracking_loss_enters_sticky_and_predicts_recent_velocity(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    first_frame, first_box = tracking_frame_factory()
    second_frame, second_box = tracking_frame_factory(dy=10)
    blank = np.zeros_like(first_frame)
    tracker = _failure_tracker(TrackingConfig(velocity_smoothing=1.0))
    tracker.update(
        first_frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(first_box)],
    )
    tracker.update(
        second_frame,
        frame_index=1,
        timestamp_ms=40,
        detections=[detection_factory(second_box, frame_index=1, timestamp_ms=40)],
    )

    result = tracker.update(blank, frame_index=2, timestamp_ms=80)

    assert result.ocr_required is True
    assert result.regions[0].state is TrackState.STICKY
    assert result.regions[0].bbox.y == second_box.y + 10


def test_sticky_region_is_maskable_for_at_least_configured_duration(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    frame, box = tracking_frame_factory()
    tracker = _failure_tracker(TrackingConfig(sticky_duration_ms=350))
    tracker.update(
        frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(box)],
    )

    first_loss = tracker.update(frame, frame_index=1, timestamp_ms=40)
    boundary = tracker.update(frame, frame_index=2, timestamp_ms=390)
    expired = tracker.update(frame, frame_index=3, timestamp_ms=391)

    assert first_loss.regions[0].state is TrackState.STICKY
    assert boundary.regions[0].state is TrackState.STICKY
    assert expired.regions == ()
    assert expired.tracks[0].state is TrackState.LOST
    assert expired.tracks[0].end_frame == 2


def test_scene_change_requests_ocr_and_retains_sticky_region(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    previous, box = tracking_frame_factory()
    replacement = np.full_like(previous, 245)
    tracker = RegionTracker()
    tracker.update(
        previous,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(box)],
    )

    result = tracker.update(replacement, frame_index=1, timestamp_ms=33)

    assert result.scene_changed is True
    assert result.ocr_required is True
    assert result.regions[0].state is TrackState.STICKY


def test_same_frame_reacquisition_restores_tracking_state(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    previous, box = tracking_frame_factory()
    replacement = np.full_like(previous, 245)
    tracker = RegionTracker()
    tracker.update(
        previous,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(box)],
    )
    tracker.update(replacement, frame_index=1, timestamp_ms=33)

    result = tracker.reacquire([detection_factory(box, frame_index=1, timestamp_ms=33)])

    assert result.ocr_performed is True
    assert result.ocr_required is False
    assert result.regions[0].state is TrackState.TRACKING
    assert result.tracks[0].regions[1] == box


def test_velocity_that_moves_a_region_fully_offscreen_ends_the_track(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    frame, _ = tracking_frame_factory()
    first_box = BoundingBox(x=250, y=80, width=50, height=30)
    second_box = BoundingBox(x=320, y=80, width=40, height=30)
    tracker = _failure_tracker(
        TrackingConfig(
            velocity_smoothing=1.0,
            detection_match_distance_ratio=10.0,
        )
    )
    tracker.update(
        frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(first_box)],
    )
    tracker.update(
        frame,
        frame_index=1,
        timestamp_ms=40,
        detections=[detection_factory(second_box, frame_index=1, timestamp_ms=40)],
    )

    result = tracker.update(frame, frame_index=2, timestamp_ms=80)

    assert result.regions == ()
    assert result.tracks[0].state is TrackState.ENDED
    assert result.tracks[0].end_frame == 1
    assert result.ocr_required is True


def test_track_output_never_contains_recognized_values(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    frame, box = tracking_frame_factory()
    result = RegionTracker().update(
        frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[detection_factory(box)],
    )

    serialized = result.model_dump()
    assert "value" not in str(serialized).lower()
    assert "text" not in str(serialized).lower()


def test_tracker_rejects_out_of_order_frames_and_mismatched_detections(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    frame, box = tracking_frame_factory()
    tracker = RegionTracker()
    tracker.update(frame, frame_index=1, timestamp_ms=40, detections=[])

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update(frame, frame_index=1, timestamp_ms=50, detections=[])
    with pytest.raises(ValueError, match="belong"):
        tracker.update(
            frame,
            frame_index=2,
            timestamp_ms=80,
            detections=[detection_factory(box)],
        )


def test_reacquisition_requires_a_current_frame() -> None:
    with pytest.raises(RuntimeError, match="updated"):
        RegionTracker().reacquire([])


def test_multiple_sensitive_types_keep_distinct_deterministic_tracks(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
    detection_factory: Callable[..., SensitiveDetection],
) -> None:
    frame, box = tracking_frame_factory()
    cvv_box = BoundingBox(x=250, y=85, width=45, height=42)
    result = RegionTracker().update(
        frame,
        frame_index=0,
        timestamp_ms=0,
        detections=[
            detection_factory(box),
            detection_factory(cvv_box, sensitive_type=SensitiveType.CARD_CVV),
        ],
    )

    assert [track.id for track in result.tracks] == [
        "track-000001",
        "track-000002",
    ]
    assert {region.type for region in result.regions} == {
        SensitiveType.CARD_NUMBER,
        SensitiveType.CARD_CVV,
    }
