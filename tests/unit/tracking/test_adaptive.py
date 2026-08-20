from __future__ import annotations

from collections.abc import Callable
from threading import Barrier

import numpy as np

from redact_video.config import TrackingConfig
from redact_video.exceptions import OcrProcessingError
from redact_video.models import BoundingBox, OcrToken, TrackState
from redact_video.tracking.optical_flow import MotionEstimate
from redact_video.tracking.tracker import AdaptiveTrackingEngine, RegionTracker


class _MockOcrEngine:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        self.calls += 1
        return [
            OcrToken(
                text="Card Number",
                confidence=0.99,
                bbox=BoundingBox(x=40, y=40, width=145, height=24),
            )
        ]


class _OneShotOcrEngine(_MockOcrEngine):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        tokens = super().scan(frame)
        return tokens if self.calls == 1 else []


class _EmptyOcrEngine(_MockOcrEngine):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        super().scan(frame)
        return []


class _FailingOcrEngine(_MockOcrEngine):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        super().scan(frame)
        raise OcrProcessingError("native OCR failed")


class _FailOnSecondOcrEngine(_EmptyOcrEngine):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        if self.calls == 1:
            self.calls += 1
            raise OcrProcessingError("native OCR failed")
        return super().scan(frame)


class _BarrierOcrEngine(_EmptyOcrEngine):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        self.barrier.wait(timeout=2)
        return super().scan(frame)


class _SecondCallBarrierOcrEngine(_MockOcrEngine):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        if self.calls == 1:
            self.barrier.wait(timeout=2)
        return super().scan(frame)


class _BarrierOpticalFlow:
    def __init__(self, barrier: Barrier) -> None:
        self.barrier = barrier
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


class _FailedOpticalFlow:
    def estimate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    estimate_gray = estimate


class _FailedTemplateMatcher:
    def match(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    match_gray = match


def test_adaptive_engine_scans_periodically_not_on_every_frame(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    ocr = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(ocr)

    first = engine.process_frame(frame, frame_index=0, timestamp_ms=0)
    second = engine.process_frame(frame, frame_index=1, timestamp_ms=40)
    third = engine.process_frame(frame, frame_index=2, timestamp_ms=130)

    assert first.ocr_performed is True
    assert second.ocr_performed is False
    assert third.ocr_performed is True
    assert ocr.calls == 2


def test_scheduled_source_ocr_overlaps_ordered_optical_flow(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    barrier = Barrier(2)
    ocr = _SecondCallBarrierOcrEngine(barrier)
    optical_flow = _BarrierOpticalFlow(barrier)
    tracking_config = TrackingConfig(optical_flow_workers=1)
    tracker = RegionTracker(
        tracking_config,
        optical_flow=optical_flow,  # type: ignore[arg-type]
    )
    engine = AdaptiveTrackingEngine(
        ocr,
        tracker=tracker,
        tracking_config=tracking_config,
    )

    try:
        first = engine.process_frame(frame, frame_index=0, timestamp_ms=0)
        second = engine.process_frame(frame, frame_index=1, timestamp_ms=130)
    finally:
        engine.close()

    assert first.regions
    assert second.regions
    assert ocr.calls == 2
    assert optical_flow.calls == 1


def test_source_ocr_overlap_preserves_tracking_results(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frames = [tracking_frame_factory(dy=offset)[0] for offset in (0, 5, 10, 15)]
    timestamps = (0, 130, 260, 390)

    def process(*, overlap: bool):  # type: ignore[no-untyped-def]
        config = TrackingConfig(overlap_source_ocr_and_tracking=overlap)
        engine = AdaptiveTrackingEngine(
            _MockOcrEngine(),
            tracking_config=config,
        )
        try:
            return tuple(
                engine.process_frame(
                    frame,
                    frame_index=index,
                    timestamp_ms=timestamp_ms,
                )
                for index, (frame, timestamp_ms) in enumerate(
                    zip(frames, timestamps, strict=True)
                )
            )
        finally:
            engine.close()

    assert process(overlap=True) == process(overlap=False)


def test_native_source_scans_use_bounded_authoritative_audits(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        tracking_config=TrackingConfig(
            authoritative_audit_interval_ms=250,
            reuse_authoritative_frames=False,
        ),
    )

    results = [
        engine.process_frame(frame, frame_index=index, timestamp_ms=timestamp_ms)
        for index, timestamp_ms in enumerate((0, 130, 260, 390, 520))
    ]

    assert all(result.ocr_performed for result in results)
    assert accelerated.calls == 5
    assert authoritative.calls == 3


def test_default_native_source_audit_cadence_reduces_paddle_calls(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        tracking_config=TrackingConfig(
            reuse_authoritative_frames=False,
        ),
    )

    for index, timestamp_ms in enumerate((0, 130, 260, 390, 520)):
        engine.process_frame(frame, frame_index=index, timestamp_ms=timestamp_ms)

    assert accelerated.calls == 5
    assert authoritative.calls == 2


def test_unchanged_source_audits_reuse_authoritative_detections(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        tracking_config=TrackingConfig(
            authoritative_audit_interval_ms=250,
        ),
    )

    results = [
        engine.process_frame(frame, frame_index=index, timestamp_ms=timestamp_ms)
        for index, timestamp_ms in enumerate((0, 130, 260, 390, 520))
    ]

    assert accelerated.calls == 5
    assert authoritative.calls == 1
    assert all(result.regions for result in results)
    assert results[-1].regions[0].bbox == results[0].regions[0].bbox


def test_changed_source_frame_forces_a_fresh_authoritative_audit() -> None:
    original = np.zeros((100, 100, 3), dtype=np.uint8)
    changed = original.copy()
    changed[0, 0] = 7
    authoritative = _MockOcrEngine()
    accelerated = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        tracking_config=TrackingConfig(
            authoritative_audit_interval_ms=250,
        ),
    )

    engine.process_frame(original, frame_index=0, timestamp_ms=0)
    engine.process_frame(changed, frame_index=1, timestamp_ms=260)

    assert accelerated.calls == 2
    assert authoritative.calls == 2


def test_small_source_changes_accumulate_against_last_real_paddle_frame() -> None:
    original = np.zeros((1000, 1000, 3), dtype=np.uint8)
    first_change = original.copy()
    first_change.reshape(-1, 3)[:15] = 7
    accumulated_change = first_change.copy()
    accumulated_change.reshape(-1, 3)[15:30] = 7
    authoritative = _MockOcrEngine()
    accelerated = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        tracking_config=TrackingConfig(
            authoritative_audit_interval_ms=250,
        ),
    )

    engine._scan(original, frame_index=0, timestamp_ms=0)
    engine._scan(first_change, frame_index=1, timestamp_ms=260)
    engine._scan(accumulated_change, frame_index=2, timestamp_ms=520)

    assert accelerated.calls == 3
    assert authoritative.calls == 2


def test_reset_clears_authoritative_frame_reuse_cache(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=_MockOcrEngine(),
        tracking_config=TrackingConfig(authoritative_audit_interval_ms=250),
    )
    engine.process_frame(frame, frame_index=0, timestamp_ms=0)
    engine.process_frame(frame, frame_index=1, timestamp_ms=260)

    engine.reset()
    engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    assert authoritative.calls == 2


def test_source_audit_catches_a_native_ocr_miss(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _EmptyOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
    )

    result = engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    assert result.regions
    assert accelerated.calls == 1
    assert authoritative.calls == 1


def test_source_audit_runs_independent_ocr_engines_concurrently(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    barrier = Barrier(2)
    authoritative = _BarrierOcrEngine(barrier)
    accelerated = _BarrierOcrEngine(barrier)
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
    )

    result = engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    assert result.ocr_performed is True
    assert accelerated.calls == 1
    assert authoritative.calls == 1


def test_tracking_risk_forces_same_frame_authoritative_source_scan(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _EmptyOcrEngine()
    tracker = RegionTracker(
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
        template_matcher=_FailedTemplateMatcher(),  # type: ignore[arg-type]
    )
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        tracker=tracker,
    )
    engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    result = engine.process_frame(frame, frame_index=1, timestamp_ms=40)

    assert result.ocr_performed is True
    assert result.ocr_required is False
    assert accelerated.calls == 2
    assert authoritative.calls == 2


def test_scheduled_native_scan_still_forces_paddle_on_scene_change(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    replacement = np.full_like(frame, 245)
    authoritative = _MockOcrEngine()
    accelerated = _EmptyOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
    )
    engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    result = engine.process_frame(replacement, frame_index=1, timestamp_ms=130)

    assert result.scene_changed is True
    assert result.ocr_performed is True
    assert result.regions
    assert accelerated.calls == 2
    assert authoritative.calls == 2


def test_native_source_runtime_failure_falls_back_to_authoritative_ocr(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _FailingOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
    )

    first = engine.process_frame(frame, frame_index=0, timestamp_ms=0)
    second = engine.process_frame(frame, frame_index=1, timestamp_ms=130)

    assert first.regions
    assert second.regions
    assert accelerated.calls == 1
    assert authoritative.calls == 2
    assert engine.fast_ocr_engine is None


def test_native_source_runtime_failure_between_audits_uses_authoritative_ocr(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    authoritative = _MockOcrEngine()
    accelerated = _FailOnSecondOcrEngine()
    engine = AdaptiveTrackingEngine(
        authoritative,
        fast_ocr_engine=accelerated,
    )

    first = engine.process_frame(frame, frame_index=0, timestamp_ms=0)
    second = engine.process_frame(frame, frame_index=1, timestamp_ms=130)

    assert first.regions
    assert second.regions
    assert accelerated.calls == 2
    assert authoritative.calls == 2
    assert engine.fast_ocr_engine is None


def test_adaptive_engine_can_omit_per_frame_track_history(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    engine = AdaptiveTrackingEngine(
        _MockOcrEngine(),
        include_track_snapshots=False,
    )

    result = engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    assert result.regions
    assert result.tracks == ()
    assert len(engine.tracker.tracks) == 1


def test_scene_change_triggers_same_frame_ocr_reacquisition(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    previous, _ = tracking_frame_factory()
    replacement = np.full_like(previous, 245)
    ocr = _MockOcrEngine()
    engine = AdaptiveTrackingEngine(ocr)
    engine.process_frame(previous, frame_index=0, timestamp_ms=0)

    result = engine.process_frame(replacement, frame_index=1, timestamp_ms=40)

    assert result.scene_changed is True
    assert result.ocr_performed is True
    assert result.ocr_required is False
    assert result.regions[0].state is TrackState.TRACKING
    assert ocr.calls == 2


def test_tracking_confidence_loss_triggers_same_frame_ocr_reacquisition(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    ocr = _MockOcrEngine()
    tracker = RegionTracker(
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
        template_matcher=_FailedTemplateMatcher(),  # type: ignore[arg-type]
    )
    engine = AdaptiveTrackingEngine(ocr, tracker=tracker)
    engine.process_frame(frame, frame_index=0, timestamp_ms=0)

    result = engine.process_frame(frame, frame_index=1, timestamp_ms=40)

    assert result.scene_changed is False
    assert result.ocr_performed is True
    assert result.ocr_required is False
    assert result.regions[0].state is TrackState.TRACKING
    assert ocr.calls == 2


def test_failed_reacquisition_retries_at_safe_rate_not_every_frame(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    ocr = _OneShotOcrEngine()
    tracker = RegionTracker(
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
        template_matcher=_FailedTemplateMatcher(),  # type: ignore[arg-type]
    )
    engine = AdaptiveTrackingEngine(ocr, tracker=tracker)

    initial = engine.process_frame(frame, frame_index=0, timestamp_ms=0)
    first_loss = engine.process_frame(frame, frame_index=1, timestamp_ms=40)
    rate_limited = engine.process_frame(frame, frame_index=2, timestamp_ms=60)
    scheduled_retry = engine.process_frame(frame, frame_index=3, timestamp_ms=130)

    assert initial.ocr_performed is True
    assert first_loss.ocr_performed is True
    assert first_loss.ocr_required is True
    assert rate_limited.ocr_performed is False
    assert rate_limited.ocr_required is True
    assert rate_limited.regions[0].state is TrackState.STICKY
    assert scheduled_retry.ocr_performed is True
    assert scheduled_retry.ocr_required is True
    assert ocr.calls == 3


def test_scene_change_bypasses_sticky_retry_rate_limit(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    frame, _ = tracking_frame_factory()
    replacement = np.full_like(frame, 245)
    ocr = _OneShotOcrEngine()
    tracker = RegionTracker(
        optical_flow=_FailedOpticalFlow(),  # type: ignore[arg-type]
        template_matcher=_FailedTemplateMatcher(),  # type: ignore[arg-type]
    )
    engine = AdaptiveTrackingEngine(ocr, tracker=tracker)

    engine.process_frame(frame, frame_index=0, timestamp_ms=0)
    first_loss = engine.process_frame(frame, frame_index=1, timestamp_ms=40)
    scene_change = engine.process_frame(
        replacement,
        frame_index=2,
        timestamp_ms=60,
    )

    assert first_loss.regions[0].state is TrackState.STICKY
    assert scene_change.scene_changed is True
    assert scene_change.ocr_performed is True
    assert scene_change.ocr_required is True
    assert ocr.calls == 3
