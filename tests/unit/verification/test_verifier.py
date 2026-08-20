from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from threading import Barrier
from typing import Self

import numpy as np
import pytest

from redact_video.config import DetectionConfig, VerificationConfig
from redact_video.exceptions import DecodeError, OcrProcessingError
from redact_video.media.decoder import VideoFrame
from redact_video.models import (
    BoundingBox,
    OcrToken,
    SensitiveDetection,
    SensitiveType,
)
from redact_video.verification.verifier import (
    SafetyVerifier,
    _PrefetchedAuthoritativeScan,
    _PrefetchedFastScan,
    sensitive_detection_config,
)


class _EmptyOcr:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        return []


class _EmptyDetector:
    def detect(
        self,
        frame: np.ndarray,
        ocr: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]:
        del frame, ocr, frame_index, timestamp_ms
        return []


class _TokenSensitiveDetector:
    def detect(
        self,
        frame: np.ndarray,
        ocr: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]:
        del frame
        if not ocr:
            return []
        return [
            SensitiveDetection(
                type=SensitiveType.CARD_NUMBER,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                bbox=ocr[0].bbox,
                confidence=ocr[0].confidence,
                source="luhn_pan",
            )
        ]


class _StaticTokenOcr:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        return [
            OcrToken(
                text="4242424242424242",
                confidence=0.99,
                bbox=BoundingBox(x=20, y=20, width=160, height=24),
            )
        ]


class _FailingOcr:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        raise OcrProcessingError("native OCR failed")


class _FailOnSecondOcr(_EmptyOcr):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        if self.calls == 1:
            self.calls += 1
            raise OcrProcessingError("native OCR failed")
        return super().scan(frame)


class _BarrierOcr(_EmptyOcr):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self.barrier = barrier

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        self.barrier.wait(timeout=2)
        return super().scan(frame)


class _BarrierOnCallOcr(_EmptyOcr):
    def __init__(self, barrier: Barrier, *, wait_on_call: int) -> None:
        super().__init__()
        self.barrier = barrier
        self.wait_on_call = wait_on_call

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        if self.calls == self.wait_on_call:
            self.barrier.wait(timeout=2)
        return []


class _ScriptedDetector:
    def detect(
        self,
        frame: np.ndarray,
        ocr: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]:
        del frame, ocr
        return [
            SensitiveDetection(
                type=SensitiveType.CARD_NUMBER,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                bbox=BoundingBox(x=20, y=40, width=160, height=24),
                confidence=0.92,
                source="semantic_anchor",
            ),
            SensitiveDetection(
                type=SensitiveType.CARD_CVV,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                bbox=BoundingBox(x=220, y=40, width=60, height=24),
                confidence=0.88,
                source="semantic_cvv_value",
            ),
        ]


def test_verifier_ignores_label_only_anchors_but_reports_readable_values() -> None:
    verifier = SafetyVerifier(
        _EmptyOcr(),
        detection_engine=_ScriptedDetector(),
    )

    leaks = verifier.inspect_frame(
        np.zeros((120, 320, 3), dtype=np.uint8),
        frame_index=4,
        timestamp_ms=400,
    )

    assert len(leaks) == 1
    assert leaks[0].type is SensitiveType.CARD_CVV
    assert leaks[0].source == "semantic_cvv_value"
    assert "value" not in leaks[0].__class__.model_fields
    assert "text" not in leaks[0].__class__.model_fields


def test_verification_detection_config_lowers_every_threshold() -> None:
    base = DetectionConfig()

    sensitive = sensitive_detection_config(
        base,
        VerificationConfig(threshold_scale=0.8),
    )

    assert sensitive.thresholds.card_number == base.thresholds.card_number * 0.8
    assert sensitive.thresholds.expiry == base.thresholds.expiry * 0.8
    assert sensitive.thresholds.cvv == base.thresholds.cvv * 0.8
    assert sensitive.thresholds.saved_card == base.thresholds.saved_card * 0.8
    assert sensitive.scan_fps == 120
    assert sensitive.safe_scan_fps == 120


def _use_frames(
    monkeypatch: pytest.MonkeyPatch,
    frames: list[VideoFrame],
) -> None:
    class _Decoder:
        def __init__(self, path: Path, fallback_frame_rate: float) -> None:
            del path, fallback_frame_rate

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(frames)

    class _SelectedDecoder(_Decoder):
        def __init__(
            self,
            path: Path,
            fallback_frame_rate: float,
            *,
            frame_indices: tuple[int, ...],
        ) -> None:
            super().__init__(path, fallback_frame_rate)
            self.frame_indices = frozenset(frame_indices)

        def __iter__(self):  # type: ignore[no-untyped-def]
            return (frame for frame in frames if frame.index in self.frame_indices)

    class _CompactDecoder(_Decoder):
        def __init__(
            self,
            path: Path,
            fallback_frame_rate: float,
            *,
            max_dimension: int,
        ) -> None:
            del max_dimension
            super().__init__(path, fallback_frame_rate)

    monkeypatch.setattr("redact_video.verification.verifier.FrameDecoder", _Decoder)
    monkeypatch.setattr(
        "redact_video.verification.verifier.CompactFrameDecoder",
        _CompactDecoder,
    )
    monkeypatch.setattr(
        "redact_video.verification.verifier.SelectedFrameDecoder",
        _SelectedDecoder,
    )


def _video_frames(images: list[np.ndarray]) -> list[VideoFrame]:
    return [
        VideoFrame(index=index, timestamp_ms=round(index * 1000 / 60), image=image)
        for index, image in enumerate(images)
    ]


def test_verifier_reuses_only_clean_unchanged_frames_until_forced_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(8)]))
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(
        ocr,
        verification_config=VerificationConfig(reuse_max_interval_ms=100),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.frames_scanned == 8
    assert result.passed is True
    assert ocr.calls == 2


def test_first_pass_builds_only_one_clean_evidence_guard_per_static_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    frames = _video_frames([image.copy() for _ in range(3)])
    _use_frames(monkeypatch, frames)
    verifier = SafetyVerifier(_EmptyOcr(), detection_engine=_EmptyDetector())
    original_guard_image = verifier._guard_image
    guard_calls = 0

    def counted_guard_image(frame: np.ndarray) -> np.ndarray:
        nonlocal guard_calls
        guard_calls += 1
        return original_guard_image(frame)

    monkeypatch.setattr(verifier, "_guard_image", counted_guard_image)

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert guard_calls == len(frames)


def test_verifier_scans_immediately_when_more_than_noise_bound_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.zeros((40, 60, 3), dtype=np.uint8)
    changed = original.copy()
    changed[:5, :5] = 255
    _use_frames(monkeypatch, _video_frames([original, changed, changed.copy()]))
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(
        ocr,
        verification_config=VerificationConfig(
            reuse_max_changed_fraction=0.01,
            minimum_scan_interval_ms=1,
        ),
        detection_engine=_EmptyDetector(),
    )

    verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert ocr.calls == 2


def test_verifier_compares_reused_frames_with_last_ocr_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.zeros((40, 60, 3), dtype=np.uint8)
    within_bound = original.copy()
    within_bound[:4, :6] = 255
    accumulated_change = within_bound.copy()
    accumulated_change[4, 6] = 255
    _use_frames(
        monkeypatch,
        _video_frames([original, within_bound, accumulated_change]),
    )
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(
        ocr,
        verification_config=VerificationConfig(
            reuse_max_changed_fraction=0.01,
            minimum_scan_interval_ms=1,
        ),
        detection_engine=_EmptyDetector(),
    )

    verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert ocr.calls == 2


def test_verifier_rate_limits_material_changes_between_scan_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images: list[np.ndarray] = []
    for index in range(8):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[:5, : 5 * index] = 255
        images.append(image)
    _use_frames(monkeypatch, _video_frames(images))
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(
        ocr,
        verification_config=VerificationConfig(minimum_scan_interval_ms=100),
        detection_engine=_EmptyDetector(),
    )

    verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert ocr.calls == 2


def test_verifier_scans_immediately_on_a_major_scene_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dark = np.zeros((40, 60, 3), dtype=np.uint8)
    light = np.full((40, 60, 3), 255, dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([dark, light, light.copy()]))
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(
        ocr,
        verification_config=VerificationConfig(minimum_scan_interval_ms=100),
        detection_engine=_EmptyDetector(),
    )

    verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert ocr.calls == 2


def test_verifier_ignores_isolated_codec_noise() -> None:
    verifier = SafetyVerifier(_EmptyOcr(), detection_engine=_EmptyDetector())
    original = np.zeros((40, 60, 3), dtype=np.uint8)
    one_changed_pixel = original.copy()
    one_changed_pixel[0, 0] = 7

    assert verifier._can_reuse_clean_scan(original, original.copy()) is True
    assert verifier._can_reuse_clean_scan(original, one_changed_pixel) is True


def test_verifier_never_exceeds_absolute_noise_pixel_cap() -> None:
    verifier = SafetyVerifier(
        _EmptyOcr(),
        verification_config=VerificationConfig(
            reuse_guard_max_dimension=512,
            reuse_pixel_delta=0,
            reuse_max_changed_fraction=0.01,
        ),
        detection_engine=_EmptyDetector(),
    )
    original = np.zeros((100, 100, 3), dtype=np.uint8)
    at_cap = original.copy()
    at_cap[:8, :8] = 255
    beyond_cap = at_cap.copy()
    beyond_cap[8, 0] = 255

    assert verifier._can_reuse_clean_scan(original, at_cap) is True
    assert verifier._can_reuse_clean_scan(original, beyond_cap) is False


def test_verifier_never_reuses_a_scan_that_found_a_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 320, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(3)]))
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(ocr, detection_engine=_ScriptedDetector())

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert ocr.calls == 3
    assert len(result.leaks) == 3


def test_verifier_reuses_matching_clean_evidence_on_the_next_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    frames = _video_frames([image.copy() for _ in range(3)])
    _use_frames(monkeypatch, frames)
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(ocr, detection_engine=_EmptyDetector())

    first = verifier.verify(Path("first.mp4"), frame_rate=60, pass_number=1)
    first_pass_calls = ocr.calls
    _use_frames(monkeypatch, frames)
    second = verifier.verify(Path("repair.mp4"), frame_rate=60, pass_number=2)

    assert first.passed is True
    assert second.passed is True
    assert first_pass_calls == 1
    assert ocr.calls == first_pass_calls


def test_verifier_rescans_a_changed_frame_between_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([original.copy() for _ in range(3)]))
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(ocr, detection_engine=_EmptyDetector())

    verifier.verify(Path("first.mp4"), frame_rate=60, pass_number=1)
    changed = original.copy()
    changed[:10, :10] = 255
    _use_frames(
        monkeypatch,
        _video_frames([original.copy(), changed, original.copy()]),
    )
    verifier.verify(Path("repair.mp4"), frame_rate=60, pass_number=2)

    assert ocr.calls == 2


def test_verifier_never_reuses_leaking_evidence_between_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 320, 3), dtype=np.uint8)
    frames = _video_frames([image])
    _use_frames(monkeypatch, frames)
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(ocr, detection_engine=_ScriptedDetector())

    first = verifier.verify(Path("first.mp4"), frame_rate=60, pass_number=1)
    _use_frames(monkeypatch, frames)
    second = verifier.verify(Path("repair.mp4"), frame_rate=60, pass_number=2)

    assert len(first.leaks) == 1
    assert len(second.leaks) == 1
    assert ocr.calls == 2


def test_fast_prefetch_verifier_reuses_clean_evidence_on_the_next_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    frames = _video_frames([image.copy() for _ in range(3)])
    _use_frames(monkeypatch, frames)
    authoritative = _EmptyOcr()
    primary = _EmptyOcr()
    secondary = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=primary,
        secondary_fast_ocr_engine=secondary,
        verification_config=VerificationConfig(prefetch_authoritative_audits=False),
        detection_engine=_EmptyDetector(),
    )

    verifier.verify(Path("first.mp4"), frame_rate=60, pass_number=1)
    calls_after_first = (authoritative.calls, primary.calls, secondary.calls)
    _use_frames(monkeypatch, frames)
    result = verifier.verify(Path("repair.mp4"), frame_rate=60, pass_number=2)

    assert result.passed is True
    assert (authoritative.calls, primary.calls, secondary.calls) == calls_after_first


def test_cross_pass_verification_falls_back_when_compact_decode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    frames = _video_frames([image.copy() for _ in range(3)])
    _use_frames(monkeypatch, frames)
    ocr = _EmptyOcr()
    verifier = SafetyVerifier(ocr, detection_engine=_EmptyDetector())
    verifier.verify(Path("first.mp4"), frame_rate=60, pass_number=1)

    class _FailingCompactDecoder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def __enter__(self) -> Self:
            raise DecodeError("compact decoder unavailable")

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr(
        "redact_video.verification.verifier.CompactFrameDecoder",
        _FailingCompactDecoder,
    )

    result = verifier.verify(Path("repair.mp4"), frame_rate=60, pass_number=2)

    assert result.passed is True
    assert result.frames_scanned == 3


def test_fast_ocr_checks_selected_frames_while_authoritative_ocr_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(8)]))
    authoritative = _EmptyOcr()
    accelerated = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        verification_config=VerificationConfig(
            reuse_clean_frames=False,
            authoritative_audit_interval_ms=100,
        ),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert accelerated.calls == 8
    assert authoritative.calls == 2


def test_authoritative_audit_catches_a_fast_ocr_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((60, 240, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image]))
    authoritative = _StaticTokenOcr()
    accelerated = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        detection_engine=_TokenSensitiveDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert accelerated.calls == 1
    assert authoritative.calls == 1
    assert len(result.leaks) == 1


def test_verification_audit_runs_independent_ocr_engines_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image]))
    barrier = Barrier(2)
    authoritative = _BarrierOcr(barrier)
    accelerated = _BarrierOcr(barrier)
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert accelerated.calls == 1
    assert authoritative.calls == 1


def test_authoritative_audit_prefetches_with_fast_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image]))
    barrier = Barrier(2)
    authoritative = _BarrierOcr(barrier)
    primary = _BarrierOcr(barrier)
    secondary = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=primary,
        secondary_fast_ocr_engine=secondary,
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert primary.calls == 1
    assert authoritative.calls == 1


def test_authoritative_prefetch_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image]))
    authoritative = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=_EmptyOcr(),
        secondary_fast_ocr_engine=_EmptyOcr(),
        verification_config=VerificationConfig(prefetch_authoritative_audits=False),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert authoritative.calls == 1


def test_stale_pending_authoritative_prefetch_falls_back_without_waiting() -> None:
    authoritative = _EmptyOcr()
    accelerated = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        detection_engine=_EmptyDetector(),
    )
    fast_future: Future[list[OcrToken]] = Future()
    fast_future.set_result([])
    stale_audit_future: Future[list[OcrToken]] = Future()

    leaks, authoritative_ran = verifier._inspect_prefetched_frame(
        np.zeros((40, 60, 3), dtype=np.uint8),
        _PrefetchedFastScan(engine=accelerated, future=fast_future),
        authoritative_due=True,
        authoritative_scan=_PrefetchedAuthoritativeScan(
            frame_index=5,
            future=stale_audit_future,
        ),
        frame_index=5,
        timestamp_ms=500,
    )

    assert leaks == ()
    assert authoritative_ran is True
    assert stale_audit_future.cancelled() is True
    assert authoritative.calls == 1


def test_two_fast_engines_prefetch_consecutive_changed_frames_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = [np.full((40, 60, 3), value, dtype=np.uint8) for value in (0, 40, 80)]
    _use_frames(monkeypatch, _video_frames(images))
    barrier = Barrier(2)
    authoritative = _EmptyOcr()
    primary = _BarrierOnCallOcr(barrier, wait_on_call=2)
    secondary = _BarrierOnCallOcr(barrier, wait_on_call=1)
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=primary,
        secondary_fast_ocr_engine=secondary,
        verification_config=VerificationConfig(reuse_clean_frames=False),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert primary.calls == 2
    assert secondary.calls == 1
    assert authoritative.calls == 1


def test_two_fast_engines_preserve_static_frame_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(3)]))
    authoritative = _EmptyOcr()
    primary = _EmptyOcr()
    secondary = _EmptyOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=primary,
        secondary_fast_ocr_engine=secondary,
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert primary.calls == 1
    assert secondary.calls == 0
    assert authoritative.calls == 1


def test_prefetch_failure_disables_both_fast_engines_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    images = [np.full((40, 60, 3), value, dtype=np.uint8) for value in (0, 40, 80)]
    _use_frames(monkeypatch, _video_frames(images))
    authoritative = _EmptyOcr()
    primary = _EmptyOcr()
    secondary = _FailingOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=primary,
        secondary_fast_ocr_engine=secondary,
        verification_config=VerificationConfig(reuse_clean_frames=False),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert primary.calls == 2
    assert secondary.calls == 1
    assert authoritative.calls == 2
    assert verifier.fast_ocr_engine is None
    assert verifier.secondary_fast_ocr_engine is None


def test_fast_leak_triggers_immediate_audit_without_duplicate_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((60, 240, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(3)]))
    authoritative = _StaticTokenOcr()
    accelerated = _StaticTokenOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        detection_engine=_TokenSensitiveDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert accelerated.calls == 3
    assert authoritative.calls == 3
    assert len(result.leaks) == 3


def test_fast_ocr_runtime_failure_falls_back_to_authoritative_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((60, 240, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(3)]))
    authoritative = _EmptyOcr()
    accelerated = _FailingOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        verification_config=VerificationConfig(reuse_clean_frames=False),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert accelerated.calls == 1
    assert authoritative.calls == 3
    assert verifier.fast_ocr_engine is None


def test_fast_ocr_failure_between_audits_uses_authoritative_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((60, 240, 3), dtype=np.uint8)
    _use_frames(monkeypatch, _video_frames([image.copy() for _ in range(2)]))
    authoritative = _EmptyOcr()
    accelerated = _FailOnSecondOcr()
    verifier = SafetyVerifier(
        authoritative,
        fast_ocr_engine=accelerated,
        verification_config=VerificationConfig(reuse_clean_frames=False),
        detection_engine=_EmptyDetector(),
    )

    result = verifier.verify(Path("unused.mp4"), frame_rate=60, pass_number=1)

    assert result.passed is True
    assert accelerated.calls == 2
    assert authoritative.calls == 2
    assert verifier.fast_ocr_engine is None
