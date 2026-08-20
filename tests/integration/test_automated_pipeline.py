from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from redact_video.config import (
    AppConfig,
    DetectionConfig,
    OcrConfig,
    TrackingConfig,
    VerificationConfig,
)
from redact_video.exceptions import OcrProcessingError
from redact_video.media.probe import probe_media
from redact_video.models import BoundingBox, JobStatus, OcrToken, SensitiveType
from redact_video.ocr.apple_vision import AppleVisionOcrEngine
from redact_video.pipeline.automated import (
    _source_detection_pre_roll_frames,
    process_video_automated,
)
from redact_video.tracking import AdaptiveTrackingEngine

pytestmark = pytest.mark.integration


def test_automated_pipeline_builds_timeline_renders_masks_and_preserves_media(
    media_factory: Callable[..., Path],
    card_field_ocr,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=True)
    output = tmp_path / "automatic.redacted.mp4"
    original_opencv_threads = cv2.getNumThreads()

    try:
        result = process_video_automated(
            source,
            output_path=output,
            ocr_engine=card_field_ocr,
        )
        assert cv2.getNumThreads() == 1
    finally:
        cv2.setNumThreads(original_opencv_threads)
    source_info = probe_media(source)
    output_info = probe_media(output)

    assert result.status is JobStatus.COMPLETED
    assert result.frames_processed == 10
    assert result.masked_frames == 10
    assert result.mask_regions >= result.masked_frames
    assert result.tracks_by_type[SensitiveType.CARD_NUMBER] >= 1
    assert result.verification.passes == 1
    assert result.verification.repairs == 0
    assert result.verification.remaining_leaks == 0
    assert len(result.timings.verification_passes_ms) == 1
    assert result.timings.repair_passes_ms == ()
    assert result.timings.total_ms >= result.timings.source_analysis_ms
    assert len(result.mask_timeline.frames) == result.masked_frames
    assert result.audio_preserved is True
    assert output_info.video_codec == "h264"
    assert output_info.pixel_format == "yuv420p"
    assert output_info.has_audio is True
    assert abs(output_info.duration_ms - source_info.duration_ms) < 100
    serialized = str(result.model_dump()).lower()
    assert "mask_timeline" not in serialized
    assert "value" not in serialized

    first_box = result.mask_timeline.boxes_for(0)[0]
    source_capture = cv2.VideoCapture(str(source))
    output_capture = cv2.VideoCapture(str(output))
    source_ok, source_frame = source_capture.read()
    output_ok, output_frame = output_capture.read()
    source_capture.release()
    output_capture.release()
    assert source_ok and output_ok
    source_roi = source_frame[
        first_box.y : first_box.bottom,
        first_box.x : first_box.right,
    ]
    output_roi = output_frame[
        first_box.y : first_box.bottom,
        first_box.x : first_box.right,
    ]
    assert output_roi.std() < source_roi.std()


class _EmptyOcr:
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        return []


class _CountingEmptyOcr:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        return []

    def close(self) -> None:
        self.closed = True


class _CountingNativeOcr(AppleVisionOcrEngine):
    def __init__(self) -> None:
        super().__init__(runner=lambda _: b"[]")
        self.calls = 0
        self.closed = False

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        return []

    def close(self) -> None:
        self.closed = True


class _PaddleLikeEmptyOcr(_CountingEmptyOcr):
    backend_name = "paddle"


class _FailingNativeOcr(_CountingNativeOcr):
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        raise OcrProcessingError("native OCR failed")


class _OneShotLeakOcr:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        if self.calls > 1:
            return []
        return [
            OcrToken(
                text="4242424242424242",
                confidence=0.99,
                bbox=BoundingBox(x=70, y=80, width=150, height=28),
            )
        ]


class _PersistentLeakOcr:
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        return [
            OcrToken(
                text="4242424242424242",
                confidence=0.99,
                bbox=BoundingBox(x=70, y=80, width=150, height=28),
            )
        ]


def test_source_pre_roll_covers_audit_and_scan_scheduler_delay() -> None:
    tracker = AdaptiveTrackingEngine(
        _EmptyOcr(),
        fast_ocr_engine=_EmptyOcr(),
        detection_config=DetectionConfig(scan_fps=8, safe_scan_fps=12),
        tracking_config=TrackingConfig(authoritative_audit_interval_ms=350),
    )

    assert _source_detection_pre_roll_frames(tracker, frame_rate=60) == 29


def test_source_pre_roll_is_disabled_without_native_acceleration() -> None:
    tracker = AdaptiveTrackingEngine(_EmptyOcr())

    assert _source_detection_pre_roll_frames(tracker, frame_rate=60) == 0


def test_automated_pipeline_uses_available_native_verification_accelerator(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    authoritative = _PaddleLikeEmptyOcr()
    primary = _CountingNativeOcr()
    secondary = _CountingNativeOcr()
    factory_calls = 0

    def create_accelerator(
        cls: type[AppleVisionOcrEngine],
        config: object,
    ) -> _CountingNativeOcr:
        nonlocal factory_calls
        del cls, config
        factory_calls += 1
        return primary

    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "try_create",
        classmethod(create_accelerator),
    )
    monkeypatch.setattr(primary, "try_create_peer", lambda: secondary)

    result = process_video_automated(
        source,
        output_path=tmp_path / "accelerated.redacted.mp4",
        ocr_engine=authoritative,
    )

    assert result.status is JobStatus.COMPLETED
    assert factory_calls == 1
    assert primary.calls > 0
    assert secondary.calls > 0
    assert authoritative.calls > 0
    assert primary.closed is True
    assert secondary.closed is True


def test_automated_pipeline_uses_native_source_acceleration_with_paddle_audits(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    authoritative = _PaddleLikeEmptyOcr()
    accelerated = _CountingEmptyOcr()
    factory_calls = 0

    def create_accelerator(
        cls: type[AppleVisionOcrEngine],
        config: object,
    ) -> _CountingEmptyOcr:
        nonlocal factory_calls
        del cls, config
        factory_calls += 1
        return accelerated

    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "try_create",
        classmethod(create_accelerator),
    )

    result = process_video_automated(
        source,
        output_path=tmp_path / "source-accelerated.redacted.mp4",
        ocr_engine=authoritative,
        config=AppConfig(
            tracking=TrackingConfig(authoritative_audit_interval_ms=250),
            verification=VerificationConfig(enabled=False),
        ),
    )

    assert result.status is JobStatus.NEEDS_REVIEW
    assert factory_calls == 1
    assert accelerated.calls > authoritative.calls > 0
    assert accelerated.closed is True


def test_default_pipeline_uses_v6_onnx_audits_with_fast_native_scans(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    audit = _PaddleLikeEmptyOcr()
    native = _CountingNativeOcr()
    paddle_configs: list[OcrConfig] = []
    native_configs: list[OcrConfig] = []

    def create_paddle(config: OcrConfig) -> _PaddleLikeEmptyOcr:
        paddle_configs.append(config)
        return audit

    def create_native(
        cls: type[AppleVisionOcrEngine],
        config: OcrConfig,
    ) -> _CountingNativeOcr:
        del cls
        native_configs.append(config)
        return native

    monkeypatch.setattr(
        "redact_video.pipeline.automated.PaddleOcrEngine",
        create_paddle,
    )
    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "try_create",
        classmethod(create_native),
    )

    process_video_automated(
        source,
        output_path=tmp_path / "optimized-default.redacted.mp4",
        config=AppConfig(
            tracking=TrackingConfig(authoritative_audit_interval_ms=250),
            verification=VerificationConfig(enabled=False),
        ),
    )

    assert len(paddle_configs) == 1
    assert len(native_configs) == 1
    paddle_config = paddle_configs[0]
    native_config = native_configs[0]
    assert paddle_config.text_detection_model_name == "PP-OCRv6_tiny_det"
    assert paddle_config.text_recognition_model_name == "PP-OCRv6_tiny_rec"
    assert paddle_config.inference_engine == "onnxruntime"
    assert native_config.apple_vision_recognition_level == "fast"
    assert native.calls > audit.calls > 0
    assert native.closed is True


def test_default_pipeline_falls_back_to_v6_onnx_if_native_ocr_fails(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    audit = _PaddleLikeEmptyOcr()
    native = _FailingNativeOcr()
    monkeypatch.setattr(
        "redact_video.pipeline.automated.PaddleOcrEngine",
        lambda config: audit,
    )
    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "try_create",
        classmethod(lambda cls, config: native),
    )

    process_video_automated(
        source,
        output_path=tmp_path / "native-fallback.redacted.mp4",
        config=AppConfig(verification=VerificationConfig(enabled=False)),
    )

    assert native.calls == 1
    assert audit.calls > 1
    assert native.closed is True


def test_automated_pipeline_does_not_initialize_disabled_native_acceleration(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    factory_calls = 0

    def create_accelerator(
        cls: type[AppleVisionOcrEngine],
        config: object,
    ) -> _CountingEmptyOcr:
        nonlocal factory_calls
        del cls, config
        factory_calls += 1
        return _CountingEmptyOcr()

    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "try_create",
        classmethod(create_accelerator),
    )

    process_video_automated(
        source,
        output_path=tmp_path / "native-disabled.redacted.mp4",
        ocr_engine=_PaddleLikeEmptyOcr(),
        config=AppConfig(
            tracking=TrackingConfig(use_apple_vision_acceleration=False),
            verification=VerificationConfig(enabled=False),
        ),
    )

    assert factory_calls == 0


def test_automated_pipeline_closes_owned_native_helper_after_failure(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    accelerated = _CountingEmptyOcr()
    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "try_create",
        classmethod(lambda cls, config: accelerated),
    )

    def fail_analysis(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("intentional analysis failure")

    monkeypatch.setattr(
        "redact_video.pipeline.automated._analyze_source",
        fail_analysis,
    )

    with pytest.raises(RuntimeError, match="intentional analysis failure"):
        process_video_automated(
            source,
            output_path=tmp_path / "failed.redacted.mp4",
            ocr_engine=_PaddleLikeEmptyOcr(),
        )

    assert accelerated.closed is True


def test_automated_pipeline_removes_staged_output_when_interrupted(
    media_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "interrupted.redacted.mp4"

    def interrupt_verification(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "redact_video.pipeline.automated.SafetyVerifier.verify",
        interrupt_verification,
    )

    with pytest.raises(KeyboardInterrupt):
        process_video_automated(
            source,
            output_path=output,
            ocr_engine=_EmptyOcr(),
        )

    assert output.exists() is False
    assert list(tmp_path.glob(".*.partial.mp4")) == []


def test_automated_pipeline_repairs_an_intentionally_missed_region(
    media_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "repaired.redacted.mp4"

    result = process_video_automated(
        source,
        output_path=output,
        ocr_engine=_EmptyOcr(),
        verification_ocr_engine=_OneShotLeakOcr(),
    )

    assert result.status is JobStatus.COMPLETED
    assert result.verification.passes == 2
    assert result.verification.repairs == 1
    assert result.verification.remaining_leaks == 0
    assert len(result.timings.verification_passes_ms) == 2
    assert len(result.timings.repair_passes_ms) == 1
    assert result.masked_frames >= 5
    assert any(
        region.track_id.startswith("repair-p01")
        for regions in result.mask_timeline.frames.values()
        for region in regions
    )

    source_capture = cv2.VideoCapture(str(source))
    output_capture = cv2.VideoCapture(str(output))
    source_decoded, source_frame = source_capture.read()
    output_decoded, output_frame = output_capture.read()
    source_capture.release()
    output_capture.release()
    assert source_decoded and output_decoded
    source_roi = source_frame[68:120, 58:232]
    repaired_roi = output_frame[68:120, 58:232]
    assert repaired_roi.std() < source_roi.std()


def test_automated_pipeline_never_completes_with_persistent_leaks(
    media_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "needs-review.redacted.mp4"

    result = process_video_automated(
        source,
        output_path=output,
        ocr_engine=_EmptyOcr(),
        verification_ocr_engine=_PersistentLeakOcr(),
    )

    assert result.status is JobStatus.NEEDS_REVIEW
    assert result.verification.passes == 3
    assert result.verification.repairs == 2
    assert result.verification.remaining_leaks == 10
    assert len(result.timings.verification_passes_ms) == 3
    assert len(result.timings.repair_passes_ms) == 2
    assert result.verification.potential_issues
    assert "possible leaks" in result.warnings[0]


def test_automated_pipeline_cannot_complete_when_verification_is_disabled(
    media_factory: Callable[..., Path],
    tmp_path: Path,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)

    result = process_video_automated(
        source,
        output_path=tmp_path / "unverified.redacted.mp4",
        ocr_engine=_EmptyOcr(),
        config=AppConfig(verification=VerificationConfig(enabled=False)),
    )

    assert result.status is JobStatus.NEEDS_REVIEW
    assert result.verification.passes == 0
    assert result.verification.repairs == 0
    assert result.verification.remaining_leaks == 0
    assert result.warnings == ("Automatic output verification is disabled",)
    assert result.timings.verification_passes_ms == ()
    assert result.timings.repair_passes_ms == ()
