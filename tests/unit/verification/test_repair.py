from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import pytest
from pydantic import ValidationError

from redact_video.config import DetectionConfig, TrackingConfig, VerificationConfig
from redact_video.media.decoder import VideoFrame
from redact_video.models import BoundingBox, OcrToken, SensitiveType
from redact_video.redaction.timeline import MaskTimeline
from redact_video.verification.models import RepairWindow, VerificationLeak
from redact_video.verification.repair import RepairEngine, build_repair_windows


def _leak(timestamp_ms: int, sensitive_type: SensitiveType) -> VerificationLeak:
    return VerificationLeak(
        type=sensitive_type,
        frame_index=timestamp_ms // 100,
        timestamp_ms=timestamp_ms,
        bbox=BoundingBox(x=40, y=60, width=120, height=24),
        confidence=0.91,
        source="pattern_luhn",
    )


def test_repair_windows_expand_clamp_merge_and_hide_values() -> None:
    windows = build_repair_windows(
        (
            _leak(100, SensitiveType.CARD_NUMBER),
            _leak(700, SensitiveType.CARD_CVV),
            _leak(1900, SensitiveType.CARD_EXPIRY),
        ),
        duration_ms=2000,
        config=VerificationConfig(
            repair_pre_roll_ms=350,
            repair_post_roll_ms=500,
        ),
    )

    assert windows == (
        RepairWindow(
            start_ms=0,
            end_ms=1200,
            types=(SensitiveType.CARD_CVV, SensitiveType.CARD_NUMBER),
        ),
        RepairWindow(
            start_ms=1550,
            end_ms=2000,
            types=(SensitiveType.CARD_EXPIRY,),
        ),
    )
    assert "value" not in str([window.model_dump() for window in windows]).lower()


def test_repair_window_rejects_an_inverted_range() -> None:
    with pytest.raises(ValidationError, match="precede"):
        RepairWindow(
            start_ms=500,
            end_ms=400,
            types=(SensitiveType.CARD_NUMBER,),
        )


class _CountingEmptyOcr:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        self.calls += 1
        return []


def test_repair_uses_fast_ocr_with_authoritative_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        VideoFrame(
            index=index,
            timestamp_ms=timestamp_ms,
            image=np.zeros((40, 60, 3), dtype=np.uint8),
        )
        for index, timestamp_ms in enumerate((0, 50, 84, 100))
    ]

    class _Decoder:
        def __init__(self, path: Path, fallback_frame_rate: float) -> None:
            del path, fallback_frame_rate

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(frames)

    monkeypatch.setattr("redact_video.verification.repair.FrameDecoder", _Decoder)
    authoritative = _CountingEmptyOcr()
    accelerated = _CountingEmptyOcr()
    engine = RepairEngine(
        authoritative,
        fast_ocr_engine=accelerated,
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(authoritative_audit_interval_ms=350),
        verification_config=VerificationConfig(),
    )

    patched = engine.patch_timeline(
        Path("unused.mp4"),
        MaskTimeline(),
        (),
        (
            RepairWindow(
                start_ms=0,
                end_ms=100,
                types=(SensitiveType.CARD_NUMBER,),
            ),
        ),
        frame_rate=60,
        pass_number=1,
    )

    assert patched == MaskTimeline()
    assert authoritative.calls == 1
    assert accelerated.calls == 4


def test_repair_seed_patch_covers_every_frame_in_the_window() -> None:
    leak = _leak(100, SensitiveType.CARD_NUMBER)
    window = RepairWindow(
        start_ms=0,
        end_ms=600,
        types=(SensitiveType.CARD_NUMBER,),
    )
    engine = RepairEngine(
        _CountingEmptyOcr(),
        detection_config=DetectionConfig(),
        tracking_config=TrackingConfig(),
        verification_config=VerificationConfig(repair_padding_px=12),
    )

    patched = engine.patch_leak_seeds(
        MaskTimeline(),
        (leak,),
        (window,),
        frame_rate=10,
        frame_count=10,
        pass_number=1,
    )

    assert set(patched.frames) == set(range(7))
    assert all(
        regions[0].bbox == leak.bbox.expanded(12, 12)
        for regions in patched.frames.values()
    )
