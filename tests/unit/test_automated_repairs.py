from redact_video.models import BoundingBox, SensitiveType
from redact_video.pipeline.automated import _build_opaque_repair_overlays
from redact_video.verification import RepairWindow, VerificationLeak


def _leak(timestamp_ms: int, bbox: BoundingBox) -> VerificationLeak:
    return VerificationLeak(
        type=SensitiveType.CARD_NUMBER,
        frame_index=timestamp_ms // 10,
        timestamp_ms=timestamp_ms,
        bbox=bbox,
        confidence=0.9,
        source="pattern_luhn",
    )


def test_opaque_repair_overlay_unions_and_clips_window_leaks() -> None:
    overlays = _build_opaque_repair_overlays(
        (
            _leak(100, BoundingBox(x=2, y=10, width=20, height=10)),
            _leak(200, BoundingBox(x=30, y=15, width=20, height=10)),
        ),
        (
            RepairWindow(
                start_ms=0,
                end_ms=500,
                types=(SensitiveType.CARD_NUMBER,),
            ),
        ),
        frame_width=60,
        frame_height=40,
        padding_px=5,
    )

    assert len(overlays) == 1
    assert overlays[0].start_ms == 0
    assert overlays[0].end_ms == 500
    assert overlays[0].bbox == BoundingBox(x=0, y=5, width=55, height=25)
