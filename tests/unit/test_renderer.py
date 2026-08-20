import cv2
import numpy as np
import pytest

from redact_video.config import BlurConfig
from redact_video.models import BoundingBox, SensitiveType
from redact_video.redaction.renderer import BlurRenderer, _gaussian_box_blur_sizes
from redact_video.redaction.timeline import MaskRegion, MaskTimeline


def test_privacy_blur_reduces_detail_and_keeps_distant_pixels() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for y in range(30, 80, 2):
        cv2.line(frame, (40, y), (119, y), (255, 255, 255), 1)
    original = frame.copy()
    renderer = BlurRenderer(
        BlurConfig(
            gaussian_sigma=8,
            min_padding_px=6,
            padding_ratio=0,
            pixel_block_size=12,
        )
    )

    output = renderer.apply(frame, [BoundingBox(x=40, y=30, width=80, height=50)])

    assert np.array_equal(output[:20, :20], original[:20, :20])
    assert output[30:80, 40:120].std() < original[30:80, 40:120].std()
    assert not np.array_equal(output[30:80, 40:120], original[30:80, 40:120])


def test_renderer_can_modify_a_frame_in_place() -> None:
    frame = np.random.default_rng(3).integers(0, 256, (80, 80, 3), dtype=np.uint8)

    output = BlurRenderer().apply(
        frame,
        [BoundingBox(x=20, y=20, width=40, height=20)],
        in_place=True,
    )

    assert output is frame


def test_box_blur_passes_approximate_configured_gaussian_variance() -> None:
    sizes = _gaussian_box_blur_sizes(8)
    variance = sum((size * size - 1) / 12 for size in sizes)

    assert len(sizes) == 3
    assert all(size % 2 == 1 for size in sizes)
    assert variance**0.5 == pytest.approx(8, abs=0.25)


def test_renderer_applies_only_the_requested_timeline_frame() -> None:
    frame = np.random.default_rng(7).integers(0, 256, (80, 80, 3), dtype=np.uint8)
    timeline = MaskTimeline(
        frames={
            4: (
                MaskRegion(
                    track_id="track-000001",
                    type=SensitiveType.CARD_NUMBER,
                    bbox=BoundingBox(x=20, y=20, width=40, height=20),
                ),
            )
        }
    )

    unchanged = BlurRenderer().apply_timeline(frame, timeline, frame_index=3)
    masked = BlurRenderer().apply_timeline(frame, timeline, frame_index=4)

    assert np.array_equal(unchanged, frame)
    assert not np.array_equal(masked, frame)
