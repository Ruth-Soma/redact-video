import pytest
from pydantic import ValidationError

from redact_video.config import TrackingConfig
from redact_video.models import BoundingBox, SensitiveTrack, SensitiveType, TrackState


def test_bounding_box_expands_and_clips_at_frame_edges() -> None:
    box = BoundingBox(x=2, y=3, width=10, height=8)

    expanded = box.expanded(6, 6)
    clipped = expanded.clipped(12, 10)

    assert clipped == BoundingBox(x=0, y=0, width=12, height=10)


def test_bounding_box_returns_none_outside_frame() -> None:
    box = BoundingBox(x=200, y=200, width=10, height=10)

    assert box.clipped(100, 100) is None


def test_sensitive_track_rejects_an_inconsistent_timeline() -> None:
    with pytest.raises(ValidationError, match="precede"):
        SensitiveTrack(
            id="track-000001",
            type=SensitiveType.CARD_NUMBER,
            start_frame=5,
            confidence=0.9,
            regions={4: BoundingBox(x=10, y=10, width=20, height=10)},
            state=TrackState.DETECTED,
        )


def test_tracking_config_rejects_inverted_flow_point_bounds() -> None:
    with pytest.raises(ValidationError, match="maximum_flow_points"):
        TrackingConfig(maximum_flow_points=4, minimum_flow_points=5)
