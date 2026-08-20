from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from redact_video.config import TrackingConfig
from redact_video.models import BoundingBox
from redact_video.tracking.optical_flow import OpticalFlowEstimator
from redact_video.tracking.scene_change import SceneChangeDetector
from redact_video.tracking.template_match import TemplateMatcher


def test_lucas_kanade_estimates_vertical_translation(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    previous, box = tracking_frame_factory()
    current, _ = tracking_frame_factory(dy=9)

    estimate = OpticalFlowEstimator().estimate(previous, current, box)

    assert estimate is not None
    assert abs(estimate.dx) < 1.5
    assert abs(estimate.dy - 9) < 1.5
    assert estimate.confidence >= TrackingConfig().minimum_flow_confidence
    assert estimate.point_count >= TrackingConfig().minimum_flow_points


def test_default_lucas_kanade_pyramid_handles_large_frame_translation(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    previous, box = tracking_frame_factory()
    current, _ = tracking_frame_factory(dy=55)

    estimate = OpticalFlowEstimator().estimate(previous, current, box)

    assert estimate is not None
    assert abs(estimate.dx) < 1.5
    assert abs(estimate.dy - 55) < 1.5
    assert estimate.confidence >= TrackingConfig().minimum_flow_confidence


def test_lucas_kanade_limits_feature_search_to_a_local_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = np.zeros((1600, 1200, 3), dtype=np.uint8)
    current = np.zeros_like(previous)
    box = BoundingBox(x=480, y=700, width=240, height=90)
    rng = np.random.default_rng(42)
    patch = rng.integers(0, 256, (box.height, box.width, 3), dtype=np.uint8)
    previous[box.y : box.bottom, box.x : box.right] = patch
    current[box.y + 8 : box.bottom + 8, box.x : box.right] = patch
    observed_shapes: list[tuple[int, int]] = []
    original = cv2.goodFeaturesToTrack

    def capture_shape(image: np.ndarray, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        observed_shapes.append(image.shape)
        return original(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", capture_shape)

    estimate = OpticalFlowEstimator().estimate(previous, current, box)

    assert estimate is not None
    assert abs(estimate.dy - 8) < 1.5
    assert observed_shapes
    roi_height, roi_width = observed_shapes[0]
    assert roi_height < previous.shape[0]
    assert roi_width < previous.shape[1]


def test_template_matching_finds_region_near_velocity_prediction(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    previous, box = tracking_frame_factory()
    current, expected = tracking_frame_factory(dx=18, dy=24)
    predicted = BoundingBox(
        x=box.x + 15,
        y=box.y + 20,
        width=box.width,
        height=box.height,
    )

    match = TemplateMatcher().match(previous, current, box, predicted)

    assert match is not None
    assert abs(match.bbox.x - expected.x) <= 1
    assert abs(match.bbox.y - expected.y) <= 1
    assert match.confidence >= TrackingConfig().minimum_template_confidence


def test_template_matching_rejects_textureless_sensitive_patch() -> None:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    box = BoundingBox(x=30, y=30, width=60, height=20)

    assert TemplateMatcher().match(frame, frame.copy(), box, box) is None


def test_scene_change_ignores_local_motion_but_detects_screen_replacement(
    tracking_frame_factory: Callable[..., tuple[np.ndarray, BoundingBox]],
) -> None:
    previous, _ = tracking_frame_factory()
    local_motion, _ = tracking_frame_factory(dy=6)
    replacement = np.full_like(previous, 245)
    detector = SceneChangeDetector()

    local_result = detector.compare(previous, local_motion)
    replacement_result = detector.compare(previous, replacement)

    assert local_result.changed is False
    assert replacement_result.changed is True
    assert replacement_result.score > local_result.score


def test_resolution_change_is_always_a_scene_change() -> None:
    previous = np.zeros((120, 180, 3), dtype=np.uint8)
    current = np.zeros((180, 120, 3), dtype=np.uint8)

    result = SceneChangeDetector().compare(previous, current)

    assert result.changed is True
    assert result.score == 1.0
