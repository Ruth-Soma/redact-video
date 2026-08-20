"""Lucas-Kanade translation estimates around sensitive regions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import TrackingConfig
from redact_video.models import BoundingBox


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    dx: float
    dy: float
    scale: float
    confidence: float
    point_count: int


def _gray(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class OpticalFlowEstimator:
    """Estimate robust local translation using forward/backward LK flow."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def estimate(
        self,
        previous_frame: NDArray[np.uint8],
        current_frame: NDArray[np.uint8],
        bbox: BoundingBox,
    ) -> MotionEstimate | None:
        if previous_frame.shape != current_frame.shape:
            return None
        return self.estimate_gray(
            _gray(previous_frame),
            _gray(current_frame),
            bbox,
        )

    def estimate_gray(
        self,
        previous_gray: NDArray[np.uint8],
        current_gray: NDArray[np.uint8],
        bbox: BoundingBox,
    ) -> MotionEstimate | None:
        """Estimate motion from cached grayscale frames within a bounded ROI."""

        if (
            previous_gray.shape != current_gray.shape
            or previous_gray.dtype != np.uint8
            or current_gray.dtype != np.uint8
            or previous_gray.ndim != 2
        ):
            return None
        height, width = previous_gray.shape
        margin = round(max(bbox.width, bbox.height) * self.config.feature_margin_ratio)
        feature_box = bbox.expanded(margin, margin).clipped(width, height)
        if feature_box is None:
            return None

        # Retain half a search window at the coarsest pyramid level around
        # feature points without constructing pyramids for the complete frame.
        flow_margin = math.ceil(
            self.config.flow_window_size * 2**self.config.flow_pyramid_levels / 2
        )
        flow_box = feature_box.expanded(flow_margin, flow_margin).clipped(
            width,
            height,
        )
        if flow_box is None:  # pragma: no cover - feature_box is already clipped
            return None
        previous_roi = previous_gray[
            flow_box.y : flow_box.bottom,
            flow_box.x : flow_box.right,
        ]
        current_roi = current_gray[
            flow_box.y : flow_box.bottom,
            flow_box.x : flow_box.right,
        ]
        local_feature_box = BoundingBox(
            x=feature_box.x - flow_box.x,
            y=feature_box.y - flow_box.y,
            width=feature_box.width,
            height=feature_box.height,
        )
        mask = np.zeros(previous_roi.shape, dtype=np.uint8)
        mask[
            local_feature_box.y : local_feature_box.bottom,
            local_feature_box.x : local_feature_box.right,
        ] = 255
        try:
            points = cv2.goodFeaturesToTrack(
                previous_roi,
                mask=mask,
                maxCorners=self.config.maximum_flow_points,
                qualityLevel=self.config.flow_quality_level,
                minDistance=self.config.minimum_feature_distance_px,
                blockSize=5,
            )
        except cv2.error:
            return None
        if points is None or len(points) < self.config.minimum_flow_points:
            return None

        window = (self.config.flow_window_size, self.config.flow_window_size)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        try:
            next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
                previous_roi,
                current_roi,
                points,
                None,
                winSize=window,
                maxLevel=self.config.flow_pyramid_levels,
                criteria=criteria,
            )
        except cv2.error:
            return None
        if next_points is None or forward_status is None:
            return None

        valid = forward_status.reshape(-1).astype(bool)
        if forward_error is not None:
            valid &= np.isfinite(forward_error.reshape(-1))
            valid &= forward_error.reshape(-1) <= self.config.maximum_flow_error
        valid &= np.isfinite(next_points.reshape(-1, 2)).all(axis=1)
        if valid.sum() < self.config.minimum_flow_points:
            return None

        original = points.reshape(-1, 2)[valid]
        moved = next_points.reshape(-1, 2)[valid]
        try:
            back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
                current_roi,
                previous_roi,
                moved.reshape(-1, 1, 2),
                None,
                winSize=window,
                maxLevel=self.config.flow_pyramid_levels,
                criteria=criteria,
            )
        except cv2.error:
            return None
        if back_points is None or back_status is None:
            return None

        backward = back_points.reshape(-1, 2)
        backward_valid = back_status.reshape(-1).astype(bool)
        backward_valid &= np.isfinite(backward).all(axis=1)
        backward_error = np.linalg.norm(backward - original, axis=1)
        backward_valid &= backward_error <= self.config.maximum_forward_backward_error
        if backward_valid.sum() < self.config.minimum_flow_points:
            return None

        displacements = moved[backward_valid] - original[backward_valid]
        dx, dy = np.median(displacements, axis=0)
        accepted_error = backward_error[backward_valid]
        retention = float(backward_valid.sum() / len(points))
        support = min(
            1.0,
            float(backward_valid.sum() / (self.config.minimum_flow_points * 2)),
        )
        consistency = max(
            0.0,
            1.0
            - float(np.median(accepted_error))
            / self.config.maximum_forward_backward_error,
        )
        confidence = min(1.0, 0.40 * retention + 0.25 * support + 0.35 * consistency)
        return MotionEstimate(
            dx=float(dx),
            dy=float(dy),
            scale=1.0,
            confidence=confidence,
            point_count=int(backward_valid.sum()),
        )
