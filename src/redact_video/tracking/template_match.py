"""Bounded in-memory template fallback for weak optical-flow tracks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import TrackingConfig
from redact_video.models import BoundingBox


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    bbox: BoundingBox
    confidence: float


class TemplateMatcher:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def match(
        self,
        previous_frame: NDArray[np.uint8],
        current_frame: NDArray[np.uint8],
        previous_bbox: BoundingBox,
        predicted_bbox: BoundingBox,
    ) -> TemplateMatch | None:
        if previous_frame.shape != current_frame.shape:
            return None
        return self.match_gray(
            cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY),
            previous_bbox,
            predicted_bbox,
        )

    def match_gray(
        self,
        previous_gray: NDArray[np.uint8],
        current_gray: NDArray[np.uint8],
        previous_bbox: BoundingBox,
        predicted_bbox: BoundingBox,
    ) -> TemplateMatch | None:
        """Match a template using cached grayscale frames."""

        previous_height, previous_width = previous_gray.shape[:2]
        current_height, current_width = current_gray.shape[:2]
        template_box = previous_bbox.clipped(previous_width, previous_height)
        predicted = predicted_bbox.clipped(current_width, current_height)
        if template_box is None or predicted is None:
            return None

        template = previous_gray[
            template_box.y : template_box.bottom,
            template_box.x : template_box.right,
        ]
        if template.size == 0 or float(template.std()) < 2.0:
            return None

        radius = max(
            self.config.minimum_template_search_px,
            round(
                max(template_box.width, template_box.height)
                * self.config.template_search_radius_ratio
            ),
        )
        search_left = max(0, predicted.x - radius)
        search_top = max(0, predicted.y - radius)
        search_right = min(current_width, predicted.right + radius)
        search_bottom = min(current_height, predicted.bottom + radius)
        if (
            search_right - search_left < template_box.width
            or search_bottom - search_top < template_box.height
        ):
            return None
        search = current_gray[search_top:search_bottom, search_left:search_right]
        try:
            scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None
        _, confidence, _, location = cv2.minMaxLoc(scores)
        if not math.isfinite(confidence):
            return None
        if confidence < self.config.minimum_template_confidence:
            return None
        return TemplateMatch(
            bbox=BoundingBox(
                x=search_left + location[0],
                y=search_top + location[1],
                width=template_box.width,
                height=template_box.height,
            ),
            confidence=float(min(1.0, confidence)),
        )
