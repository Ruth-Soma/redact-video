"""Strong pixelation-plus-Gaussian-equivalent privacy blur."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import BlurConfig
from redact_video.models import BoundingBox

if TYPE_CHECKING:
    from redact_video.redaction.timeline import MaskTimeline


class BlurRenderer:
    """Apply deterministic, safety-padded privacy masks to BGR frames."""

    def __init__(self, config: BlurConfig | None = None) -> None:
        self.config = config or BlurConfig()
        self._box_blur_sizes = _gaussian_box_blur_sizes(self.config.gaussian_sigma)

    def apply(
        self,
        frame: NDArray[np.uint8],
        boxes: list[BoundingBox],
        *,
        in_place: bool = False,
    ) -> NDArray[np.uint8]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Renderer expects an 8-bit, three-channel BGR frame")
        rendered = frame if in_place else frame.copy()
        frame_height, frame_width = rendered.shape[:2]
        padding = self.config.padding_for(frame_width, frame_height)
        for box in boxes:
            padded = box.expanded(padding, padding).clipped(frame_width, frame_height)
            if padded is None:
                continue
            roi = rendered[padded.y : padded.bottom, padded.x : padded.right]
            small_width = max(1, padded.width // self.config.pixel_block_size)
            small_height = max(1, padded.height // self.config.pixel_block_size)
            reduced = cv2.resize(
                roi,
                (small_width, small_height),
                interpolation=cv2.INTER_AREA,
            )
            pixelated = cv2.resize(
                reduced,
                (padded.width, padded.height),
                interpolation=cv2.INTER_NEAREST,
            )
            # Three box filters approximate a Gaussian to the configured
            # sigma while using OpenCV's constant-time box-filter path. This
            # preserves the strong pixelation-plus-softening transform without
            # paying for a wide Gaussian kernel independently in every region.
            safe = pixelated
            for blur_size in self._box_blur_sizes:
                if blur_size > 1:
                    safe = cv2.blur(safe, (blur_size, blur_size))
            rendered[padded.y : padded.bottom, padded.x : padded.right] = safe
        return rendered

    def apply_timeline(
        self,
        frame: NDArray[np.uint8],
        timeline: MaskTimeline,
        *,
        frame_index: int,
        in_place: bool = False,
    ) -> NDArray[np.uint8]:
        """Apply the mask regions assigned to one timeline frame."""

        return self.apply(
            frame,
            timeline.boxes_for(frame_index),
            in_place=in_place,
        )


def _gaussian_box_blur_sizes(sigma: float, passes: int = 3) -> tuple[int, ...]:
    """Return odd box widths whose combined variance approximates a Gaussian."""

    ideal_width = math.sqrt((12 * sigma * sigma / passes) + 1)
    lower_width = math.floor(ideal_width)
    if lower_width % 2 == 0:
        lower_width -= 1
    lower_width = max(1, lower_width)
    upper_width = lower_width + 2
    lower_count = round(
        (
            (12 * sigma * sigma)
            - (passes * lower_width * lower_width)
            - (4 * passes * lower_width)
            - (3 * passes)
        )
        / (-4 * lower_width - 4)
    )
    lower_count = max(0, min(passes, lower_count))
    return (lower_width,) * lower_count + (upper_width,) * (passes - lower_count)
