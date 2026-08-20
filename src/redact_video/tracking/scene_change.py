"""Cheap downscaled scene-change detection for adaptive OCR scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import TrackingConfig


@dataclass(frozen=True, slots=True)
class SceneChangeResult:
    changed: bool
    score: float


class SceneChangeDetector:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()

    def compare(
        self,
        previous_frame: NDArray[np.uint8],
        current_frame: NDArray[np.uint8],
    ) -> SceneChangeResult:
        if previous_frame.shape != current_frame.shape:
            return SceneChangeResult(changed=True, score=1.0)
        return self.compare_gray(
            self.prepare(previous_frame),
            self.prepare(current_frame),
        )

    def prepare(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Build the compact grayscale representation used by scene checks."""

        size = (self.config.scene_change_size, self.config.scene_change_size)
        # This guard detects broad scene replacement, not fine text changes.
        # Nearest-neighbor sampling avoids reading and averaging every source
        # pixel merely to produce a 64x64 scheduling signal.
        reduced = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)

    def compare_gray(
        self,
        previous_gray: NDArray[np.uint8],
        current_gray: NDArray[np.uint8],
    ) -> SceneChangeResult:
        """Compare cached grayscale frames at the configured small resolution."""

        if previous_gray.shape != current_gray.shape:
            return SceneChangeResult(changed=True, score=1.0)
        size = (self.config.scene_change_size, self.config.scene_change_size)
        previous = (
            previous_gray
            if previous_gray.shape == size
            else cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
        )
        current = (
            current_gray
            if current_gray.shape == size
            else cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
        )
        difference = cv2.absdiff(previous, current)
        score = float(np.mean(difference >= self.config.scene_pixel_delta))
        return SceneChangeResult(
            changed=score >= self.config.scene_change_fraction,
            score=score,
        )
