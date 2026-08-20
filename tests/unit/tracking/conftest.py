from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from redact_video.models import BoundingBox, SensitiveDetection, SensitiveType


@pytest.fixture
def tracking_frame_factory() -> Callable[..., tuple[np.ndarray, BoundingBox]]:
    def create(
        dx: int = 0,
        dy: int = 0,
        *,
        inverted: bool = False,
    ) -> tuple[np.ndarray, BoundingBox]:
        background = 235 if inverted else 20
        foreground = 20 if inverted else 235
        frame = np.full((240, 360, 3), background, dtype=np.uint8)
        box = BoundingBox(x=90 + dx, y=85 + dy, width=130, height=42)
        cv2.rectangle(
            frame,
            (box.x, box.y),
            (box.right - 1, box.bottom - 1),
            (foreground, foreground, foreground),
            2,
        )
        cv2.putText(
            frame,
            "FIELD",
            (box.x + 8, box.y + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (foreground, foreground, foreground),
            2,
            cv2.LINE_AA,
        )
        for x_offset in (0, box.width - 1):
            for y_offset in (0, box.height - 1):
                cv2.circle(
                    frame,
                    (box.x + x_offset, box.y + y_offset),
                    3,
                    (foreground, foreground, foreground),
                    -1,
                )
        return frame, box

    return create


@pytest.fixture
def detection_factory() -> Callable[..., SensitiveDetection]:
    def create(
        bbox: BoundingBox,
        *,
        frame_index: int = 0,
        timestamp_ms: int = 0,
        sensitive_type: SensitiveType = SensitiveType.CARD_NUMBER,
        confidence: float = 0.96,
    ) -> SensitiveDetection:
        return SensitiveDetection(
            type=sensitive_type,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            bbox=bbox,
            confidence=confidence,
            source="synthetic",
        )

    return create
