from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from redact_video.models import BoundingBox, OcrToken


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((600, 800, 3), dtype=np.uint8)


@pytest.fixture
def token_factory() -> Callable[..., OcrToken]:
    def create(
        text: str,
        x: int,
        y: int,
        width: int = 100,
        height: int = 22,
        confidence: float = 0.98,
    ) -> OcrToken:
        return OcrToken(
            text=text,
            confidence=confidence,
            bbox=BoundingBox(x=x, y=y, width=width, height=height),
        )

    return create
