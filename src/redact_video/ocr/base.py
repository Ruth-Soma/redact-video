"""OCR boundary consumed by detectors and scanning pipelines."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from redact_video.models import OcrToken


@runtime_checkable
class OcrEngine(Protocol):
    """Frame-oriented OCR interface independent of any vendor runtime."""

    def scan(self, frame: NDArray[np.uint8]) -> list[OcrToken]:
        """Recognize transient text tokens in one display-oriented BGR frame."""
        ...
