"""Detector protocol and immutable per-frame context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from redact_video.models import OcrToken, SensitiveDetection, SensitiveType

if TYPE_CHECKING:
    from redact_video.detection.geometry import OcrSpan
    from redact_video.detection.payment_context import PaymentContextResult


@dataclass(frozen=True, slots=True)
class DetectionContext:
    frame_index: int
    timestamp_ms: int
    frame_width: int
    frame_height: int
    payment: PaymentContextResult
    ocr_spans: tuple[OcrSpan, ...]

    def spans(self, *, max_tokens: int) -> tuple[OcrSpan, ...]:
        """Return prebuilt spans up to the requested adjacent-token width."""

        return tuple(span for span in self.ocr_spans if span.token_count <= max_tokens)


@runtime_checkable
class SensitiveDetector(Protocol):
    @property
    def types(self) -> set[SensitiveType]: ...

    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]: ...
