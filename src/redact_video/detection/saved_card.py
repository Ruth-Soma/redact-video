"""Masked/last-four saved-card and associated expiry detection."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from redact_video.config import DetectionConfig
from redact_video.detection.base import DetectionContext
from redact_video.detection.common import make_detection
from redact_video.detection.expiry import expiry_candidates
from redact_video.detection.geometry import box_distance, text_subspan_box
from redact_video.models import OcrToken, SensitiveDetection, SensitiveType
from redact_video.ocr.normalizer import NumericField, normalize_numeric_candidate

_MASKED_CARD_PATTERN = re.compile(
    r"(?P<masked>(?:[*•●xX][\s-]*){2,}(?P<last>(?:\d[\s-]*){4}))(?!\d)"
)
_ENDING_IN_PATTERN = re.compile(r"(?i)ending\s+in\s+(?P<last>\d(?:[\s-]*\d){3})(?!\d)")


class SavedCardDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    @property
    def types(self) -> set[SensitiveType]:
        return {
            SensitiveType.SAVED_CARD_NUMBER,
            SensitiveType.SAVED_CARD_EXPIRY,
        }

    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]:
        del frame
        if not self.config.blur_saved_card:
            return []

        number_detections = self._number_detections(ocr, context)
        detections = list(number_detections)
        threshold = self.config.thresholds.saved_card
        for candidate in expiry_candidates(
            ocr,
            self.config,
            spans=context.ocr_spans,
        ):
            if not any(
                box_distance(candidate.bbox, saved.bbox)
                <= max(context.frame_height * 0.14, saved.bbox.width * 1.2)
                for saved in number_detections
            ):
                continue
            confidence = 0.80 * (0.80 + 0.20 * candidate.confidence)
            if confidence >= threshold:
                detections.append(
                    make_detection(
                        sensitive_type=SensitiveType.SAVED_CARD_EXPIRY,
                        bbox=candidate.bbox,
                        confidence=confidence,
                        source="saved_card_expiry",
                        context=context,
                    )
                )
        return detections

    def _number_detections(
        self,
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]:
        detections: list[SensitiveDetection] = []
        threshold = self.config.thresholds.saved_card
        seen: set[tuple[int, int, int, int]] = set()
        for span in context.ocr_spans:
            for match in _MASKED_CARD_PATTERN.finditer(span.text):
                bbox = text_subspan_box(span, *match.span("masked"))
                key = (bbox.x, bbox.y, bbox.width, bbox.height)
                if key in seen:
                    continue
                seen.add(key)
                confidence = 0.80 * (0.80 + 0.20 * span.confidence)
                if confidence >= threshold:
                    detections.append(
                        make_detection(
                            sensitive_type=SensitiveType.SAVED_CARD_NUMBER,
                            bbox=bbox,
                            confidence=confidence,
                            source="saved_card_mask",
                            context=context,
                        )
                    )

            for match in _ENDING_IN_PATTERN.finditer(span.text):
                bbox = text_subspan_box(span, *match.span("last"))
                key = (bbox.x, bbox.y, bbox.width, bbox.height)
                if key in seen:
                    continue
                seen.add(key)
                confidence = 0.78 * (0.80 + 0.20 * span.confidence)
                if confidence >= threshold:
                    detections.append(
                        make_detection(
                            sensitive_type=SensitiveType.SAVED_CARD_NUMBER,
                            bbox=bbox,
                            confidence=confidence,
                            source="saved_card_ending",
                            context=context,
                        )
                    )

            normalized = normalize_numeric_candidate(
                span.text,
                expected_field=NumericField.SAVED_CARD,
            )
            if normalized is None or not re.fullmatch(r"\*{2,}\d{4}", normalized):
                continue
            key = (
                span.bbox.x,
                span.bbox.y,
                span.bbox.width,
                span.bbox.height,
            )
            if key in seen:
                continue
            seen.add(key)
            confidence = 0.76 * (0.80 + 0.20 * span.confidence)
            if confidence >= threshold:
                detections.append(
                    make_detection(
                        sensitive_type=SensitiveType.SAVED_CARD_NUMBER,
                        bbox=span.bbox,
                        confidence=confidence,
                        source="saved_card_normalized_mask",
                        context=context,
                    )
                )
        return detections
