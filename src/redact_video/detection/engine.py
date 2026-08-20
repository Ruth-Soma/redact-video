"""Card-detector orchestration and deterministic region deduplication."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from redact_video.config import DetectionConfig
from redact_video.detection.base import DetectionContext, SensitiveDetector
from redact_video.detection.card_number import CardNumberDetector
from redact_video.detection.cvv import CvvDetector
from redact_video.detection.expiry import ExpiryDetector
from redact_video.detection.geometry import (
    build_ocr_spans,
    overlap_fraction,
    union_boxes,
)
from redact_video.detection.payment_context import PaymentContextDetector
from redact_video.detection.saved_card import SavedCardDetector
from redact_video.models import OcrToken, SensitiveDetection, SensitiveType


def _merge_same_type(
    detections: list[SensitiveDetection],
) -> list[SensitiveDetection]:
    selected: list[SensitiveDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        match_index = next(
            (
                index
                for index, existing in enumerate(selected)
                if existing.type is detection.type
                and overlap_fraction(existing.bbox, detection.bbox) >= 0.45
            ),
            None,
        )
        if match_index is None:
            selected.append(detection)
            continue
        existing = selected[match_index]
        merged_source = (
            "semantic_anchor"
            if existing.source == detection.source == "semantic_anchor"
            else "merged"
        )
        selected[match_index] = SensitiveDetection(
            type=existing.type,
            frame_index=existing.frame_index,
            timestamp_ms=existing.timestamp_ms,
            bbox=union_boxes([existing.bbox, detection.bbox]),
            confidence=max(existing.confidence, detection.confidence),
            source=merged_source,
        )
    return selected


def _prefer_saved_card_types(
    detections: list[SensitiveDetection],
) -> list[SensitiveDetection]:
    saved_number_boxes = [
        item.bbox for item in detections if item.type is SensitiveType.SAVED_CARD_NUMBER
    ]
    saved_expiry_boxes = [
        item.bbox for item in detections if item.type is SensitiveType.SAVED_CARD_EXPIRY
    ]
    selected: list[SensitiveDetection] = []
    for detection in detections:
        if detection.type is SensitiveType.CARD_NUMBER and any(
            overlap_fraction(detection.bbox, saved) >= 0.45
            for saved in saved_number_boxes
        ):
            continue
        if detection.type is SensitiveType.CARD_EXPIRY and any(
            overlap_fraction(detection.bbox, saved) >= 0.45
            for saved in saved_expiry_boxes
        ):
            continue
        selected.append(detection)
    return selected


class DetectionEngine:
    """Run plugin-style sensitive detectors over one OCR-scanned frame."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        *,
        detectors: Sequence[SensitiveDetector] | None = None,
        payment_context_detector: PaymentContextDetector | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self.payment_context_detector = (
            payment_context_detector or PaymentContextDetector(self.config)
        )
        self.detectors: tuple[SensitiveDetector, ...] = tuple(
            (
                CardNumberDetector(self.config),
                ExpiryDetector(self.config),
                CvvDetector(self.config),
                SavedCardDetector(self.config),
            )
            if detectors is None
            else detectors
        )

    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Detection expects an 8-bit, three-channel BGR frame")
        if frame_index < 0 or timestamp_ms < 0:
            raise ValueError("Frame index and timestamp must be non-negative")
        frame_height, frame_width = frame.shape[:2]
        ocr_spans = tuple(
            build_ocr_spans(ocr, max_tokens=self.config.max_ocr_span_tokens)
        )
        payment = self.payment_context_detector.analyze(
            ocr,
            frame_width=frame_width,
            frame_height=frame_height,
            spans=ocr_spans,
        )
        context = DetectionContext(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            frame_width=frame_width,
            frame_height=frame_height,
            payment=payment,
            ocr_spans=ocr_spans,
        )
        detections: list[SensitiveDetection] = []
        for detector in self.detectors:
            detections.extend(detector.detect(frame, ocr, context))
        deduplicated = _prefer_saved_card_types(_merge_same_type(detections))
        return sorted(
            deduplicated,
            key=lambda item: (item.bbox.y, item.bbox.x, item.type.value),
        )
