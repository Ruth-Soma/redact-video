"""Strongly semantic CVV/CVC/CSC/CID detection."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from redact_video.config import DetectionConfig
from redact_video.detection.base import DetectionContext
from redact_video.detection.common import (
    best_anchor_association,
    has_neighboring_payment_field,
    make_detection,
    semantic_anchor_detections,
)
from redact_video.detection.geometry import OcrSpan, build_ocr_spans, text_subspan_box
from redact_video.detection.payment_context import FieldRole
from redact_video.models import BoundingBox, OcrToken, SensitiveDetection, SensitiveType
from redact_video.ocr.normalizer import NumericField, normalize_numeric_candidate

_INLINE_SECURITY_CODE = re.compile(
    r"(?i)(?:cvv|cvc|csc|cid|security\s+code)\D{0,8}(?P<value>\d{1,4})(?!\d)"
)


@dataclass(frozen=True, slots=True)
class _CvvCandidate:
    digits: str = field(repr=False)
    bbox: BoundingBox
    confidence: float


def _cvv_candidates(
    ocr: list[OcrToken],
    config: DetectionConfig,
    *,
    spans: Iterable[OcrSpan] | None = None,
) -> list[_CvvCandidate]:
    candidates: list[_CvvCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    active_spans = (
        build_ocr_spans(ocr, max_tokens=config.max_ocr_span_tokens)
        if spans is None
        else spans
    )
    for span in active_spans:
        normalized = normalize_numeric_candidate(
            span.text,
            expected_field=NumericField.SECURITY_CODE,
        )
        if normalized is not None and 1 <= len(normalized) <= 4:
            key = (span.bbox.x, span.bbox.y, span.bbox.width, span.bbox.height)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    _CvvCandidate(
                        digits=normalized,
                        bbox=span.bbox,
                        confidence=span.confidence,
                    )
                )
        for match in _INLINE_SECURITY_CODE.finditer(span.text):
            start, end = match.span("value")
            bbox = text_subspan_box(span, start, end)
            key = (bbox.x, bbox.y, bbox.width, bbox.height)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                _CvvCandidate(
                    digits=match.group("value"),
                    bbox=bbox,
                    confidence=span.confidence,
                )
            )
    return candidates


class CvvDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    @property
    def types(self) -> set[SensitiveType]:
        return {SensitiveType.CARD_CVV}

    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]:
        del frame
        threshold = self.config.thresholds.cvv
        detections = semantic_anchor_detections(
            roles={FieldRole.CVV},
            sensitive_type=SensitiveType.CARD_CVV,
            threshold=threshold,
            context=context,
        )
        anchors = context.payment.anchors_for(FieldRole.CVV)
        if not anchors:
            return detections

        for candidate in _cvv_candidates(
            ocr,
            self.config,
            spans=context.ocr_spans,
        ):
            anchor, geometry = best_anchor_association(candidate.bbox, anchors)
            if anchor is None or geometry < 0.65:
                continue
            structure = 1.0 if len(candidate.digits) in {3, 4} else 0.72
            confidence = (
                self.config.weights.semantic_label
                + self.config.weights.payment_context * context.payment.score
                + self.config.weights.numeric_structure * structure
                + self.config.weights.field_geometry * geometry
                + (
                    self.config.weights.neighboring_field
                    if has_neighboring_payment_field(context, excluding={FieldRole.CVV})
                    else 0.0
                )
            )
            confidence = max(confidence, anchor.confidence * 0.84)
            confidence *= 0.75 + 0.25 * candidate.confidence
            if confidence >= threshold:
                detections.append(
                    make_detection(
                        sensitive_type=SensitiveType.CARD_CVV,
                        bbox=candidate.bbox,
                        confidence=confidence,
                        source="semantic_cvv_value",
                        context=context,
                    )
                )
        return detections
