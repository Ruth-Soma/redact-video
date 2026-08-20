"""Context-gated card-expiry and split month/year detection."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from redact_video.config import DetectionConfig
from redact_video.detection.base import DetectionContext
from redact_video.detection.card_number import luhn_pan_boxes
from redact_video.detection.common import (
    best_anchor_association,
    make_detection,
    semantic_anchor_detections,
)
from redact_video.detection.geometry import (
    OcrSpan,
    box_distance,
    build_ocr_spans,
    text_subspan_box,
)
from redact_video.detection.payment_context import FieldAnchor, FieldRole
from redact_video.models import BoundingBox, OcrToken, SensitiveDetection, SensitiveType
from redact_video.ocr.normalizer import NumericField, normalize_numeric_candidate

_EXPIRY_PATTERN = re.compile(
    r"(?<!\d)(?P<month>0[1-9]|1[0-2])\s*[/\-]\s*(?P<year>\d{2}|\d{4})(?!\d)"
)


@dataclass(frozen=True, slots=True)
class ExpiryCandidate:
    canonical: str = field(repr=False)
    bbox: BoundingBox
    confidence: float


def _valid_expiry_canonical(value: str) -> bool:
    compact = value.replace("/", "")
    return (
        len(compact) in {4, 6}
        and compact.isdigit()
        and len(compact[:2]) == 2
        and 1 <= int(compact[:2]) <= 12
        and len(compact[2:]) in {2, 4}
    )


def expiry_candidates(
    ocr: list[OcrToken],
    config: DetectionConfig,
    *,
    spans: Iterable[OcrSpan] | None = None,
) -> list[ExpiryCandidate]:
    candidates: list[ExpiryCandidate] = []
    seen: set[tuple[int, int, int, int]] = set()
    active_spans = (
        build_ocr_spans(ocr, max_tokens=config.max_ocr_span_tokens)
        if spans is None
        else spans
    )
    for span in active_spans:
        normalized = normalize_numeric_candidate(
            span.text,
            expected_field=NumericField.EXPIRY,
        )
        if normalized is not None and _valid_expiry_canonical(normalized):
            key = (span.bbox.x, span.bbox.y, span.bbox.width, span.bbox.height)
            if key not in seen:
                seen.add(key)
                candidates.append(
                    ExpiryCandidate(
                        canonical=normalized,
                        bbox=span.bbox,
                        confidence=span.confidence,
                    )
                )

        raw = unicodedata.normalize("NFKC", span.text)
        for match in _EXPIRY_PATTERN.finditer(raw):
            bbox = text_subspan_box(span, match.start(), match.end())
            key = (bbox.x, bbox.y, bbox.width, bbox.height)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ExpiryCandidate(
                    canonical=f"{match.group('month')}/{match.group('year')}",
                    bbox=bbox,
                    confidence=span.confidence,
                )
            )
    return candidates


def _near_card_field(candidate: BoundingBox, anchors: tuple[FieldAnchor, ...]) -> bool:
    return any(
        box_distance(candidate, anchor.value_bbox)
        <= max(anchor.value_bbox.height * 8, anchor.value_bbox.width * 0.55)
        for anchor in anchors
    )


def _near_valid_pan(
    candidate: BoundingBox,
    pan_boxes: tuple[BoundingBox, ...],
    *,
    frame_height: int,
) -> bool:
    return any(
        box_distance(candidate, pan_box)
        <= max(frame_height * 0.12, pan_box.width * 0.55)
        for pan_box in pan_boxes
    )


class ExpiryDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    @property
    def types(self) -> set[SensitiveType]:
        return {SensitiveType.CARD_EXPIRY}

    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]:
        del frame
        threshold = self.config.thresholds.expiry
        expiry_roles = {
            FieldRole.EXPIRY,
            FieldRole.EXPIRY_MONTH,
            FieldRole.EXPIRY_YEAR,
        }
        detections = semantic_anchor_detections(
            roles=expiry_roles,
            sensitive_type=SensitiveType.CARD_EXPIRY,
            threshold=threshold,
            context=context,
        )
        direct_anchors = context.payment.anchors_for(FieldRole.EXPIRY)
        card_anchors = context.payment.anchors_for(FieldRole.CARD_NUMBER)
        valid_pan_boxes = luhn_pan_boxes(
            ocr,
            self.config,
            spans=context.ocr_spans,
        )

        for candidate in expiry_candidates(
            ocr,
            self.config,
            spans=context.ocr_spans,
        ):
            anchor, geometry = best_anchor_association(candidate.bbox, direct_anchors)
            if anchor is not None and geometry >= 0.65:
                confidence = (
                    self.config.weights.semantic_label
                    + self.config.weights.payment_context * context.payment.score
                    + self.config.weights.numeric_structure
                    + self.config.weights.field_geometry * geometry
                    + (self.config.weights.neighboring_field if card_anchors else 0.0)
                )
                confidence = max(confidence, anchor.confidence * 0.84)
                confidence *= 0.75 + 0.25 * candidate.confidence
                if confidence >= threshold:
                    detections.append(
                        make_detection(
                            sensitive_type=SensitiveType.CARD_EXPIRY,
                            bbox=candidate.bbox,
                            confidence=confidence,
                            source="semantic_expiry_value",
                            context=context,
                        )
                    )
                continue

            if _near_card_field(candidate.bbox, card_anchors) or _near_valid_pan(
                candidate.bbox,
                valid_pan_boxes,
                frame_height=context.frame_height,
            ):
                confidence = 0.74 * (0.80 + 0.20 * candidate.confidence)
                if confidence >= threshold:
                    detections.append(
                        make_detection(
                            sensitive_type=SensitiveType.CARD_EXPIRY,
                            bbox=candidate.bbox,
                            confidence=confidence,
                            source="payment_proximity_expiry",
                            context=context,
                        )
                    )

        detections.extend(self._split_field_values(ocr, context, threshold))
        return detections

    def _split_field_values(
        self,
        ocr: list[OcrToken],
        context: DetectionContext,
        threshold: float,
    ) -> list[SensitiveDetection]:
        month_anchors = context.payment.anchors_for(FieldRole.EXPIRY_MONTH)
        year_anchors = context.payment.anchors_for(FieldRole.EXPIRY_YEAR)
        if not month_anchors and not year_anchors:
            return []

        detections: list[SensitiveDetection] = []
        for span in context.spans(max_tokens=1):
            value = normalize_numeric_candidate(
                span.text,
                expected_field=NumericField.EXPIRY,
            )
            if value is None or "/" in value:
                continue
            role_anchors: tuple[FieldAnchor, ...] = ()
            if value.isdigit() and len(value) == 2 and 1 <= int(value) <= 12:
                role_anchors = month_anchors
            if value.isdigit() and len(value) in {2, 4}:
                _, year_score = best_anchor_association(span.bbox, year_anchors)
                _, month_score = best_anchor_association(span.bbox, month_anchors)
                if year_score > month_score:
                    role_anchors = year_anchors
            anchor, geometry = best_anchor_association(span.bbox, role_anchors)
            if anchor is None or geometry < 0.65:
                continue
            confidence = max(0.0, anchor.confidence * 0.84)
            confidence *= 0.75 + 0.25 * span.confidence
            if confidence >= threshold:
                detections.append(
                    make_detection(
                        sensitive_type=SensitiveType.CARD_EXPIRY,
                        bbox=span.bbox,
                        confidence=confidence,
                        source="semantic_split_expiry",
                        context=context,
                    )
                )
        return detections
