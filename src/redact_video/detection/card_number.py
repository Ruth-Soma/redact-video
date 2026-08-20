"""Full, partial, semantic-field, and Luhn-backed PAN detection."""

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
from redact_video.detection.luhn import luhn_valid
from redact_video.detection.payment_context import FieldRole
from redact_video.models import BoundingBox, OcrToken, SensitiveDetection, SensitiveType
from redact_video.ocr.normalizer import NumericField, normalize_numeric_candidate

_INLINE_CARD_NUMBER = re.compile(
    r"(?i)(?:credit\s+card\s+number|debit\s+card\s+number|card\s+number)"
    r"\D{0,8}(?P<value>[0-9OIlS](?:[0-9OIlS -]*[0-9OIlS])?)"
)


@dataclass(frozen=True, slots=True)
class _PanCandidate:
    digits: str = field(repr=False)
    span: OcrSpan


def luhn_pan_boxes(
    ocr: list[OcrToken],
    config: DetectionConfig,
    *,
    spans: Iterable[OcrSpan] | None = None,
) -> tuple[BoundingBox, ...]:
    """Return geometry only for full, Luhn-valid PAN candidates."""

    detector = CardNumberDetector(config)
    return tuple(
        candidate.span.bbox
        for candidate in detector._candidates(ocr, spans=spans)
        if 13 <= len(candidate.digits) <= 19 and luhn_valid(candidate.digits)
    )


class CardNumberDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    @property
    def types(self) -> set[SensitiveType]:
        return {SensitiveType.CARD_NUMBER}

    def _candidates(
        self,
        ocr: list[OcrToken],
        *,
        spans: Iterable[OcrSpan] | None = None,
    ) -> list[_PanCandidate]:
        candidates: list[_PanCandidate] = []
        seen: set[tuple[int, int, int, int, str]] = set()
        active_spans = (
            build_ocr_spans(ocr, max_tokens=self.config.max_ocr_span_tokens)
            if spans is None
            else spans
        )
        for span in active_spans:
            candidate_spans = [span]
            for match in _INLINE_CARD_NUMBER.finditer(span.text):
                value_start, value_end = match.span("value")
                candidate_spans.append(
                    OcrSpan(
                        text=match.group("value"),
                        confidence=span.confidence,
                        bbox=text_subspan_box(span, value_start, value_end),
                        token_count=span.token_count,
                    )
                )
            for candidate_span in candidate_spans:
                digits = normalize_numeric_candidate(
                    candidate_span.text,
                    expected_field=NumericField.CARD_NUMBER,
                )
                if digits is None or len(digits) > 19:
                    continue
                key = (
                    candidate_span.bbox.x,
                    candidate_span.bbox.y,
                    candidate_span.bbox.width,
                    candidate_span.bbox.height,
                    digits,
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(_PanCandidate(digits=digits, span=candidate_span))
        return candidates

    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        context: DetectionContext,
    ) -> list[SensitiveDetection]:
        del frame
        threshold = self.config.thresholds.card_number
        detections = semantic_anchor_detections(
            roles={FieldRole.CARD_NUMBER},
            sensitive_type=SensitiveType.CARD_NUMBER,
            threshold=threshold,
            context=context,
        )
        anchors = context.payment.anchors_for(FieldRole.CARD_NUMBER)
        weights = self.config.weights

        for candidate in self._candidates(ocr, spans=context.ocr_spans):
            length = len(candidate.digits)
            anchor, geometry_score = best_anchor_association(
                candidate.span.bbox, anchors
            )
            valid_luhn = luhn_valid(candidate.digits) if length >= 13 else False

            if anchor is not None and geometry_score >= 0.65 and 1 <= length <= 19:
                structure = 1.0 if length >= 13 else 0.65
                confidence = (
                    weights.semantic_label
                    + weights.payment_context * context.payment.score
                    + weights.numeric_structure * structure
                    + weights.field_geometry * geometry_score
                    + (weights.luhn if valid_luhn else 0.0)
                    + (
                        weights.neighboring_field
                        if has_neighboring_payment_field(
                            context, excluding={FieldRole.CARD_NUMBER}
                        )
                        else 0.0
                    )
                )
                confidence = max(confidence, anchor.confidence * 0.84)
                confidence *= 0.70 + 0.30 * candidate.span.confidence
                if confidence >= threshold:
                    detections.append(
                        make_detection(
                            sensitive_type=SensitiveType.CARD_NUMBER,
                            bbox=candidate.span.bbox,
                            confidence=confidence,
                            source=(
                                "semantic_value_luhn"
                                if valid_luhn
                                else "semantic_value"
                            ),
                            context=context,
                        )
                    )
                continue

            if 13 <= length <= 19 and valid_luhn:
                confidence = (
                    self.config.payment_luhn_confidence
                    if context.payment.score >= self.config.minimum_payment_context
                    else self.config.unlabeled_luhn_confidence
                )
                confidence *= 0.80 + 0.20 * candidate.span.confidence
                if confidence >= threshold:
                    detections.append(
                        make_detection(
                            sensitive_type=SensitiveType.CARD_NUMBER,
                            bbox=candidate.span.bbox,
                            confidence=confidence,
                            source="pattern_luhn",
                            context=context,
                        )
                    )
        return detections
