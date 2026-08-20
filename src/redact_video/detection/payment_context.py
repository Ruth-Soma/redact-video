"""Payment-screen context, fuzzy label matching, and semantic field anchors."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

from redact_video.config import DetectionConfig
from redact_video.detection.geometry import (
    OcrSpan,
    build_ocr_spans,
    overlap_fraction,
    text_subspan_box,
)
from redact_video.models import BoundingBox, OcrToken, SensitiveType
from redact_video.ocr.normalizer import normalize_context_text


class FieldRole(str, Enum):
    CARD_NUMBER = "card_number"
    EXPIRY = "expiry"
    EXPIRY_MONTH = "expiry_month"
    EXPIRY_YEAR = "expiry_year"
    CVV = "cvv"


@dataclass(frozen=True, slots=True)
class FieldAnchor:
    role: FieldRole
    sensitive_type: SensitiveType
    label_bbox: BoundingBox
    value_bbox: BoundingBox
    confidence: float


@dataclass(frozen=True, slots=True)
class PaymentContextResult:
    score: float
    anchors: tuple[FieldAnchor, ...]
    categories: frozenset[str]

    def anchors_for(self, role: FieldRole) -> tuple[FieldAnchor, ...]:
        return tuple(anchor for anchor in self.anchors if anchor.role is role)


@dataclass(frozen=True, slots=True)
class _LabelSpec:
    phrase: str
    category: str
    role: FieldRole | None = None


@dataclass(frozen=True, slots=True)
class _LabelMatch:
    spec: _LabelSpec
    score: float
    bbox: BoundingBox


_DIRECT_LABELS = (
    _LabelSpec("credit card number", "card_number", FieldRole.CARD_NUMBER),
    _LabelSpec("debit card number", "card_number", FieldRole.CARD_NUMBER),
    _LabelSpec("card number", "card_number", FieldRole.CARD_NUMBER),
    _LabelSpec("card security code", "cvv", FieldRole.CVV),
    _LabelSpec("security code", "cvv", FieldRole.CVV),
    _LabelSpec("expiration date", "expiry", FieldRole.EXPIRY),
    _LabelSpec("expiry date", "expiry", FieldRole.EXPIRY),
    _LabelSpec("expiration", "expiry", FieldRole.EXPIRY),
    _LabelSpec("expires", "expiry", FieldRole.EXPIRY),
    _LabelSpec("expiry", "expiry", FieldRole.EXPIRY),
    _LabelSpec("cvv", "cvv", FieldRole.CVV),
    _LabelSpec("cvc", "cvv", FieldRole.CVV),
    _LabelSpec("csc", "cvv", FieldRole.CVV),
    _LabelSpec("cid", "cvv", FieldRole.CVV),
)

_GENERIC_LABELS = (
    _LabelSpec("payment details", "payment"),
    _LabelSpec("payment method", "payment"),
    _LabelSpec("credit card", "payment"),
    _LabelSpec("debit card", "payment"),
    _LabelSpec("name on card", "payment"),
    _LabelSpec("save card", "payment"),
    _LabelSpec("add card", "payment"),
    _LabelSpec("billing", "billing"),
    _LabelSpec("card", "payment"),
)

_SPLIT_EXPIRY_LABELS = (
    _LabelSpec("month", "expiry", FieldRole.EXPIRY_MONTH),
    _LabelSpec("mm", "expiry", FieldRole.EXPIRY_MONTH),
    _LabelSpec("year", "expiry", FieldRole.EXPIRY_YEAR),
    _LabelSpec("yyyy", "expiry", FieldRole.EXPIRY_YEAR),
    _LabelSpec("yy", "expiry", FieldRole.EXPIRY_YEAR),
)

_MINIMUM_VALUE_HEIGHT_RATIO = 0.035


def _phrase_match(
    text: str, phrase: str, fuzzy_threshold: float
) -> tuple[float, tuple[int, int] | None]:
    if not text:
        return 0.0, None
    exact = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text)
    if exact:
        return 1.0, exact.span()
    if len(phrase) <= 4:
        return 0.0, None
    length_ratio = len(text) / len(phrase)
    if not 0.55 <= length_ratio <= 1.65:
        return 0.0, None
    matcher = SequenceMatcher(None, text, phrase)
    if (
        matcher.real_quick_ratio() < fuzzy_threshold
        or matcher.quick_ratio() < fuzzy_threshold
    ):
        return 0.0, None
    score = matcher.ratio()
    return (score, None) if score >= fuzzy_threshold else (0.0, None)


def _match_span(
    span: OcrSpan,
    specs: tuple[_LabelSpec, ...],
    fuzzy_threshold: float,
) -> list[_LabelMatch]:
    normalized = normalize_context_text(span.text)
    matches: list[_LabelMatch] = []
    for spec in specs:
        score, character_span = _phrase_match(normalized, spec.phrase, fuzzy_threshold)
        if score <= 0:
            continue
        bbox = (
            text_subspan_box(span, *character_span)
            if character_span is not None
            else span.bbox
        )
        matches.append(_LabelMatch(spec=spec, score=score, bbox=bbox))
    return matches


def _infer_value_box(
    role: FieldRole,
    label: BoundingBox,
    *,
    frame_width: int,
    frame_height: int,
) -> BoundingBox:
    width_ratio = {
        FieldRole.CARD_NUMBER: 0.36,
        FieldRole.EXPIRY: 0.16,
        FieldRole.EXPIRY_MONTH: 0.10,
        FieldRole.EXPIRY_YEAR: 0.12,
        FieldRole.CVV: 0.12,
    }[role]
    label_multiplier = {
        FieldRole.CARD_NUMBER: 2.8,
        FieldRole.EXPIRY: 1.8,
        FieldRole.EXPIRY_MONTH: 1.5,
        FieldRole.EXPIRY_YEAR: 1.5,
        FieldRole.CVV: 1.7,
    }[role]
    width = round(max(label.width * label_multiplier, frame_width * width_ratio))
    width = max(1, min(width, frame_width - min(label.x, frame_width - 1)))
    # Keep the inferred field footprint stable across OCR backends. Native OCR
    # commonly returns tighter label boxes than Paddle for the same screen text;
    # tying this entirely to label height can then reject a value that v5 masks.
    height = round(max(label.height * 1.55, frame_height * _MINIMUM_VALUE_HEIGHT_RATIO))
    height = max(1, min(height, frame_height))
    gap = max(2, round(label.height * 0.20))
    y = label.bottom + gap
    if y + height > frame_height:
        y = max(0, label.y - gap - height)
    x = min(label.x, frame_width - width)
    return BoundingBox(x=x, y=y, width=width, height=height)


def _deduplicate_matches(matches: list[_LabelMatch]) -> list[_LabelMatch]:
    ordered = sorted(matches, key=lambda item: item.score, reverse=True)
    selected: list[_LabelMatch] = []
    for match in ordered:
        if any(
            existing.spec.role is match.spec.role
            and overlap_fraction(existing.bbox, match.bbox) >= 0.55
            for existing in selected
        ):
            continue
        selected.append(match)
    return selected


class PaymentContextDetector:
    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

    def analyze(
        self,
        ocr: list[OcrToken],
        *,
        frame_width: int,
        frame_height: int,
        spans: Sequence[OcrSpan] | None = None,
    ) -> PaymentContextResult:
        maximum_span_tokens = min(4, self.config.max_ocr_span_tokens)
        active_spans = (
            build_ocr_spans(ocr, max_tokens=maximum_span_tokens)
            if spans is None
            else tuple(
                span for span in spans if span.token_count <= maximum_span_tokens
            )
        )
        direct_matches: list[_LabelMatch] = []
        generic_matches: list[_LabelMatch] = []
        for span in active_spans:
            direct_matches.extend(
                _match_span(span, _DIRECT_LABELS, self.config.fuzzy_label_threshold)
            )
            generic_matches.extend(
                _match_span(span, _GENERIC_LABELS, self.config.fuzzy_label_threshold)
            )
        direct_matches = _deduplicate_matches(direct_matches)
        generic_matches = _deduplicate_matches(generic_matches)

        categories = {
            match.spec.category for match in [*direct_matches, *generic_matches]
        }
        best_direct = max((match.score for match in direct_matches), default=0.0)
        best_generic = max((match.score for match in generic_matches), default=0.0)
        score = max(best_generic * 0.55, best_direct * 0.78)
        if len(categories) >= 2:
            score += 0.10
        if len(categories) >= 3:
            score += 0.07
        score = min(1.0, score)

        if score >= self.config.minimum_payment_context:
            split_matches: list[_LabelMatch] = []
            for span in active_spans:
                split_matches.extend(
                    _match_span(
                        span,
                        _SPLIT_EXPIRY_LABELS,
                        self.config.fuzzy_label_threshold,
                    )
                )
            direct_matches.extend(_deduplicate_matches(split_matches))
            if split_matches:
                categories.add("expiry")

        anchors: list[FieldAnchor] = []
        for match in _deduplicate_matches(direct_matches):
            role = match.spec.role
            if role is None:
                continue
            sensitive_type = {
                FieldRole.CARD_NUMBER: SensitiveType.CARD_NUMBER,
                FieldRole.EXPIRY: SensitiveType.CARD_EXPIRY,
                FieldRole.EXPIRY_MONTH: SensitiveType.CARD_EXPIRY,
                FieldRole.EXPIRY_YEAR: SensitiveType.CARD_EXPIRY,
                FieldRole.CVV: SensitiveType.CARD_CVV,
            }[role]
            confidence = min(
                0.99,
                self.config.semantic_anchor_base_confidence
                + 0.18 * match.score
                + 0.05 * score,
            )
            anchors.append(
                FieldAnchor(
                    role=role,
                    sensitive_type=sensitive_type,
                    label_bbox=match.bbox,
                    value_bbox=_infer_value_box(
                        role,
                        match.bbox,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    ),
                    confidence=confidence,
                )
            )
        return PaymentContextResult(
            score=score,
            anchors=tuple(anchors),
            categories=frozenset(categories),
        )
