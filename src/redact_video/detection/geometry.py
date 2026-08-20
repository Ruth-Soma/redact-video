"""Transient OCR span and field-association geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from redact_video.models import BoundingBox, OcrToken


@dataclass(frozen=True, slots=True)
class OcrSpan:
    text: str = field(repr=False)
    confidence: float
    bbox: BoundingBox
    token_count: int


def union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def overlap_fraction(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.right, second.right)
    bottom = min(first.bottom, second.bottom)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min(first.width * first.height, second.width * second.height)
    return intersection / smaller if smaller else 0.0


def box_distance(first: BoundingBox, second: BoundingBox) -> float:
    dx = max(first.x - second.right, second.x - first.right, 0)
    dy = max(first.y - second.bottom, second.y - first.bottom, 0)
    return math.hypot(dx, dy)


def _same_line(first: BoundingBox, second: BoundingBox) -> bool:
    overlap = max(0, min(first.bottom, second.bottom) - max(first.y, second.y))
    minimum_height = min(first.height, second.height)
    center_delta = abs((first.y + first.height / 2) - (second.y + second.height / 2))
    return overlap >= minimum_height * 0.35 or center_delta <= minimum_height * 0.55


def build_ocr_spans(tokens: list[OcrToken], *, max_tokens: int = 6) -> list[OcrSpan]:
    """Build individual and adjacent same-line spans without persisting text."""

    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda token: (token.bbox.y, token.bbox.x))
    lines: list[list[OcrToken]] = []
    for token in ordered:
        matching_line = next(
            (
                line
                for line in lines
                if any(_same_line(existing.bbox, token.bbox) for existing in line)
            ),
            None,
        )
        if matching_line is None:
            lines.append([token])
        else:
            matching_line.append(token)

    spans: list[OcrSpan] = []
    for line in lines:
        line.sort(key=lambda token: token.bbox.x)
        for start in range(len(line)):
            previous: OcrToken | None = None
            text = ""
            confidence = 1.0
            left = top = right = bottom = 0
            for token_count, token in enumerate(
                line[start : start + max_tokens],
                start=1,
            ):
                if previous is not None:
                    gap = token.bbox.x - previous.bbox.right
                    allowed_gap = max(
                        24, round(max(previous.bbox.height, token.bbox.height) * 2.5)
                    )
                    if gap > allowed_gap:
                        break
                    text = f"{text} {token.text}"
                    left = min(left, token.bbox.x)
                    top = min(top, token.bbox.y)
                    right = max(right, token.bbox.right)
                    bottom = max(bottom, token.bbox.bottom)
                else:
                    text = token.text
                    left = token.bbox.x
                    top = token.bbox.y
                    right = token.bbox.right
                    bottom = token.bbox.bottom
                confidence = min(confidence, token.confidence)
                spans.append(
                    OcrSpan(
                        text=text,
                        confidence=confidence,
                        bbox=BoundingBox(
                            x=left,
                            y=top,
                            width=right - left,
                            height=bottom - top,
                        ),
                        token_count=token_count,
                    )
                )
                previous = token
    return spans


def text_subspan_box(span: OcrSpan, start: int, end: int) -> BoundingBox:
    """Approximate a horizontal substring rectangle inside an OCR span."""

    text_length = max(1, len(span.text))
    safe_start = max(0, min(start, text_length))
    safe_end = max(safe_start + 1, min(end, text_length))
    left = span.bbox.x + math.floor(span.bbox.width * safe_start / text_length)
    right = span.bbox.x + math.ceil(span.bbox.width * safe_end / text_length)
    right = max(left + 1, min(right, span.bbox.right))
    return BoundingBox(
        x=left,
        y=span.bbox.y,
        width=right - left,
        height=span.bbox.height,
    )


def field_association_score(
    label_box: BoundingBox,
    inferred_value_box: BoundingBox,
    candidate_box: BoundingBox,
) -> float:
    """Score whether candidate geometry belongs to a semantic field label."""

    overlap = overlap_fraction(inferred_value_box, candidate_box)
    if overlap > 0:
        return min(1.0, 0.70 + 0.30 * overlap)

    vertical_overlap = max(
        0,
        min(label_box.bottom, candidate_box.bottom) - max(label_box.y, candidate_box.y),
    )
    if vertical_overlap >= min(label_box.height, candidate_box.height) * 0.35:
        horizontal_gap = candidate_box.x - label_box.right
        if (
            0
            <= horizontal_gap
            <= max(label_box.height * 5, inferred_value_box.width * 0.4)
        ):
            return 0.78

    vertical_gap = candidate_box.y - label_box.bottom
    center_delta = abs(
        (candidate_box.x + candidate_box.width / 2)
        - (inferred_value_box.x + inferred_value_box.width / 2)
    )
    if (
        0 <= vertical_gap <= max(label_box.height * 5, inferred_value_box.height * 3)
        and center_delta <= inferred_value_box.width * 0.65
    ):
        return 0.72
    return 0.0
