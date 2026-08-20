"""Shared detector scoring and sanitized detection construction."""

from __future__ import annotations

from collections.abc import Iterable

from redact_video.detection.base import DetectionContext
from redact_video.detection.geometry import field_association_score
from redact_video.detection.payment_context import FieldAnchor, FieldRole
from redact_video.models import BoundingBox, SensitiveDetection, SensitiveType


def make_detection(
    *,
    sensitive_type: SensitiveType,
    bbox: BoundingBox,
    confidence: float,
    source: str,
    context: DetectionContext,
) -> SensitiveDetection:
    return SensitiveDetection(
        type=sensitive_type,
        frame_index=context.frame_index,
        timestamp_ms=context.timestamp_ms,
        bbox=bbox,
        confidence=max(0.0, min(1.0, confidence)),
        source=source,
    )


def semantic_anchor_detections(
    *,
    roles: Iterable[FieldRole],
    sensitive_type: SensitiveType,
    threshold: float,
    context: DetectionContext,
) -> list[SensitiveDetection]:
    accepted_roles = set(roles)
    return [
        make_detection(
            sensitive_type=sensitive_type,
            bbox=anchor.value_bbox,
            confidence=anchor.confidence,
            source="semantic_anchor",
            context=context,
        )
        for anchor in context.payment.anchors
        if anchor.role in accepted_roles and anchor.confidence >= threshold
    ]


def best_anchor_association(
    candidate: BoundingBox,
    anchors: Iterable[FieldAnchor],
) -> tuple[FieldAnchor | None, float]:
    best_anchor: FieldAnchor | None = None
    best_score = 0.0
    for anchor in anchors:
        score = field_association_score(
            anchor.label_bbox,
            anchor.value_bbox,
            candidate,
        )
        if score > best_score:
            best_anchor = anchor
            best_score = score
    return best_anchor, best_score


def has_neighboring_payment_field(
    context: DetectionContext,
    *,
    excluding: set[FieldRole],
) -> bool:
    return any(anchor.role not in excluding for anchor in context.payment.anchors)
