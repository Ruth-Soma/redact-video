from __future__ import annotations

from collections.abc import Callable

from redact_video.detection.payment_context import (
    FieldRole,
    PaymentContextDetector,
)
from redact_video.models import OcrToken


def test_context_fuzzy_matches_split_ocr_labels_and_builds_anchors(
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card", 40, 40, 55),
        token_factory("Nurnber", 102, 40, 90),
        token_factory("Expiry", 40, 130, 75),
        token_factory("CVV", 210, 130, 45),
    ]

    result = PaymentContextDetector().analyze(tokens, frame_width=800, frame_height=600)

    roles = {anchor.role for anchor in result.anchors}
    assert result.score >= 0.80
    assert FieldRole.CARD_NUMBER in roles
    assert FieldRole.EXPIRY in roles
    assert FieldRole.CVV in roles
    assert all(anchor.value_bbox.y != anchor.label_bbox.y for anchor in result.anchors)
    assert all(anchor.value_bbox.right <= 800 for anchor in result.anchors)


def test_month_and_year_become_anchors_only_in_payment_context(
    token_factory: Callable[..., OcrToken],
) -> None:
    detector = PaymentContextDetector()
    ordinary = [
        token_factory("Month", 40, 40),
        token_factory("Year", 180, 40),
    ]
    payment = [token_factory("Card Number", 40, 10), *ordinary]

    ordinary_result = detector.analyze(ordinary, frame_width=800, frame_height=600)
    payment_result = detector.analyze(payment, frame_width=800, frame_height=600)

    assert ordinary_result.anchors == ()
    assert FieldRole.EXPIRY_MONTH in {item.role for item in payment_result.anchors}
    assert FieldRole.EXPIRY_YEAR in {item.role for item in payment_result.anchors}


def test_unrelated_screen_has_no_payment_context(
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Order history", 30, 30),
        token_factory("Room 729", 30, 70),
        token_factory("Delivery date", 30, 110),
    ]

    result = PaymentContextDetector().analyze(tokens, frame_width=800, frame_height=600)

    assert result.score == 0
    assert result.anchors == ()


def test_inferred_value_height_is_stable_for_tight_native_ocr_boxes(
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [token_factory("Card Number", 50, 100, 180, 20)]

    result = PaymentContextDetector().analyze(
        tokens,
        frame_width=1260,
        frame_height=2736,
    )

    anchor = next(item for item in result.anchors if item.role is FieldRole.CARD_NUMBER)
    assert anchor.value_bbox.height == 96
