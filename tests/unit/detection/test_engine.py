from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from redact_video.config import DetectionConfig
from redact_video.detection import engine as engine_module
from redact_video.detection.base import SensitiveDetector
from redact_video.detection.engine import DetectionEngine
from redact_video.models import OcrToken, SensitiveType


def _types(detections):  # type: ignore[no-untyped-def]
    return {item.type for item in detections}


def test_static_card_form_detects_all_value_types_without_storing_values(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory("4242 4242 4242 4242", 50, 80, 230),
        token_factory("Expiry", 50, 140, 75),
        token_factory("08/29", 50, 170, 70),
        token_factory("CVV", 210, 140, 45),
        token_factory("729", 210, 170, 45),
        token_factory("Name", 50, 230, 60),
        token_factory("John Example", 50, 260, 150),
    ]

    detections = DetectionEngine().detect(
        frame, tokens, frame_index=12, timestamp_ms=400
    )

    assert SensitiveType.CARD_NUMBER in _types(detections)
    assert SensitiveType.CARD_EXPIRY in _types(detections)
    assert SensitiveType.CARD_CVV in _types(detections)
    assert all(
        item.frame_index == 12 and item.timestamp_ms == 400 for item in detections
    )
    serialized = [item.model_dump() for item in detections]
    serialized_text = str(serialized)
    assert "4242" not in serialized_text
    assert "729" not in serialized_text
    assert all("value" not in item for item in serialized)


@pytest.mark.parametrize("partial", ["4", "42", "424", "4242"])
def test_partial_pan_is_protected_after_field_label_is_seen(
    partial: str,
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory(partial, 50, 80, max(15, len(partial) * 13)),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=1, timestamp_ms=33)

    assert any(item.type is SensitiveType.CARD_NUMBER for item in detections)


def test_inline_partial_pan_is_protected_without_blurring_the_label(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [token_factory("Card Number: 4242", 50, 50, 230)]

    detections = DetectionEngine().detect(frame, tokens, frame_index=1, timestamp_ms=33)

    value_detections = [
        item
        for item in detections
        if item.type is SensitiveType.CARD_NUMBER
        and item.source in {"semantic_value", "semantic_value_luhn", "merged"}
    ]
    assert value_detections
    assert any(item.bbox.x > 150 for item in value_detections)


def test_empty_semantic_fields_are_anchored_before_typing(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory("Expiry", 50, 140, 75),
        token_factory("CVV", 210, 140, 45),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert {
        SensitiveType.CARD_NUMBER,
        SensitiveType.CARD_EXPIRY,
        SensitiveType.CARD_CVV,
    }.issubset(_types(detections))
    assert all(item.source in {"semantic_anchor", "merged"} for item in detections)


def test_luhn_failure_remains_sensitive_in_a_semantic_card_field(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory("4242 4242 4242 4241", 50, 80, 230),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.CARD_NUMBER in _types(detections)


def test_unlabeled_pan_requires_luhn_validation(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    valid = [token_factory("4242 4242 4242 4242", 50, 80, 230)]
    invalid = [token_factory("4242 4242 4242 4241", 50, 80, 230)]

    valid_result = DetectionEngine().detect(frame, valid, frame_index=0, timestamp_ms=0)
    invalid_result = DetectionEngine().detect(
        frame, invalid, frame_index=0, timestamp_ms=0
    )

    assert SensitiveType.CARD_NUMBER in _types(valid_result)
    assert SensitiveType.CARD_NUMBER not in _types(invalid_result)


def test_expiry_near_an_unlabeled_luhn_valid_pan_is_sensitive(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("4242 4242 4242 4242", 50, 80, 230),
        token_factory("08/29", 50, 120, 70),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.CARD_NUMBER in _types(detections)
    assert SensitiveType.CARD_EXPIRY in _types(detections)


@pytest.mark.parametrize("expiry", ["08/29", "08 / 29", "08-29", "08 29", "08/2029"])
def test_supported_expiry_formats(
    expiry: str, frame: np.ndarray, token_factory
) -> None:  # type: ignore[no-untyped-def]
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory("Expiry", 50, 140, 75),
        token_factory(expiry, 50, 170, 85),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.CARD_EXPIRY in _types(detections)


def test_split_month_and_year_fields_are_protected(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card Number", 50, 30, 145),
        token_factory("Month", 50, 140, 70),
        token_factory("08", 50, 170, 30),
        token_factory("Year", 180, 140, 55),
        token_factory("2029", 180, 170, 60),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    expiry_detections = [
        item for item in detections if item.type is SensitiveType.CARD_EXPIRY
    ]
    assert len(expiry_detections) >= 2


@pytest.mark.parametrize("label", ["CVV", "CVC", "CSC", "CID", "Security Code"])
@pytest.mark.parametrize("partial", ["7", "72", "729", "7291"])
def test_partial_cvv_is_protected_only_with_security_code_semantics(
    label: str,
    partial: str,
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory(label, 210, 140, 140),
        token_factory(partial, 210, 170, 45),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.CARD_CVV in _types(detections)


def test_inline_card_security_code_is_detected(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [token_factory("Card Security Code: 729", 50, 50, 270)]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.CARD_CVV in _types(detections)


def test_saved_card_masks_and_expiry_are_detected_without_blurring_brand(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Visa", 30, 70, 45),
        token_factory("•••• •••• •••• 4242", 100, 70, 220),
        token_factory("Expires", 100, 110, 75),
        token_factory("08/29", 100, 140, 70),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    numbers = [
        item for item in detections if item.type is SensitiveType.SAVED_CARD_NUMBER
    ]
    assert numbers
    assert all(item.bbox.x >= 75 for item in numbers)
    assert SensitiveType.SAVED_CARD_EXPIRY in _types(detections)


@pytest.mark.parametrize(
    "presentation",
    [
        "**** 4242",
        "•••• 4242",
        "xxxx 4242",
        "ending in 4242",
        "Visa •••• 4242",
        "Mastercard **** 4242",
    ],
)
def test_saved_card_presentations_are_supported(
    presentation: str,
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [token_factory(presentation, 100, 70, 240)]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.SAVED_CARD_NUMBER in _types(detections)


def test_saved_card_policy_can_be_disabled(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [token_factory("ending in 4242", 100, 70, 180)]

    detections = DetectionEngine(DetectionConfig(blur_saved_card=False)).detect(
        frame, tokens, frame_index=0, timestamp_ms=0
    )

    assert SensitiveType.SAVED_CARD_NUMBER not in _types(detections)
    assert SensitiveType.SAVED_CARD_EXPIRY not in _types(detections)


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("OTP", "729"),
        ("Order", "424242"),
        ("Price", "£729.00"),
        ("Room", "729"),
        ("Date", "08/29"),
        ("Invoice ID", "2029"),
        ("Postal code", "729"),
    ],
)
def test_false_positive_pairs_remain_unredacted(
    label: str,
    value: str,
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory(label, 50, 50, 100),
        token_factory(value, 180, 50, 120),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert detections == []


def test_arbitrary_three_digits_are_not_cvv_even_on_card_screen(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory("OTP", 500, 400, 50),
        token_factory("729", 570, 400, 45),
    ]

    detections = DetectionEngine().detect(frame, tokens, frame_index=0, timestamp_ms=0)

    assert SensitiveType.CARD_CVV not in _types(detections)


def test_detector_protocol_is_structural_and_empty_plugin_list_is_respected(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
) -> None:
    assert isinstance(DetectionEngine().detectors[0], SensitiveDetector)
    tokens = [token_factory("Card Number", 50, 50, 145)]

    detections = DetectionEngine(detectors=()).detect(
        frame,
        tokens,
        frame_index=0,
        timestamp_ms=0,
    )

    assert detections == []


def test_engine_builds_shared_ocr_spans_once(
    frame: np.ndarray,
    token_factory: Callable[..., OcrToken],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = [
        token_factory("Card Number", 50, 50, 145),
        token_factory("4242 4242 4242 4242", 50, 80, 230),
        token_factory("Expiry", 50, 140, 75),
        token_factory("08/29", 50, 170, 70),
    ]
    original = engine_module.build_ocr_spans
    calls = 0

    def counted_spans(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, "build_ocr_spans", counted_spans)

    detections = DetectionEngine().detect(
        frame,
        tokens,
        frame_index=0,
        timestamp_ms=0,
    )

    assert detections
    assert calls == 1
