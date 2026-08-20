from __future__ import annotations

import pytest

from redact_video.ocr.normalizer import (
    NumericField,
    normalize_context_text,
    normalize_numeric_candidate,
)


def test_context_normalization_does_not_apply_numeric_confusions() -> None:
    assert normalize_context_text("  Card—Nurnber:  ") == "card nurnber"
    assert normalize_context_text("Security   Code") == "security code"
    assert normalize_context_text("SAVE_CARD") == "save card"


@pytest.mark.parametrize(
    ("text", "field", "expected"),
    [
        ("4242 4242 4O42 4242", NumericField.CARD_NUMBER, "4242424240424242"),
        ("4242-4242-I242-4242", NumericField.CARD_NUMBER, "4242424212424242"),
        ("08 / 2S", NumericField.EXPIRY, "08/25"),
        ("08-2029", NumericField.EXPIRY, "08/2029"),
        ("7l9", NumericField.SECURITY_CODE, "719"),
        ("•••• •••• •••• 4242", NumericField.SAVED_CARD, "************4242"),
        ("xxxx-4O42", NumericField.SAVED_CARD, "****4042"),
        ("４２４２", NumericField.CARD_NUMBER, "4242"),
    ],
)
def test_numeric_normalization_is_field_constrained(
    text: str, field: NumericField, expected: str
) -> None:
    assert normalize_numeric_candidate(text, expected_field=field) == expected


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ("Order 4242", NumericField.CARD_NUMBER),
        ("CVV 729", NumericField.SECURITY_CODE),
        ("08.29", NumericField.EXPIRY),
        ("•••• 4242", NumericField.CARD_NUMBER),
        ("xxxx", NumericField.SAVED_CARD),
        ("", NumericField.CARD_NUMBER),
    ],
)
def test_numeric_normalization_rejects_non_candidate_text(
    text: str, field: NumericField
) -> None:
    assert normalize_numeric_candidate(text, expected_field=field) is None
