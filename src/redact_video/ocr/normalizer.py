"""Constrained OCR normalization helpers.

Numeric correction is deliberately gated by an explicit expected field. These
functions must only be used after detector context has identified a numeric card
field candidate; they do not mutate or replace the original OCR token.
"""

from __future__ import annotations

import unicodedata
from enum import Enum


class NumericField(str, Enum):
    CARD_NUMBER = "card_number"
    EXPIRY = "expiry"
    SECURITY_CODE = "security_code"
    SAVED_CARD = "saved_card"


_NUMERIC_CONFUSIONS = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "s": "5",
}
_SAVED_CARD_MASKS = frozenset({"*", "•", "●", "·", "x", "X"})


def normalize_context_text(text: str) -> str:
    """Normalize label/context text without applying numeric substitutions."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    semantic = "".join(char if char.isalnum() else " " for char in normalized)
    return " ".join(semantic.split())


def normalize_numeric_candidate(
    text: str,
    *,
    expected_field: NumericField,
) -> str | None:
    """Return a temporary canonical candidate or ``None`` when structurally invalid.

    Requiring ``expected_field`` prevents the OCR confusion map from being used
    as a global text transform. The returned value may itself be sensitive and
    must be discarded immediately after detector validation.
    """

    candidate = unicodedata.normalize("NFKC", text).strip()
    if not candidate:
        return None

    output: list[str] = []
    for char in candidate:
        if "0" <= char <= "9":
            output.append(char)
            continue
        if char in _NUMERIC_CONFUSIONS:
            output.append(_NUMERIC_CONFUSIONS[char])
            continue
        if char.isspace():
            continue
        if expected_field is NumericField.CARD_NUMBER and char == "-":
            continue
        if expected_field is NumericField.EXPIRY and char in {"/", "-"}:
            if output and output[-1] != "/":
                output.append("/")
            continue
        if expected_field is NumericField.SAVED_CARD:
            if char == "-":
                continue
            if char in _SAVED_CARD_MASKS:
                output.append("*")
                continue
        return None

    if expected_field is NumericField.EXPIRY:
        while output and output[-1] == "/":
            output.pop()
    canonical = "".join(output)
    if not canonical or not any(char.isdigit() for char in canonical):
        return None
    return canonical
