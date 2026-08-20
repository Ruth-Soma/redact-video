from __future__ import annotations

import pytest

from redact_video.detection.luhn import luhn_valid


@pytest.mark.parametrize(
    "number",
    [
        "4242424242424242",
        "4242 4242 4242 4242",
        "4242-4242-4242-4242",
        "4000000000000002",
    ],
)
def test_luhn_accepts_valid_synthetic_pan(number: str) -> None:
    assert luhn_valid(number) is True


@pytest.mark.parametrize(
    "number",
    [
        "4242424242424241",
        "424242424242",
        "42424242424242424242",
        "4242x424242424242",
        "",
    ],
)
def test_luhn_rejects_invalid_candidates(number: str) -> None:
    assert luhn_valid(number) is False
