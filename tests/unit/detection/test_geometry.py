from __future__ import annotations

from collections.abc import Callable

import pytest

from redact_video.detection.geometry import build_ocr_spans
from redact_video.models import OcrToken


@pytest.mark.parametrize("max_tokens", [1, 2, 3, 4, 5, 6])
def test_wide_span_build_exactly_supplies_narrower_requests(
    token_factory: Callable[..., OcrToken],
    max_tokens: int,
) -> None:
    tokens = [
        token_factory("Card", 20, 20, 50, confidence=0.97),
        token_factory("Number", 76, 20, 75, confidence=0.91),
        token_factory("4242", 160, 20, 55, confidence=0.88),
        token_factory("Expiry", 20, 80, 65, confidence=0.95),
        token_factory("08/29", 94, 80, 60, confidence=0.90),
        token_factory("Separated", 400, 80, 100, confidence=0.93),
    ]

    wide = build_ocr_spans(tokens, max_tokens=6)
    filtered = [span for span in wide if span.token_count <= max_tokens]

    assert filtered == build_ocr_spans(tokens, max_tokens=max_tokens)
