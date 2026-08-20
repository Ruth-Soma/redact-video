from __future__ import annotations

import numpy as np

from redact_video.models import BoundingBox, OcrToken
from redact_video.ocr.base import OcrEngine


class MockOcrEngine:
    def __init__(self, tokens: list[OcrToken]) -> None:
        self.tokens = tokens

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        assert frame.dtype == np.uint8
        return self.tokens


def test_protocol_accepts_a_mock_without_paddle_dependency() -> None:
    token = OcrToken(
        text="Card Number",
        confidence=0.9,
        bbox=BoundingBox(x=10, y=20, width=100, height=20),
    )
    engine = MockOcrEngine([token])

    assert isinstance(engine, OcrEngine)
    assert engine.scan(np.zeros((40, 160, 3), dtype=np.uint8)) == [token]


def test_ocr_token_hides_text_from_repr() -> None:
    token = OcrToken(
        text="4242 4242 4242 4242",
        confidence=0.99,
        bbox=BoundingBox(x=10, y=20, width=100, height=20),
    )

    assert "4242" not in repr(token)
    assert "confidence=0.99" in repr(token)
    assert "text" not in token.model_dump()
    assert "4242" not in token.model_dump_json()
