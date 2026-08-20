"""Deterministic OCR oracle driven by decoded synthetic frame markers."""

from __future__ import annotations

import numpy as np

from redact_video.models import OcrToken, SensitiveType
from tests.synthetic.generator import SyntheticScenario, decode_frame_marker


class ScenarioOcrEngine:
    """Return scenario tokens for the actual decoded frame presented to OCR."""

    def __init__(
        self,
        scenario: SyntheticScenario,
        *,
        include_sensitive: bool = True,
        excluded_types: frozenset[SensitiveType] = frozenset(),
        excluded_texts: frozenset[str] = frozenset(),
        sensitive_passes: int | None = None,
    ) -> None:
        self.scenario = scenario
        self.include_sensitive = include_sensitive
        self.excluded_types = excluded_types
        self.excluded_texts = excluded_texts
        self.sensitive_passes = sensitive_passes
        self.calls = 0
        self._scan_pass = 1
        self._last_frame_index: int | None = None

    @property
    def scan_pass(self) -> int:
        return self._scan_pass

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        frame_index = decode_frame_marker(frame)
        if frame_index >= len(self.scenario.frames):
            raise ValueError("Decoded synthetic frame index is outside the scenario")
        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            self._scan_pass += 1
        self._last_frame_index = frame_index
        self.calls += 1

        allow_sensitive = self.include_sensitive and (
            self.sensitive_passes is None or self._scan_pass <= self.sensitive_passes
        )
        tokens: list[OcrToken] = []
        for token in self.scenario.frames[frame_index].tokens:
            if token.text in self.excluded_texts:
                continue
            if token.sensitive_type is not None and (
                not allow_sensitive or token.sensitive_type in self.excluded_types
            ):
                continue
            tokens.append(
                OcrToken(
                    text=token.text,
                    confidence=token.confidence,
                    bbox=token.bbox,
                )
            )
        return tokens
