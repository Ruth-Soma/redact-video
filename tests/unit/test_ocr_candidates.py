from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.ocr_candidates import (
    BenchmarkFrame,
    CandidateSpec,
    CandidateStatus,
    _even_frame_indices,
    run_ocr_candidate_benchmark,
)
from redact_video.models import BoundingBox, OcrToken


class _StaticOcr:
    def __init__(self, tokens: list[OcrToken]) -> None:
        self.tokens = tokens

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        return self.tokens


def _private_tokens() -> list[OcrToken]:
    return [
        OcrToken(
            text="Payment details",
            confidence=0.99,
            bbox=BoundingBox(x=20, y=20, width=180, height=30),
        ),
        OcrToken(
            text="Card Number",
            confidence=0.99,
            bbox=BoundingBox(x=20, y=100, width=150, height=30),
        ),
        OcrToken(
            text="4242 4242 4242 4242",
            confidence=0.99,
            bbox=BoundingBox(x=220, y=100, width=260, height=30),
        ),
        OcrToken(
            text="Expiry",
            confidence=0.99,
            bbox=BoundingBox(x=20, y=160, width=100, height=30),
        ),
        OcrToken(
            text="12/29",
            confidence=0.99,
            bbox=BoundingBox(x=220, y=160, width=80, height=30),
        ),
        OcrToken(
            text="CVV",
            confidence=0.99,
            bbox=BoundingBox(x=20, y=220, width=80, height=30),
        ),
        OcrToken(
            text="123",
            confidence=0.99,
            bbox=BoundingBox(x=220, y=220, width=60, height=30),
        ),
    ]


def test_candidate_benchmark_persists_only_value_free_evidence(tmp_path: Path) -> None:
    frames = tuple(
        BenchmarkFrame(
            frame_index=index,
            timestamp_ms=index * 100,
            image=np.zeros((400, 800, 3), dtype=np.uint8),
        )
        for index in range(2)
    )
    report_path = tmp_path / "private-benchmark.json"

    report = run_ocr_candidate_benchmark(
        frames,
        (
            CandidateSpec("paddle_v5", lambda: _StaticOcr(_private_tokens())),
            CandidateSpec("equivalent", lambda: _StaticOcr(_private_tokens())),
            CandidateSpec(
                "unavailable",
                lambda: (_ for _ in ()).throw(
                    RuntimeError("model failed near 4242 4242 4242 4242")
                ),
            ),
        ),
        report_path=report_path,
        projected_scan_count=76,
    )

    equivalent = next(item for item in report.candidates if item.name == "equivalent")
    unavailable = next(item for item in report.candidates if item.name == "unavailable")
    assert report.has_sensitive_coverage is True
    assert equivalent.status is CandidateStatus.COMPLETED
    assert equivalent.equivalence is not None
    assert equivalent.equivalence.preserved_all_findings is True
    assert equivalent.eligible is True
    assert equivalent.projected_scan_seconds is not None
    assert unavailable.status is CandidateStatus.UNAVAILABLE

    serialized = json.dumps(json.loads(report_path.read_text(encoding="utf-8")))
    assert "4242" not in serialized
    assert "12/29" not in serialized
    assert "Payment details" not in serialized
    assert "model failed" not in serialized
    assert "input_path" not in serialized


def test_candidate_without_sensitive_reference_frames_is_not_eligible(
    tmp_path: Path,
) -> None:
    frame = BenchmarkFrame(
        frame_index=0,
        timestamp_ms=0,
        image=np.zeros((100, 200, 3), dtype=np.uint8),
    )

    report = run_ocr_candidate_benchmark(
        (frame,),
        (
            CandidateSpec("paddle_v5", lambda: _StaticOcr([])),
            CandidateSpec("empty", lambda: _StaticOcr([])),
        ),
        report_path=tmp_path / "empty.json",
    )

    assert report.has_sensitive_coverage is False
    assert report.recommended_candidate is None
    assert all(candidate.eligible is False for candidate in report.candidates)


def test_even_frame_selection_is_bounded_and_deterministic() -> None:
    assert _even_frame_indices(100, 3) == (25, 50, 74)
    assert _even_frame_indices(3, 7) == (0, 1, 2)
