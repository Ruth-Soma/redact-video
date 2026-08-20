"""Opt-in smoke test for the real local PaddleOCR runtime."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from redact_video.config import OcrConfig
from redact_video.detection import DetectionEngine
from redact_video.models import SensitiveType
from redact_video.ocr.paddle import PaddleOcrEngine
from redact_video.redaction.renderer import BlurRenderer
from redact_video.redaction.timeline import MaskTimelineBuilder
from redact_video.tracking import RegionTracker
from redact_video.verification import SafetyVerifier

pytestmark = pytest.mark.ocr_runtime


def test_real_paddle_runtime_scans_an_in_memory_frame() -> None:
    if os.getenv("RUN_OCR_RUNTIME_TEST") != "1":
        pytest.skip("Set RUN_OCR_RUNTIME_TEST=1 to load local PaddleOCR models")

    pytest.importorskip("paddle")
    pytest.importorskip("paddleocr")
    frame = np.full((240, 720, 3), 255, dtype=np.uint8)
    cv2.putText(
        frame,
        "Card Number",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    ocr_engine = PaddleOcrEngine(OcrConfig(enable_mkldnn=False, cpu_threads=2))
    tokens = ocr_engine.scan(frame)
    detections = DetectionEngine().detect(
        frame,
        tokens,
        frame_index=0,
        timestamp_ms=0,
    )
    tracking = RegionTracker().update(
        frame,
        frame_index=0,
        timestamp_ms=0,
        detections=detections,
    )
    timeline_builder = MaskTimelineBuilder()
    timeline_builder.add(tracking)
    timeline = timeline_builder.build()
    rendered = BlurRenderer().apply_timeline(frame, timeline, frame_index=0)
    remaining_leaks = SafetyVerifier(ocr_engine).inspect_frame(
        rendered,
        frame_index=0,
        timestamp_ms=0,
    )

    assert tokens
    assert all(token.bbox.width > 0 and token.bbox.height > 0 for token in tokens)
    assert any(item.type is SensitiveType.CARD_NUMBER for item in detections)
    assert any(item.type is SensitiveType.CARD_NUMBER for item in tracking.regions)
    assert timeline.boxes_for(0)
    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered, frame)
    assert not remaining_leaks
