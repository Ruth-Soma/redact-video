from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from redact_video.config import OcrConfig
from redact_video.exceptions import OcrProcessingError, OcrUnavailableError
from redact_video.models import BoundingBox
from redact_video.ocr.paddle import PaddleOcrEngine


class FakeResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.json = {"res": payload}


class FakePipeline:
    def __init__(self, results: Iterable[object]) -> None:
        self.results = results
        self.predict_calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> Iterable[object]:
        self.predict_calls.append(kwargs)
        return self.results


def _frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("available_cores", "expected_threads"),
    [
        (None, 4),
        (1, 1),
        (4, 4),
        (10, 8),
        (96, 8),
    ],
)
def test_default_cpu_threads_follow_available_cores_with_a_bounded_cap(
    monkeypatch: pytest.MonkeyPatch,
    available_cores: int | None,
    expected_threads: int,
) -> None:
    monkeypatch.setattr(
        "redact_video.config.os.cpu_count",
        lambda: available_cores,
    )

    assert OcrConfig().cpu_threads == expected_threads


def test_adapter_lazily_initializes_and_parses_current_v3_result_shape() -> None:
    result = FakeResult(
        {
            "rec_texts": [" Card Number ", "729", ""],
            "rec_scores": np.array([0.98, 0.10, 0.99]),
            "rec_boxes": np.array(
                [
                    [-5, 10, 105, 31],
                    [120, 10, 150, 30],
                    [0, 0, 1, 1],
                ]
            ),
        }
    )
    pipeline = FakePipeline([result])
    factory_calls: list[dict[str, Any]] = []

    def factory(**options: Any) -> FakePipeline:
        factory_calls.append(options)
        return pipeline

    engine = PaddleOcrEngine(pipeline_factory=factory)
    assert factory_calls == []

    tokens = engine.scan(_frame())

    assert len(factory_calls) == 1
    assert factory_calls[0]["text_detection_model_name"] == "PP-OCRv6_tiny_det"
    assert factory_calls[0]["text_recognition_model_name"] == "PP-OCRv6_tiny_rec"
    assert factory_calls[0]["engine"] == "onnxruntime"
    assert factory_calls[0]["device"] == "cpu"
    assert factory_calls[0]["use_doc_orientation_classify"] is False
    assert factory_calls[0]["use_doc_unwarping"] is False
    assert factory_calls[0]["use_textline_orientation"] is False
    assert factory_calls[0]["cpu_threads"] == OcrConfig().cpu_threads
    assert "apple_vision_max_input_dimension" not in factory_calls[0]
    assert "lang" not in factory_calls[0]
    assert "ocr_version" not in factory_calls[0]
    assert len(tokens) == 1
    assert tokens[0].text == "Card Number"
    assert tokens[0].confidence == pytest.approx(0.98)
    assert tokens[0].bbox == BoundingBox(x=0, y=10, width=105, height=21)
    assert pipeline.predict_calls[0]["text_rec_score_thresh"] == 0.20
    assert pipeline.predict_calls[0]["text_det_limit_side_len"] == 832
    assert pipeline.predict_calls[0]["input"].flags.c_contiguous


def test_adapter_uses_polygon_geometry_when_rectangles_are_absent() -> None:
    pipeline = FakePipeline(
        [
            FakeResult(
                {
                    "rec_texts": ["Expiry"],
                    "rec_scores": [0.91],
                    "rec_polys": [
                        [[10.2, 20.8], [80.7, 19.9], [81.1, 40.1], [9.8, 41.2]]
                    ],
                }
            )
        ]
    )

    tokens = PaddleOcrEngine(pipeline_factory=lambda **_: pipeline).scan(_frame())

    assert tokens[0].bbox == BoundingBox(x=9, y=19, width=73, height=23)


def test_adapter_accepts_empty_results() -> None:
    pipeline = FakePipeline(
        [FakeResult({"rec_texts": [], "rec_scores": [], "rec_boxes": []})]
    )

    assert PaddleOcrEngine(pipeline_factory=lambda **_: pipeline).scan(_frame()) == []


def test_adapter_downscales_large_frames_and_maps_boxes_to_source_pixels() -> None:
    pipeline = FakePipeline(
        [
            FakeResult(
                {
                    "rec_texts": ["Card Number"],
                    "rec_scores": [0.98],
                    "rec_boxes": [[10, 20, 100, 60]],
                }
            )
        ]
    )
    frame = np.zeros((2000, 1000, 3), dtype=np.uint8)
    engine = PaddleOcrEngine(
        OcrConfig(max_input_dimension=1000),
        pipeline_factory=lambda **_: pipeline,
    )

    tokens = engine.scan(frame)

    assert pipeline.predict_calls[0]["input"].shape == (1000, 500, 3)
    assert tokens[0].bbox == BoundingBox(x=20, y=40, width=180, height=80)


def test_adapter_retries_full_resolution_when_scaled_scan_has_no_text() -> None:
    class ResolutionAwarePipeline:
        def __init__(self) -> None:
            self.input_shapes: list[tuple[int, ...]] = []

        def predict(self, **kwargs: Any) -> Iterable[object]:
            shape = kwargs["input"].shape
            self.input_shapes.append(shape)
            if shape[0] == 1000:
                return [
                    FakeResult({"rec_texts": [], "rec_scores": [], "rec_boxes": []})
                ]
            return [
                FakeResult(
                    {
                        "rec_texts": ["CVV"],
                        "rec_scores": [0.95],
                        "rec_boxes": [[100, 200, 300, 260]],
                    }
                )
            ]

    pipeline = ResolutionAwarePipeline()
    engine = PaddleOcrEngine(
        OcrConfig(max_input_dimension=1000),
        pipeline_factory=lambda **_: pipeline,
    )

    tokens = engine.scan(np.zeros((2000, 1000, 3), dtype=np.uint8))

    assert pipeline.input_shapes == [(1000, 500, 3), (2000, 1000, 3)]
    assert tokens[0].bbox == BoundingBox(x=100, y=200, width=200, height=60)


def test_adapter_can_disable_empty_result_full_resolution_fallback() -> None:
    pipeline = FakePipeline(
        [FakeResult({"rec_texts": [], "rec_scores": [], "rec_boxes": []})]
    )
    engine = PaddleOcrEngine(
        OcrConfig(max_input_dimension=1000, full_resolution_fallback=False),
        pipeline_factory=lambda **_: pipeline,
    )

    assert engine.scan(np.zeros((2000, 1000, 3), dtype=np.uint8)) == []
    assert len(pipeline.predict_calls) == 1


def test_adapter_passes_local_model_directories() -> None:
    pipeline = FakePipeline(
        [FakeResult({"rec_texts": [], "rec_scores": [], "rec_boxes": []})]
    )
    options_seen: dict[str, Any] = {}

    def factory(**options: Any) -> FakePipeline:
        options_seen.update(options)
        return pipeline

    config = OcrConfig(
        text_detection_model_dir=Path("/models/detection"),
        text_recognition_model_dir=Path("/models/recognition"),
    )
    PaddleOcrEngine(config, pipeline_factory=factory).scan(_frame())

    assert options_seen["text_detection_model_dir"] == "/models/detection"
    assert options_seen["text_recognition_model_dir"] == "/models/recognition"


def test_adapter_can_delegate_language_and_version_to_paddle_defaults() -> None:
    pipeline = FakePipeline(
        [FakeResult({"rec_texts": [], "rec_scores": [], "rec_boxes": []})]
    )
    options_seen: dict[str, Any] = {}

    def factory(**options: Any) -> FakePipeline:
        options_seen.update(options)
        return pipeline

    config = OcrConfig(
        language="en",
        ocr_version="PP-OCRv5",
        text_detection_model_name=None,
        text_recognition_model_name=None,
    )
    PaddleOcrEngine(config, pipeline_factory=factory).scan(_frame())

    assert options_seen["lang"] == "en"
    assert options_seen["ocr_version"] == "PP-OCRv5"


def test_adapter_can_select_an_explicit_inference_engine() -> None:
    pipeline = FakePipeline(
        [FakeResult({"rec_texts": [], "rec_scores": [], "rec_boxes": []})]
    )
    options_seen: dict[str, Any] = {}

    def factory(**options: Any) -> FakePipeline:
        options_seen.update(options)
        return pipeline

    PaddleOcrEngine(
        OcrConfig(inference_engine="onnxruntime"),
        pipeline_factory=factory,
    ).scan(_frame())

    assert options_seen["engine"] == "onnxruntime"


@pytest.mark.parametrize(
    "payload",
    [
        {"rec_texts": ["CVV"], "rec_scores": []},
        {"rec_texts": ["CVV"], "rec_scores": [0.9]},
        {
            "rec_texts": ["CVV"],
            "rec_scores": [1.2],
            "rec_boxes": [[1, 1, 20, 20]],
        },
        {
            "rec_texts": ["CVV"],
            "rec_scores": [0.9],
            "rec_boxes": [[1, 1, 1, 20]],
        },
    ],
)
def test_adapter_fails_closed_on_malformed_results(payload: dict[str, Any]) -> None:
    pipeline = FakePipeline([FakeResult(payload)])

    with pytest.raises(OcrProcessingError):
        PaddleOcrEngine(pipeline_factory=lambda **_: pipeline).scan(_frame())


def test_inference_errors_do_not_echo_sensitive_text() -> None:
    class FailingPipeline:
        def predict(self, **kwargs: Any) -> Iterable[object]:
            raise RuntimeError("failed near 4242 4242 4242 4242")

    with pytest.raises(OcrProcessingError) as caught:
        PaddleOcrEngine(pipeline_factory=lambda **_: FailingPipeline()).scan(_frame())

    assert "4242" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_invalid_frame_is_rejected_before_pipeline_initialization() -> None:
    initialized = False

    def factory(**_: Any) -> FakePipeline:
        nonlocal initialized
        initialized = True
        return FakePipeline([])

    engine = PaddleOcrEngine(pipeline_factory=factory)

    with pytest.raises(OcrProcessingError, match="three-channel BGR"):
        engine.scan(np.zeros((20, 20), dtype=np.uint8))
    assert initialized is False


def test_pipeline_initialization_error_is_sanitized() -> None:
    def factory(**_: Any) -> FakePipeline:
        raise RuntimeError("configuration included 4242 4242 4242 4242")

    with pytest.raises(OcrUnavailableError) as caught:
        PaddleOcrEngine(pipeline_factory=factory).scan(_frame())

    assert "4242" not in str(caught.value)
    assert caught.value.__cause__ is None
