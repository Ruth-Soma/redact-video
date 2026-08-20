"""Local PaddleOCR 3.x adapter.

PaddleOCR is imported lazily so probing, manual rendering, tests, and future
non-Paddle backends remain usable without the heavyweight inference runtime.
"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import OcrConfig
from redact_video.exceptions import OcrProcessingError, OcrUnavailableError
from redact_video.models import BoundingBox, OcrToken


class _PaddlePipeline(Protocol):
    def predict(self, **kwargs: Any) -> Iterable[object]: ...


PipelineFactory = Callable[..., _PaddlePipeline]


def _load_paddle_pipeline(**options: Any) -> _PaddlePipeline:
    try:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        module = importlib.import_module("paddleocr")
        paddle_ocr = module.PaddleOCR
        return paddle_ocr(**options)
    except Exception:  # noqa: BLE001 - third-party import/init must be sanitized
        raise OcrUnavailableError(
            "PaddleOCR could not initialize. Install the OCR extra and a compatible "
            "local PaddlePaddle 3.x inference runtime."
        ) from None


def _as_sequence(value: Any, *, field_name: str) -> Sequence[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise OcrProcessingError(f"PaddleOCR returned invalid {field_name} metadata")


def _result_payload(result: object) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        raw: Any = result
    else:
        try:
            raw = cast(Any, result).json
            if callable(raw):
                raw = raw()
        except Exception:  # noqa: BLE001 - third-party result access is untrusted
            raise OcrProcessingError(
                "PaddleOCR returned an unreadable result object"
            ) from None
    if not isinstance(raw, Mapping):
        raise OcrProcessingError("PaddleOCR returned invalid result metadata")
    nested = raw.get("res", raw)
    if not isinstance(nested, Mapping):
        raise OcrProcessingError("PaddleOCR returned invalid nested result metadata")
    return nested


def _bounding_box_from_geometry(
    geometry: Any,
    *,
    frame_width: int,
    frame_height: int,
    coordinate_scale_x: float = 1.0,
    coordinate_scale_y: float = 1.0,
) -> BoundingBox:
    try:
        coordinates = np.asarray(geometry, dtype=np.float64)
    except (TypeError, ValueError):
        raise OcrProcessingError(
            "PaddleOCR returned invalid bounding-box metadata"
        ) from None

    if coordinates.ndim == 1 and coordinates.shape == (4,):
        x_min, y_min, x_max, y_max = coordinates.tolist()
    elif (
        coordinates.ndim == 2
        and coordinates.shape[0] >= 3
        and coordinates.shape[1] == 2
    ):
        x_min = float(coordinates[:, 0].min())
        y_min = float(coordinates[:, 1].min())
        x_max = float(coordinates[:, 0].max())
        y_max = float(coordinates[:, 1].max())
    else:
        raise OcrProcessingError("PaddleOCR returned invalid bounding-box metadata")

    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise OcrProcessingError("PaddleOCR returned non-finite bounding-box metadata")
    left = max(0, min(math.floor(x_min * coordinate_scale_x), frame_width))
    top = max(0, min(math.floor(y_min * coordinate_scale_y), frame_height))
    right = max(0, min(math.ceil(x_max * coordinate_scale_x), frame_width))
    bottom = max(0, min(math.ceil(y_max * coordinate_scale_y), frame_height))
    if right <= left or bottom <= top:
        raise OcrProcessingError("PaddleOCR returned an empty bounding box")
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


class PaddleOcrEngine:
    """PaddleOCR 3.x implementation of the frame-oriented OCR protocol."""

    backend_name = "paddle"

    def __init__(
        self,
        config: OcrConfig | None = None,
        *,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.config = config or OcrConfig()
        self._pipeline_factory = pipeline_factory or _load_paddle_pipeline
        self._pipeline: _PaddlePipeline | None = None

    def _pipeline_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "device": self.config.device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "cpu_threads": self.config.cpu_threads,
            "enable_mkldnn": self.config.enable_mkldnn,
        }
        if self.config.inference_engine is not None:
            options["engine"] = self.config.inference_engine
        explicit_models = any(
            value is not None
            for value in (
                self.config.text_detection_model_name,
                self.config.text_recognition_model_name,
                self.config.text_detection_model_dir,
                self.config.text_recognition_model_dir,
            )
        )
        if explicit_models:
            if self.config.text_detection_model_name is not None:
                options["text_detection_model_name"] = (
                    self.config.text_detection_model_name
                )
            if self.config.text_recognition_model_name is not None:
                options["text_recognition_model_name"] = (
                    self.config.text_recognition_model_name
                )
        else:
            options["lang"] = self.config.language
            options["ocr_version"] = self.config.ocr_version
        if self.config.text_detection_model_dir is not None:
            options["text_detection_model_dir"] = str(
                self.config.text_detection_model_dir
            )
        if self.config.text_recognition_model_dir is not None:
            options["text_recognition_model_dir"] = str(
                self.config.text_recognition_model_dir
            )
        return options

    def _get_pipeline(self) -> _PaddlePipeline:
        if self._pipeline is None:
            try:
                self._pipeline = self._pipeline_factory(**self._pipeline_options())
            except OcrUnavailableError:
                raise
            except Exception:  # noqa: BLE001 - injected/vendor factories are untrusted
                raise OcrUnavailableError(
                    "The local PaddleOCR pipeline could not initialize"
                ) from None
        return self._pipeline

    def scan(self, frame: NDArray[np.uint8]) -> list[OcrToken]:
        """Scan one BGR frame and return transient, non-persisted tokens."""

        if not isinstance(frame, np.ndarray):
            raise OcrProcessingError("OCR expects a NumPy frame")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise OcrProcessingError("OCR expects an 8-bit, three-channel BGR frame")
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise OcrProcessingError("OCR cannot scan an empty frame")

        try:
            scan_frame, coordinate_scale_x, coordinate_scale_y = (
                self._prepare_scan_frame(frame)
            )
            tokens = self._predict(
                scan_frame,
                frame_width=frame_width,
                frame_height=frame_height,
                coordinate_scale_x=coordinate_scale_x,
                coordinate_scale_y=coordinate_scale_y,
            )
            if (
                not tokens
                and scan_frame is not frame
                and self.config.full_resolution_fallback
            ):
                tokens = self._predict(
                    frame,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            return tokens
        except (OcrProcessingError, OcrUnavailableError):
            raise
        except Exception:  # noqa: BLE001 - Paddle may raise backend-specific errors
            raise OcrProcessingError("Local PaddleOCR inference failed") from None

    def _prepare_scan_frame(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], float, float]:
        max_dimension = self.config.max_input_dimension
        frame_height, frame_width = frame.shape[:2]
        largest_dimension = max(frame_width, frame_height)
        if max_dimension is None or largest_dimension <= max_dimension:
            return frame, 1.0, 1.0

        resize_ratio = max_dimension / largest_dimension
        scan_width = max(1, round(frame_width * resize_ratio))
        scan_height = max(1, round(frame_height * resize_ratio))
        scan_frame = cv2.resize(
            frame,
            (scan_width, scan_height),
            interpolation=cv2.INTER_AREA,
        )
        return (
            scan_frame,
            frame_width / scan_width,
            frame_height / scan_height,
        )

    def _predict(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_width: int,
        frame_height: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
    ) -> list[OcrToken]:
        results = self._get_pipeline().predict(
            input=np.ascontiguousarray(frame),
            text_rec_score_thresh=self.config.minimum_confidence,
            text_det_limit_side_len=self.config.text_detection_limit_side_length,
        )
        if results is None:
            raise OcrProcessingError("PaddleOCR returned no result collection")
        tokens: list[OcrToken] = []
        for result in results:
            payload = _result_payload(result)
            tokens.extend(
                self._tokens_from_payload(
                    payload,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    coordinate_scale_x=coordinate_scale_x,
                    coordinate_scale_y=coordinate_scale_y,
                )
            )
        return tokens

    def _tokens_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        frame_width: int,
        frame_height: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
    ) -> list[OcrToken]:
        if "rec_texts" not in payload or "rec_scores" not in payload:
            raise OcrProcessingError("PaddleOCR result is missing recognition metadata")
        texts = _as_sequence(payload["rec_texts"], field_name="text")
        scores = _as_sequence(payload["rec_scores"], field_name="confidence")
        geometry_value = payload.get("rec_boxes")
        if geometry_value is None:
            geometry_value = payload.get("rec_polys")
        if geometry_value is None:
            if not texts and not scores:
                return []
            raise OcrProcessingError("PaddleOCR result is missing geometry metadata")
        geometries = _as_sequence(geometry_value, field_name="geometry")
        if not (len(texts) == len(scores) == len(geometries)):
            raise OcrProcessingError("PaddleOCR result metadata lengths do not match")

        tokens: list[OcrToken] = []
        for text_value, score_value, geometry in zip(
            texts, scores, geometries, strict=True
        ):
            if not isinstance(text_value, str):
                raise OcrProcessingError(
                    "PaddleOCR returned non-text recognition metadata"
                )
            text = text_value.strip()
            try:
                confidence = float(score_value)
            except (TypeError, ValueError):
                raise OcrProcessingError(
                    "PaddleOCR returned invalid confidence metadata"
                ) from None
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise OcrProcessingError(
                    "PaddleOCR returned out-of-range confidence metadata"
                )
            if not text or confidence < self.config.minimum_confidence:
                continue
            bbox = _bounding_box_from_geometry(
                geometry,
                frame_width=frame_width,
                frame_height=frame_height,
                coordinate_scale_x=coordinate_scale_x,
                coordinate_scale_y=coordinate_scale_y,
            )
            tokens.append(OcrToken(text=text, confidence=confidence, bbox=bbox))
        return tokens
