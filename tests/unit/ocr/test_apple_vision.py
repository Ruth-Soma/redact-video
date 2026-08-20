from __future__ import annotations

import json
import os
import struct
import subprocess
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from redact_video.config import OcrConfig
from redact_video.exceptions import OcrProcessingError, OcrUnavailableError
from redact_video.models import BoundingBox
from redact_video.ocr.apple_vision import AppleVisionOcrEngine


def _frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


class _FakePipe:
    def __init__(self) -> None:
        self.closed = False

    def fileno(self) -> int:
        return 42

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakePipe()
        self.stdout = _FakePipe()
        self.wait_calls = 0

    def poll(self) -> None:
        return None

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.wait_calls += 1
        return 0

    def terminate(self) -> None:
        raise AssertionError("graceful helper shutdown should not terminate")

    def kill(self) -> None:
        raise AssertionError("graceful helper shutdown should not kill")


def _read_file_descriptor(file_descriptor: int, size: int) -> bytes | None:
    result = bytearray()
    while len(result) < size:
        chunk = os.read(file_descriptor, size - len(result))
        if not chunk:
            return None
        result.extend(chunk)
    return bytes(result)


class _ProtocolProcess:
    def __init__(self, response: bytes = b"[]") -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        self.stdin = os.fdopen(request_write, "wb", buffering=0)
        self.stdout = os.fdopen(response_read, "rb", buffering=0)
        self.requests: list[bytes] = []
        self._request_read = request_read
        self._response_write = response_write
        self._response = response
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            while True:
                header = _read_file_descriptor(self._request_read, 8)
                if header is None:
                    return
                payload = _read_file_descriptor(
                    self._request_read,
                    struct.unpack(">Q", header)[0],
                )
                if payload is None:
                    return
                self.requests.append(payload)
                os.write(
                    self._response_write,
                    struct.pack(">Q", len(self._response)) + self._response,
                )
        finally:
            os.close(self._request_read)
            os.close(self._response_write)

    def poll(self) -> int | None:
        return None if self._thread.is_alive() else 0

    def wait(self, *, timeout: float) -> int:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired("vision-ocr", timeout)
        return 0

    def terminate(self) -> None:
        raise AssertionError("graceful helper shutdown should not terminate")

    def kill(self) -> None:
        raise AssertionError("graceful helper shutdown should not kill")


def test_adapter_parses_tokens_and_clips_boxes_to_source_frame() -> None:
    payload = json.dumps(
        [
            {
                "text": " Card Number ",
                "confidence": 0.98,
                "bbox": [-2.2, 20.8, 50.1, 40.2],
            },
            {
                "text": "729",
                "confidence": 0.10,
                "bbox": [60, 20, 90, 40],
            },
        ]
    ).encode()
    inputs: list[bytes] = []

    def runner(image: bytes) -> bytes:
        inputs.append(image)
        return payload

    tokens = AppleVisionOcrEngine(runner=runner).scan(_frame())

    assert len(inputs) == 1
    assert inputs[0].startswith(b"BM")
    assert len(tokens) == 1
    assert tokens[0].text == "Card Number"
    assert tokens[0].confidence == pytest.approx(0.98)
    assert tokens[0].bbox == BoundingBox(x=0, y=20, width=51, height=21)


def test_adapter_uses_png_when_bmp_would_exceed_helper_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs: list[bytes] = []
    monkeypatch.setattr(
        "redact_video.ocr.apple_vision.MAX_HELPER_INPUT_BYTES",
        1_000,
    )

    engine = AppleVisionOcrEngine(
        runner=lambda image: inputs.append(image) or b"[]",
    )

    assert engine.scan(_frame()) == []
    assert inputs[0].startswith(b"\x89PNG")


def test_adapter_bmp_transport_preserves_every_source_pixel() -> None:
    frame = (np.arange(60_000, dtype=np.uint32) % 251).astype(np.uint8)
    frame = frame.reshape(100, 200, 3)
    transported: list[np.ndarray] = []

    def runner(image: bytes) -> bytes:
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        transported.append(decoded)
        return b"[]"

    assert AppleVisionOcrEngine(runner=runner).scan(frame) == []
    assert np.array_equal(transported[0], frame)


def test_adapter_falls_back_to_png_when_bmp_encoding_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs: list[bytes] = []
    original_imencode = cv2.imencode

    def imencode(extension: str, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        if extension == ".bmp":
            return False, np.array([], dtype=np.uint8)
        return original_imencode(extension, frame)

    monkeypatch.setattr("redact_video.ocr.apple_vision.cv2.imencode", imencode)
    engine = AppleVisionOcrEngine(
        runner=lambda image: inputs.append(image) or b"[]",
    )

    assert engine.scan(_frame()) == []
    assert inputs[0].startswith(b"\x89PNG")


def test_adapter_scales_oversized_input_and_maps_boxes_to_source_frame() -> None:
    input_shapes: list[tuple[int, ...]] = []
    payload = json.dumps(
        [
            {
                "text": "Card Number",
                "confidence": 0.98,
                "bbox": [10, 20, 100, 60],
            }
        ]
    ).encode()

    def runner(image: bytes) -> bytes:
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        input_shapes.append(decoded.shape)
        return payload

    engine = AppleVisionOcrEngine(
        OcrConfig(
            max_input_dimension=1000,
            apple_vision_full_resolution_fallback=True,
        ),
        runner=runner,
    )

    tokens = engine.scan(np.zeros((2000, 1000, 3), dtype=np.uint8))

    assert input_shapes == [(1000, 500, 3)]
    assert tokens[0].bbox == BoundingBox(x=20, y=40, width=180, height=80)


def test_adapter_applies_the_native_cap_below_the_shared_ocr_ceiling() -> None:
    frame = np.zeros((2736, 1260, 3), dtype=np.uint8)
    engine = AppleVisionOcrEngine(
        OcrConfig(
            max_input_dimension=1920,
            apple_vision_max_input_dimension=1440,
        ),
        runner=lambda _: b"[]",
    )

    scan_frame, scale_x, scale_y = engine._prepare_scan_frame(frame)

    assert scan_frame.shape == (1440, 663, 3)
    assert scale_x == pytest.approx(1260 / 663)
    assert scale_y == pytest.approx(2736 / 1440)


def test_shared_ocr_ceiling_can_reduce_apple_vision_further() -> None:
    frame = np.zeros((2000, 1000, 3), dtype=np.uint8)
    engine = AppleVisionOcrEngine(
        OcrConfig(
            max_input_dimension=1000,
            apple_vision_max_input_dimension=1440,
        ),
        runner=lambda _: b"[]",
    )

    scan_frame, _, _ = engine._prepare_scan_frame(frame)

    assert scan_frame.shape == (1000, 500, 3)


def test_null_shared_ocr_ceiling_disables_apple_vision_scaling() -> None:
    frame = np.zeros((2000, 1000, 3), dtype=np.uint8)
    engine = AppleVisionOcrEngine(
        OcrConfig(max_input_dimension=None),
        runner=lambda _: b"[]",
    )

    scan_frame, scale_x, scale_y = engine._prepare_scan_frame(frame)

    assert scan_frame is frame
    assert scale_x == 1
    assert scale_y == 1


def test_adapter_retries_full_resolution_when_scaled_scan_has_no_text() -> None:
    input_shapes: list[tuple[int, ...]] = []

    def runner(image: bytes) -> bytes:
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        input_shapes.append(decoded.shape)
        if decoded.shape[0] == 1000:
            return b"[]"
        return json.dumps(
            [
                {
                    "text": "CVV",
                    "confidence": 0.95,
                    "bbox": [100, 200, 300, 260],
                }
            ]
        ).encode()

    engine = AppleVisionOcrEngine(
        OcrConfig(
            max_input_dimension=1000,
            apple_vision_full_resolution_fallback=True,
        ),
        runner=runner,
    )

    tokens = engine.scan(np.zeros((2000, 1000, 3), dtype=np.uint8))

    assert input_shapes == [(1000, 500, 3), (2000, 1000, 3)]
    assert tokens[0].bbox == BoundingBox(x=100, y=200, width=200, height=60)


def test_adapter_does_not_retry_empty_result_at_full_resolution_by_default() -> None:
    input_shapes: list[tuple[int, ...]] = []

    def runner(image: bytes) -> bytes:
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        input_shapes.append(decoded.shape)
        return b"[]"

    engine = AppleVisionOcrEngine(
        OcrConfig(max_input_dimension=1000),
        runner=runner,
    )

    assert engine.scan(np.zeros((2000, 1000, 3), dtype=np.uint8)) == []
    assert input_shapes == [(1000, 500, 3)]


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json-4242",
        json.dumps({"text": "4242"}).encode(),
        json.dumps([{"text": "4242", "confidence": 2, "bbox": [1, 2, 3, 4]}]).encode(),
        json.dumps(
            [{"text": "4242", "confidence": 0.9, "bbox": [1, 2, 1, 4]}]
        ).encode(),
    ],
)
def test_adapter_rejects_malformed_metadata_without_echoing_text(
    payload: bytes,
) -> None:
    engine = AppleVisionOcrEngine(runner=lambda _: payload)

    with pytest.raises(OcrProcessingError) as caught:
        engine.scan(_frame())

    assert "4242" not in str(caught.value)


def test_runner_errors_are_sanitized() -> None:
    def runner(_: bytes) -> bytes:
        raise RuntimeError("failed near 4242 4242 4242 4242")

    with pytest.raises(OcrProcessingError) as caught:
        AppleVisionOcrEngine(runner=runner).scan(_frame())

    assert "4242" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_persistent_helper_protocol_is_reused_and_closed() -> None:
    process = _ProtocolProcess()
    factory_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def process_factory(*args: Any, **kwargs: Any) -> _ProtocolProcess:
        factory_calls.append((args, kwargs))
        return process

    engine = AppleVisionOcrEngine(process_factory=process_factory)  # type: ignore[arg-type]
    engine._helper_path = Path("/private/vision-ocr")

    assert engine._run(b"first") == b"[]"
    assert engine._run(b"second") == b"[]"
    engine.close()

    assert len(factory_calls) == 1
    assert factory_calls[0][0][0] == ["/private/vision-ocr", "fast"]
    assert process.requests == [b"first", b"second"]
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_accurate_recognition_level_is_forwarded_to_native_helper() -> None:
    process = _ProtocolProcess()
    commands: list[list[str]] = []

    def process_factory(command: list[str], **_: Any) -> _ProtocolProcess:
        commands.append(command)
        return process

    engine = AppleVisionOcrEngine(
        OcrConfig(apple_vision_recognition_level="accurate"),
        process_factory=process_factory,  # type: ignore[arg-type]
    )
    engine._helper_path = Path("/private/vision-ocr")

    assert engine._run(b"frame") == b"[]"
    engine.close()

    assert commands == [["/private/vision-ocr", "accurate"]]


def test_peer_reuses_compiled_helper_with_independent_cleanup(tmp_path: Path) -> None:
    compiled_helper = tmp_path / "vision-ocr"
    compiled_helper.write_bytes(b"compiled-swift-helper")
    compiled_helper.chmod(0o700)
    engine = AppleVisionOcrEngine()
    engine._helper_path = compiled_helper

    peer = engine.try_create_peer()

    assert peer is not None
    assert peer is not engine
    assert peer.config is engine.config
    assert peer._helper_path is not None
    assert peer._helper_path != compiled_helper
    assert peer._helper_path.read_bytes() == compiled_helper.read_bytes()
    peer_helper = peer._helper_path

    peer.close()

    assert peer_helper.exists() is False
    assert compiled_helper.exists() is True


def test_persistent_helper_failure_disables_instance_without_exposing_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    factory_calls = 0

    def process_factory(*args: Any, **kwargs: Any) -> _FakeProcess:
        nonlocal factory_calls
        del args, kwargs
        factory_calls += 1
        return process

    engine = AppleVisionOcrEngine(process_factory=process_factory)  # type: ignore[arg-type]
    engine._helper_path = Path("/private/vision-ocr")
    monkeypatch.setattr(
        "redact_video.ocr.apple_vision.os.set_blocking", lambda *_: None
    )

    def fail_exchange(*_: object) -> bytes:
        raise RuntimeError("native failure near 4242 4242 4242 4242")

    monkeypatch.setattr(engine, "_exchange", fail_exchange)

    with pytest.raises(OcrProcessingError) as caught:
        engine._run(b"frame")
    with pytest.raises(OcrUnavailableError):
        engine._run(b"next-frame")

    assert "4242" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert factory_calls == 1
    assert process.wait_calls == 1


def test_persistent_helper_invalid_metadata_disables_instance() -> None:
    process = _ProtocolProcess(b"not-json-4242")
    engine = AppleVisionOcrEngine(
        process_factory=lambda *args, **kwargs: process  # type: ignore[arg-type]
    )
    engine._helper_path = Path("/private/vision-ocr")

    with pytest.raises(OcrProcessingError) as caught:
        engine.scan(_frame())
    with pytest.raises(OcrUnavailableError):
        engine.scan(_frame())

    assert "4242" not in str(caught.value)
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_persistent_helper_rejects_oversized_input_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_called = False

    def process_factory(*args: Any, **kwargs: Any) -> _FakeProcess:
        nonlocal factory_called
        del args, kwargs
        factory_called = True
        return _FakeProcess()

    engine = AppleVisionOcrEngine(process_factory=process_factory)  # type: ignore[arg-type]
    monkeypatch.setattr("redact_video.ocr.apple_vision.MAX_HELPER_INPUT_BYTES", 2)

    with pytest.raises(OcrProcessingError, match="local limit"):
        engine._run(b"too large")

    assert factory_called is False


def test_invalid_frame_is_rejected_before_runner() -> None:
    called = False

    def runner(_: bytes) -> bytes:
        nonlocal called
        called = True
        return b"[]"

    with pytest.raises(OcrProcessingError, match="three-channel BGR"):
        AppleVisionOcrEngine(runner=runner).scan(np.zeros((20, 20), dtype=np.uint8))

    assert called is False


def test_try_create_returns_none_when_platform_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AppleVisionOcrEngine,
        "is_supported",
        staticmethod(lambda: False),
    )

    assert AppleVisionOcrEngine.try_create(OcrConfig()) is None
