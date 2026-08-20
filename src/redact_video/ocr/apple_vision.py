"""Optional on-device Apple Vision OCR accelerator."""

from __future__ import annotations

import json
import math
import os
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import OcrConfig
from redact_video.exceptions import OcrProcessingError, OcrUnavailableError
from redact_video.models import BoundingBox, OcrToken

MAX_HELPER_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_HELPER_INPUT_BYTES = 64 * 1024 * 1024
MAX_TOKENS_PER_FRAME = 10_000
HELPER_TIMEOUT_SECONDS = 30

VisionRunner = Callable[[bytes], bytes]
VisionProcessFactory = Callable[..., subprocess.Popen[bytes]]


class AppleVisionOcrEngine:
    """Use macOS Vision OCR without persisting frames or recognized text."""

    backend_name = "apple_vision"

    def __init__(
        self,
        config: OcrConfig | None = None,
        *,
        runner: VisionRunner | None = None,
        process_factory: VisionProcessFactory | None = None,
    ) -> None:
        self.config = config or OcrConfig()
        self._runner = runner
        self._process_factory = process_factory or subprocess.Popen
        self._temporary_directory: TemporaryDirectory[str] | None = None
        self._helper_path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._disabled = False
        self._process_lock = threading.Lock()

    @staticmethod
    def is_supported() -> bool:
        return sys.platform == "darwin" and shutil.which("swiftc") is not None

    @classmethod
    def try_create(cls, config: OcrConfig | None = None) -> AppleVisionOcrEngine | None:
        if not cls.is_supported():
            return None
        engine = cls(config)
        try:
            engine._ensure_helper()
        except OcrUnavailableError:
            return None
        return engine

    def try_create_peer(self) -> AppleVisionOcrEngine | None:
        """Create an independent process wrapper without recompiling Swift."""

        peer_directory: TemporaryDirectory[str] | None = None
        try:
            with self._process_lock:
                if self._runner is not None or self._disabled:
                    return None
                compiled_helper = self._ensure_helper()

            job_root = Path(tempfile.gettempdir()) / "video-redactor"
            job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            peer_directory = TemporaryDirectory(
                prefix="apple-vision-peer.",
                dir=job_root,
            )
            peer_helper = Path(peer_directory.name) / "vision-ocr"
            try:
                os.link(compiled_helper, peer_helper)
            except OSError:
                shutil.copy2(compiled_helper, peer_helper)
            peer_helper.chmod(0o700)

            peer = AppleVisionOcrEngine(
                self.config,
                process_factory=self._process_factory,
            )
            peer._temporary_directory = peer_directory
            peer._helper_path = peer_helper
            return peer
        except Exception:  # noqa: BLE001 - local helper setup must be sanitized
            if peer_directory is not None:
                with suppress(OSError):
                    peer_directory.cleanup()
            return None

    def scan(self, frame: NDArray[np.uint8]) -> list[OcrToken]:
        """Recognize text in one BGR frame through an in-memory helper pipe."""

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
            tokens = self._scan_frame(
                scan_frame,
                frame_width=frame_width,
                frame_height=frame_height,
                coordinate_scale_x=coordinate_scale_x,
                coordinate_scale_y=coordinate_scale_y,
            )
            if (
                not tokens
                and scan_frame is not frame
                and self.config.apple_vision_full_resolution_fallback
            ):
                tokens = self._scan_frame(
                    frame,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
            return tokens
        except (OcrProcessingError, OcrUnavailableError):
            raise
        except Exception:  # noqa: BLE001 - native helper errors must be sanitized
            raise OcrProcessingError("Local Apple Vision OCR failed") from None

    def _prepare_scan_frame(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[NDArray[np.uint8], float, float]:
        max_dimension = self.config.max_input_dimension
        if max_dimension is not None:
            max_dimension = min(
                max_dimension,
                self.config.apple_vision_max_input_dimension,
            )
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

    def _scan_frame(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_width: int,
        frame_height: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
    ) -> list[OcrToken]:
        raw_result = self._run(self._encode_scan_frame(frame))
        try:
            return self._parse_tokens(
                raw_result,
                frame_width=frame_width,
                frame_height=frame_height,
                coordinate_scale_x=coordinate_scale_x,
                coordinate_scale_y=coordinate_scale_y,
            )
        except OcrProcessingError:
            if self._runner is None:
                with self._process_lock:
                    self._disable()
            raise

    @staticmethod
    def _encode_scan_frame(frame: NDArray[np.uint8]) -> bytes:
        """Encode losslessly, preferring bounded BMP for lower CPU overhead.

        The estimate is exact for OpenCV's 24-bit, four-byte-aligned BMP rows.
        PNG remains the bounded fallback for unusually large frames or codecs
        without BMP support.
        """

        frame_height, frame_width = frame.shape[:2]
        bmp_row_bytes = ((frame_width * 3 + 3) // 4) * 4
        estimated_bmp_bytes = 54 + bmp_row_bytes * frame_height
        if estimated_bmp_bytes <= MAX_HELPER_INPUT_BYTES:
            try:
                encoded, payload = cv2.imencode(".bmp", frame)
            except cv2.error:
                encoded = False
            if encoded:
                return payload.tobytes()

        encoded, payload = cv2.imencode(".png", frame)
        if not encoded:
            raise OcrProcessingError("Apple Vision input encoding failed")
        return payload.tobytes()

    def _ensure_helper(self) -> Path:
        if self._helper_path is not None:
            return self._helper_path
        if not self.is_supported():
            raise OcrUnavailableError("Apple Vision OCR is unavailable")

        job_root = Path(tempfile.gettempdir()) / "video-redactor"
        try:
            job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._temporary_directory = TemporaryDirectory(
                prefix="apple-vision.",
                dir=job_root,
            )
            helper_path = Path(self._temporary_directory.name) / "vision-ocr"
            source_path = Path(__file__).with_name("apple_vision_helper.swift")
            completed = subprocess.run(
                ["swiftc", "-O", str(source_path), "-o", str(helper_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=HELPER_TIMEOUT_SECONDS,
            )
            if completed.returncode != 0 or not helper_path.is_file():
                raise OcrUnavailableError("Apple Vision OCR could not initialize")
            helper_path.chmod(0o700)
        except OcrUnavailableError:
            raise
        except Exception:  # noqa: BLE001 - compiler/filesystem failures are sanitized
            raise OcrUnavailableError("Apple Vision OCR could not initialize") from None
        self._helper_path = helper_path
        return helper_path

    def _run(self, payload: bytes) -> bytes:
        if self._runner is not None:
            return self._runner(payload)
        if not payload or len(payload) > MAX_HELPER_INPUT_BYTES:
            raise OcrProcessingError("Apple Vision OCR input exceeds the local limit")
        with self._process_lock:
            process = self._ensure_process()
            try:
                return self._exchange(process, payload)
            except OcrProcessingError:
                self._disable()
                raise
            except Exception:  # noqa: BLE001 - helper failures must be sanitized
                self._disable()
                raise OcrProcessingError("Local Apple Vision OCR failed") from None

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self._disabled:
            raise OcrUnavailableError("Apple Vision OCR is unavailable")
        if self._process is not None:
            if self._process.poll() is None:
                return self._process
            self._stop_process()
            self._disabled = True
            raise OcrUnavailableError("Apple Vision OCR is unavailable")

        helper_path = self._ensure_helper()
        try:
            process = self._process_factory(
                [
                    str(helper_path),
                    self.config.apple_vision_recognition_level,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._process = process
            if process.stdin is None or process.stdout is None:
                raise OSError("native helper pipes unavailable")
            os.set_blocking(process.stdin.fileno(), False)
            os.set_blocking(process.stdout.fileno(), False)
        except Exception:  # noqa: BLE001 - process failures must be sanitized
            self._stop_process()
            self._disabled = True
            raise OcrUnavailableError("Apple Vision OCR could not initialize") from None
        return process

    def _exchange(self, process: subprocess.Popen[bytes], payload: bytes) -> bytes:
        if process.stdin is None or process.stdout is None:
            raise OcrProcessingError("Local Apple Vision OCR failed")
        deadline = time.monotonic() + HELPER_TIMEOUT_SECONDS
        request = struct.pack(">Q", len(payload)) + payload
        self._write_all(process.stdin.fileno(), request, deadline)
        response_length = struct.unpack(
            ">Q",
            self._read_exact(process.stdout.fileno(), 8, deadline),
        )[0]
        if response_length > MAX_HELPER_OUTPUT_BYTES:
            raise OcrProcessingError("Apple Vision OCR returned excessive metadata")
        return self._read_exact(process.stdout.fileno(), response_length, deadline)

    @staticmethod
    def _write_all(file_descriptor: int, payload: bytes, deadline: float) -> None:
        remaining = memoryview(payload)
        while remaining:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise OcrProcessingError("Local Apple Vision OCR timed out")
            _, writable, _ = select.select([], [file_descriptor], [], timeout)
            if not writable:
                raise OcrProcessingError("Local Apple Vision OCR timed out")
            try:
                written = os.write(file_descriptor, remaining)
            except BlockingIOError:
                continue
            if written <= 0:
                raise OcrProcessingError("Local Apple Vision OCR failed")
            remaining = remaining[written:]

    @staticmethod
    def _read_exact(file_descriptor: int, size: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < size:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise OcrProcessingError("Local Apple Vision OCR timed out")
            readable, _, _ = select.select([file_descriptor], [], [], timeout)
            if not readable:
                raise OcrProcessingError("Local Apple Vision OCR timed out")
            try:
                chunk = os.read(file_descriptor, size - len(result))
            except BlockingIOError:
                continue
            if not chunk:
                raise OcrProcessingError("Local Apple Vision OCR failed")
            result.extend(chunk)
        return bytes(result)

    def _disable(self) -> None:
        self._disabled = True
        self._stop_process()

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                with suppress(OSError):
                    process.kill()
                with suppress(OSError, subprocess.TimeoutExpired):
                    process.wait(timeout=1)
            except OSError:
                pass
        except OSError:
            pass
        if process.stdout is not None:
            with suppress(OSError):
                process.stdout.close()

    def close(self) -> None:
        """Stop the private helper and remove its compiled temporary files."""

        with self._process_lock:
            self._disabled = True
            self._stop_process()
            if self._temporary_directory is not None:
                with suppress(OSError):
                    self._temporary_directory.cleanup()
                self._temporary_directory = None
                self._helper_path = None

    def _parse_tokens(
        self,
        payload: bytes,
        *,
        frame_width: int,
        frame_height: int,
        coordinate_scale_x: float = 1.0,
        coordinate_scale_y: float = 1.0,
    ) -> list[OcrToken]:
        try:
            raw: Any = json.loads(payload)
        except Exception:  # noqa: BLE001 - helper output is untrusted
            raise OcrProcessingError(
                "Apple Vision OCR returned invalid metadata"
            ) from None
        if not isinstance(raw, list) or len(raw) > MAX_TOKENS_PER_FRAME:
            raise OcrProcessingError("Apple Vision OCR returned invalid metadata")

        tokens: list[OcrToken] = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise OcrProcessingError("Apple Vision OCR returned invalid metadata")
            text_value = item.get("text")
            confidence_value = item.get("confidence")
            bbox_value = item.get("bbox")
            if not isinstance(text_value, str):
                raise OcrProcessingError("Apple Vision OCR returned invalid metadata")
            text = text_value.strip()
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError):
                raise OcrProcessingError(
                    "Apple Vision OCR returned invalid metadata"
                ) from None
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise OcrProcessingError("Apple Vision OCR returned invalid metadata")
            if not text or confidence < self.config.minimum_confidence:
                continue
            if (
                not isinstance(bbox_value, Sequence)
                or isinstance(bbox_value, (str, bytes, bytearray))
                or len(bbox_value) != 4
            ):
                raise OcrProcessingError("Apple Vision OCR returned invalid metadata")
            try:
                left_value, top_value, right_value, bottom_value = (
                    float(value) for value in bbox_value
                )
            except (TypeError, ValueError):
                raise OcrProcessingError(
                    "Apple Vision OCR returned invalid metadata"
                ) from None
            if not all(
                math.isfinite(value)
                for value in (left_value, top_value, right_value, bottom_value)
            ):
                raise OcrProcessingError("Apple Vision OCR returned invalid metadata")
            left_value *= coordinate_scale_x
            right_value *= coordinate_scale_x
            top_value *= coordinate_scale_y
            bottom_value *= coordinate_scale_y
            left = max(0, min(math.floor(left_value), frame_width))
            top = max(0, min(math.floor(top_value), frame_height))
            right = max(0, min(math.ceil(right_value), frame_width))
            bottom = max(0, min(math.ceil(bottom_value), frame_height))
            if right <= left or bottom <= top:
                raise OcrProcessingError("Apple Vision OCR returned invalid metadata")
            tokens.append(
                OcrToken(
                    text=text,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x=left,
                        y=top,
                        width=right - left,
                        height=bottom - top,
                    ),
                )
            )
        return tokens
