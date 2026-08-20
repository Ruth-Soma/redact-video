"""Sequential video frame decoding."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self

import cv2
import numpy as np

from redact_video.exceptions import DecodeError
from redact_video.media.probe import probe_media


@dataclass(frozen=True)
class VideoFrame:
    """A decoded video frame and its source timestamp."""

    index: int
    timestamp_ms: int
    image: np.ndarray


class FrameDecoder:
    """Decode display-oriented frames while preserving timeline timestamps.

    Constant-frame-rate inputs use FFmpeg's raw-video pipe. This avoids the
    substantial per-frame overhead observed in OpenCV's VideoCapture on macOS.
    Variable-frame-rate inputs retain the OpenCV path because it exposes each
    decoded frame's presentation timestamp.
    """

    def __init__(self, path: Path, fallback_frame_rate: float) -> None:
        self.path = path
        self.fallback_frame_rate = fallback_frame_rate
        self._capture: cv2.VideoCapture | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._frame_size = 0

    def __enter__(self) -> Self:
        media = probe_media(
            self.path,
            ffprobe_binary=os.getenv("REDACT_VIDEO_FFPROBE", "ffprobe"),
        )
        if media.is_variable_frame_rate:
            self._open_opencv()
            return self

        self._frame_width = media.display_width
        self._frame_height = media.display_height
        self._frame_size = self._frame_width * self._frame_height * 3
        ffmpeg_binary = os.getenv("REDACT_VIDEO_FFMPEG", "ffmpeg")
        if shutil.which(ffmpeg_binary) is None:
            raise DecodeError("FFmpeg is required to decode video frames")

        command = [
            ffmpeg_binary,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(self.path),
            "-map",
            "0:v:0",
            "-an",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise DecodeError("Could not start FFmpeg video decoder") from error

        if self._process.stdout is None:
            self.close()
            raise DecodeError("Could not open FFmpeg video stream")
        return self

    def _open_opencv(self) -> None:
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise DecodeError("Could not open input video stream")
        # OpenCV builds differ in whether orientation metadata is applied by
        # default. Request it explicitly so detection and render coordinates
        # always match the display-oriented dimensions returned by ffprobe.
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self._capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    def __iter__(self) -> Iterator[VideoFrame]:
        if self._capture is not None:
            yield from self._iter_opencv()
            return
        if self._process is None or self._process.stdout is None:
            raise DecodeError("Video decoder is not open")

        index = 0
        while True:
            payload = self._read_frame()
            if payload is None:
                break
            image = np.frombuffer(payload, dtype=np.uint8).reshape(
                self._frame_height,
                self._frame_width,
                3,
            )
            timestamp_ms = round(index * 1000.0 / self.fallback_frame_rate)
            yield VideoFrame(index=index, timestamp_ms=timestamp_ms, image=image)
            index += 1

        process = self._process
        self._process = None
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise DecodeError("FFmpeg failed while decoding the video stream")

    def _read_frame(self) -> bytearray | None:
        assert self._process is not None
        assert self._process.stdout is not None
        payload = bytearray(self._frame_size)
        view = memoryview(payload)
        offset = 0
        while offset < self._frame_size:
            count = self._process.stdout.readinto(view[offset:])
            if not count:
                if offset == 0:
                    return None
                raise DecodeError("FFmpeg returned an incomplete video frame")
            offset += count
        return payload

    def _iter_opencv(self) -> Iterator[VideoFrame]:
        assert self._capture is not None
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            timestamp_ms = round(self._capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0 and index > 0:
                timestamp_ms = round(index * 1000.0 / self.fallback_frame_rate)
            yield VideoFrame(index=index, timestamp_ms=timestamp_ms, image=frame)
            index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.kill()
                process.wait()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class CompactFrameDecoder:
    """Decode a small grayscale stream for cross-pass visual guards."""

    def __init__(
        self,
        path: Path,
        fallback_frame_rate: float,
        *,
        max_dimension: int,
    ) -> None:
        self.path = path
        self.fallback_frame_rate = fallback_frame_rate
        self.max_dimension = max_dimension
        self._process: subprocess.Popen[bytes] | None = None
        self._frame_width = 0
        self._frame_height = 0
        self._frame_size = 0

    def __enter__(self) -> Self:
        media = probe_media(
            self.path,
            ffprobe_binary=os.getenv("REDACT_VIDEO_FFPROBE", "ffprobe"),
        )
        scale = min(
            1.0,
            self.max_dimension / max(media.display_width, media.display_height),
        )
        self._frame_width = max(1, round(media.display_width * scale))
        self._frame_height = max(1, round(media.display_height * scale))
        self._frame_size = self._frame_width * self._frame_height
        ffmpeg_binary = os.getenv("REDACT_VIDEO_FFMPEG", "ffmpeg")
        if shutil.which(ffmpeg_binary) is None:
            raise DecodeError("FFmpeg is required to decode video frames")

        command = [
            ffmpeg_binary,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(self.path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (f"scale={self._frame_width}:{self._frame_height}:flags=area,format=gray"),
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise DecodeError("Could not start FFmpeg video decoder") from error
        if self._process.stdout is None:
            self.close()
            raise DecodeError("Could not open FFmpeg video stream")
        return self

    def __iter__(self) -> Iterator[VideoFrame]:
        if self._process is None or self._process.stdout is None:
            raise DecodeError("Video decoder is not open")

        index = 0
        while True:
            payload = _read_exact_frame(self._process.stdout, self._frame_size)
            if payload is None:
                break
            image = np.frombuffer(payload, dtype=np.uint8).reshape(
                self._frame_height,
                self._frame_width,
            )
            timestamp_ms = round(index * 1000.0 / self.fallback_frame_rate)
            yield VideoFrame(index=index, timestamp_ms=timestamp_ms, image=image)
            index += 1
        self._finish_process()

    def _finish_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        assert process.stdout is not None
        process.stdout.close()
        if process.wait() != 0:
            raise DecodeError("FFmpeg failed while decoding the video stream")

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.kill()
                process.wait()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


class SelectedFrameDecoder(FrameDecoder):
    """Decode only selected absolute frame indices from a validated candidate."""

    _MAX_SELECTION_TERMS = 48

    def __init__(
        self,
        path: Path,
        fallback_frame_rate: float,
        *,
        frame_indices: tuple[int, ...],
    ) -> None:
        if not frame_indices:
            raise ValueError("At least one selected frame index is required")
        if tuple(sorted(set(frame_indices))) != frame_indices or frame_indices[0] < 0:
            raise ValueError("Selected frame indices must be unique and ordered")
        super().__init__(path, fallback_frame_rate)
        self.frame_indices = frame_indices

    def __enter__(self) -> Self:
        media = probe_media(
            self.path,
            ffprobe_binary=os.getenv("REDACT_VIDEO_FFPROBE", "ffprobe"),
        )
        if media.is_variable_frame_rate:
            self._open_opencv()
            return self

        self._frame_width = media.display_width
        self._frame_height = media.display_height
        self._frame_size = self._frame_width * self._frame_height * 3
        ffmpeg_binary = os.getenv("REDACT_VIDEO_FFMPEG", "ffmpeg")
        if shutil.which(ffmpeg_binary) is None:
            raise DecodeError("FFmpeg is required to decode video frames")
        return self

    def _start_selected_process(self, frame_indices: tuple[int, ...]) -> None:
        ffmpeg_binary = os.getenv("REDACT_VIDEO_FFMPEG", "ffmpeg")
        selection = _selection_expression(frame_indices)
        command = [
            ffmpeg_binary,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(self.path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"select={selection}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            raise DecodeError("Could not start FFmpeg video decoder") from error
        if self._process.stdout is None:
            self.close()
            raise DecodeError("Could not open FFmpeg video stream")

    def __iter__(self) -> Iterator[VideoFrame]:
        if self._capture is not None:
            yield from self._iter_selected_opencv()
            return
        if self._frame_size <= 0:
            raise DecodeError("Video decoder is not open")

        for frame_indices in _selection_chunks(
            self.frame_indices,
            max_terms=self._MAX_SELECTION_TERMS,
        ):
            self._start_selected_process(frame_indices)
            assert self._process is not None
            assert self._process.stdout is not None
            process = self._process
            for frame_index in frame_indices:
                payload = self._read_frame()
                if payload is None:
                    self.close()
                    raise DecodeError("FFmpeg omitted a selected video frame")
                image = np.frombuffer(payload, dtype=np.uint8).reshape(
                    self._frame_height,
                    self._frame_width,
                    3,
                )
                timestamp_ms = round(frame_index * 1000.0 / self.fallback_frame_rate)
                yield VideoFrame(
                    index=frame_index,
                    timestamp_ms=timestamp_ms,
                    image=image,
                )

            self._process = None
            process.stdout.close()
            if process.wait() != 0:
                raise DecodeError("FFmpeg failed while decoding the video stream")

    def _iter_selected_opencv(self) -> Iterator[VideoFrame]:
        assert self._capture is not None
        selected = set(self.frame_indices)
        last_selected = self.frame_indices[-1]
        index = 0
        while index <= last_selected:
            ok, frame = self._capture.read()
            if not ok:
                break
            if index in selected:
                timestamp_ms = round(self._capture.get(cv2.CAP_PROP_POS_MSEC))
                if timestamp_ms <= 0 and index > 0:
                    timestamp_ms = round(index * 1000.0 / self.fallback_frame_rate)
                yield VideoFrame(
                    index=index,
                    timestamp_ms=timestamp_ms,
                    image=frame,
                )
            index += 1
        if index <= last_selected:
            raise DecodeError("OpenCV omitted a selected video frame")


def _read_exact_frame(
    stream: BinaryIO,
    frame_size: int,
) -> bytearray | None:
    """Read one exact raw frame from a binary stream."""

    payload = bytearray(frame_size)
    view = memoryview(payload)
    offset = 0
    while offset < frame_size:
        count = stream.readinto(view[offset:])
        if not count:
            if offset == 0:
                return None
            raise DecodeError("FFmpeg returned an incomplete video frame")
        offset += count
    return payload


def _selection_expression(frame_indices: tuple[int, ...]) -> str:
    """Return a compact FFmpeg select expression for ordered frame indices."""

    ranges: list[tuple[int, int]] = []
    start = previous = frame_indices[0]
    for frame_index in frame_indices[1:]:
        if frame_index == previous + 1:
            previous = frame_index
            continue
        ranges.append((start, previous))
        start = previous = frame_index
    ranges.append((start, previous))
    return "+".join(
        (f"eq(n\\,{start})" if start == end else f"between(n\\,{start}\\,{end})")
        for start, end in ranges
    )


def _selection_chunks(
    frame_indices: tuple[int, ...],
    *,
    max_terms: int,
) -> tuple[tuple[int, ...], ...]:
    """Split indices before FFmpeg's expression evaluator becomes too deep."""

    chunks: list[tuple[int, ...]] = []
    chunk_start = 0
    terms = 1
    for position in range(1, len(frame_indices)):
        if frame_indices[position] == frame_indices[position - 1] + 1:
            continue
        if terms < max_terms:
            terms += 1
            continue
        chunks.append(frame_indices[chunk_start:position])
        chunk_start = position
        terms = 1
    chunks.append(frame_indices[chunk_start:])
    return tuple(chunks)
