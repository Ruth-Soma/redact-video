"""FFmpeg-backed H.264 encoding and source-audio muxing."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray

from redact_video.config import OutputConfig
from redact_video.exceptions import EncodeError
from redact_video.media.probe import MediaInfo
from redact_video.models import BoundingBox

MP4_COPY_AUDIO_CODECS = frozenset({"aac", "ac3", "alac", "eac3", "mp3"})
MAX_DURATION_DRIFT_MS = 100


@dataclass(frozen=True, slots=True)
class TimedRepairOverlay:
    """One opaque, value-free fail-safe region active over a time interval."""

    start_ms: int
    end_ms: int
    bbox: BoundingBox


def _require_ffmpeg(ffmpeg_binary: str) -> None:
    if shutil.which(ffmpeg_binary) is None:
        raise EncodeError("FFmpeg is not installed or is not available on PATH")


class FrameEncoder:
    """Stream fixed-size BGR frames into an H.264 MP4."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        frame_rate: float,
        ffmpeg_binary: str,
        config: OutputConfig,
    ) -> None:
        if width % 2 or height % 2:
            raise EncodeError("Output dimensions must be even for yuv420p")
        self.path = Path(path)
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.ffmpeg_binary = ffmpeg_binary
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self.frames_written = 0

    def __enter__(self) -> Self:
        _require_ffmpeg(self.ffmpeg_binary)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            f"{self.frame_rate:.9f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.config.codec,
            "-preset",
            self.config.preset,
            "-crf",
            str(self.config.crf),
            "-pix_fmt",
            self.config.pixel_format,
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise EncodeError("FFmpeg could not start the video encoder") from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def write(self, frame: NDArray[np.uint8]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("FrameEncoder must be used as a context manager")
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise EncodeError("Frame shape or pixel type changed during encoding")
        try:
            payload = memoryview(np.ascontiguousarray(frame)).cast("B")
            while payload:
                written = self._process.stdin.write(payload)
                if written is None or written <= 0:
                    raise BrokenPipeError
                payload = payload[written:]
        except (BrokenPipeError, OSError) as exc:
            self.abort()
            raise EncodeError("FFmpeg stopped while encoding video frames") from exc
        self.frames_written += 1

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code != 0:
            self.path.unlink(missing_ok=True)
            detail = stderr.decode("utf-8", errors="replace").strip()
            message = "FFmpeg failed to encode the processed video"
            if detail:
                message = f"{message}: {detail[-500:]}"
            raise EncodeError(message)
        if self.frames_written == 0 or not self.path.is_file():
            self.path.unlink(missing_ok=True)
            raise EncodeError("Encoder produced no video frames")

    def abort(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.path.unlink(missing_ok=True)


def mux_original_audio(
    *,
    silent_video: Path,
    source_video: Path,
    output_path: Path,
    source_info: MediaInfo,
    ffmpeg_binary: str,
    config: OutputConfig,
) -> bool:
    """Create the staged final MP4 and return whether audio was preserved."""

    _require_ffmpeg(ffmpeg_binary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.preserve_audio or not source_info.has_audio:
        os.replace(silent_video, output_path)
        return False

    pad_audio = (
        source_info.audio_duration_ms is not None
        and source_info.video_duration_ms - source_info.audio_duration_ms
        >= MAX_DURATION_DRIFT_MS
    )
    copy_audio = source_info.audio_codec in MP4_COPY_AUDIO_CODECS and not pad_audio
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy" if copy_audio else "aac",
    ]
    if not copy_audio:
        command.extend(["-b:a", config.audio_bitrate])
    if pad_audio:
        command.extend(["-af", "apad"])
    command.extend(
        [
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        output_path.unlink(missing_ok=True)
        raise EncodeError("FFmpeg could not mux the source audio") from exc
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        message = "FFmpeg failed to mux the source audio"
        if detail:
            message = f"{message}: {detail[-500:]}"
        raise EncodeError(message)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise EncodeError("Audio muxing produced no output")
    silent_video.unlink(missing_ok=True)
    return True


def apply_opaque_repair_overlays(
    *,
    candidate: Path,
    output_path: Path,
    overlays: tuple[TimedRepairOverlay, ...],
    ffmpeg_binary: str,
    config: OutputConfig,
) -> None:
    """Burn verified leak windows into an existing candidate with FFmpeg."""

    if not overlays:
        raise EncodeError("Automatic repair requires at least one overlay")
    _require_ffmpeg(ffmpeg_binary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    for overlay in overlays:
        start_seconds = overlay.start_ms / 1000
        end_seconds = overlay.end_ms / 1000
        box = overlay.bbox
        enable = f"between(t\\,{start_seconds:.6f}\\,{end_seconds:.6f})"
        filters.append(
            "drawbox="
            f"x={box.x}:y={box.y}:w={box.width}:h={box.height}:"
            f"color=black:t=fill:enable={enable}"
        )
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(candidate),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(filters),
        "-c:v",
        config.codec,
        "-preset",
        config.preset,
        "-crf",
        str(config.crf),
        "-pix_fmt",
        config.pixel_format,
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        output_path.unlink(missing_ok=True)
        raise EncodeError("FFmpeg could not apply automatic repair masks") from exc
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise EncodeError("FFmpeg failed while applying automatic repair masks")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise EncodeError("Automatic repair produced no output")


def validate_encoded_media(
    output_info: MediaInfo,
    source_info: MediaInfo,
    *,
    preserve_audio: bool,
) -> None:
    """Reject encoded media with changed geometry, duration, or required audio."""

    if output_info.display_width != source_info.display_width:
        raise EncodeError("Output width does not match the displayed input width")
    if output_info.display_height != source_info.display_height:
        raise EncodeError("Output height does not match the displayed input height")
    if (
        abs(output_info.video_duration_ms - source_info.video_duration_ms)
        >= MAX_DURATION_DRIFT_MS
    ):
        raise EncodeError("Output video duration does not match the input video")
    if not source_info.has_audio or not preserve_audio:
        return
    if not output_info.has_audio or output_info.audio_duration_ms is None:
        raise EncodeError("Source audio was not preserved")
    if (
        abs(output_info.video_duration_ms - output_info.audio_duration_ms)
        >= MAX_DURATION_DRIFT_MS
    ):
        raise EncodeError("Output audio and video durations are not synchronized")
