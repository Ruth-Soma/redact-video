"""FFprobe-backed media inspection and input validation."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from redact_video.config import MediaLimits
from redact_video.exceptions import MediaProbeError, UnsupportedInputError

SUPPORTED_INPUT_SUFFIXES = frozenset({".mp4", ".mov"})


class MediaInfo(BaseModel):
    """Sanitized media properties required by the processing pipeline."""

    model_config = ConfigDict(frozen=True)

    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_ms: int = Field(gt=0)
    video_duration_ms: int = Field(gt=0)
    audio_duration_ms: int | None = Field(default=None, gt=0)
    frame_rate: float = Field(gt=0)
    nominal_frame_rate: float = Field(gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    rotation: int = 0
    video_codec: str
    pixel_format: str | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    is_variable_frame_rate: bool = False

    @property
    def display_width(self) -> int:
        return self.height if abs(self.rotation) % 180 == 90 else self.width

    @property
    def display_height(self) -> int:
        return self.width if abs(self.rotation) % 180 == 90 else self.height


def _parse_rate(value: Any) -> float:
    if value in (None, "", "0/0"):
        return 0.0
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return rate if math.isfinite(rate) and rate > 0 else 0.0


def _parse_duration_ms(*values: Any) -> int | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            milliseconds = round(float(value) * 1000)
        except (TypeError, ValueError):
            continue
        if milliseconds > 0:
            return milliseconds
    return None


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _rotation_for(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                return round(float(side_data["rotation"])) % 360
            except (TypeError, ValueError):
                pass
    try:
        return round(float(stream.get("tags", {}).get("rotate", 0))) % 360
    except (TypeError, ValueError):
        return 0


def _validate_dimensions(width: int, height: int, limits: MediaLimits) -> None:
    if width < limits.min_dimension or height < limits.min_dimension:
        raise UnsupportedInputError("Video dimensions are below the supported minimum")
    if width > limits.max_dimension or height > limits.max_dimension:
        raise UnsupportedInputError("Video dimensions exceed the supported maximum")
    if width * height > limits.max_pixels:
        raise UnsupportedInputError("Video pixel count exceeds the supported maximum")
    if width % 2 or height % 2:
        raise UnsupportedInputError(
            "Video dimensions must be even for lossless-size yuv420p output"
        )


def probe_media(
    path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    limits: MediaLimits | None = None,
) -> MediaInfo:
    """Inspect and validate an MP4 or MOV without decoding its frames."""

    input_path = Path(path)
    if input_path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
        raise UnsupportedInputError("Only MP4 and MOV inputs are supported")
    if not input_path.is_file():
        raise UnsupportedInputError("Input video does not exist or is not a file")
    if input_path.stat().st_size <= 0:
        raise UnsupportedInputError("Input video is empty")
    if shutil.which(ffprobe_binary) is None:
        raise MediaProbeError("ffprobe is not installed or is not available on PATH")

    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError("ffprobe could not inspect the input video") from exc
    if result.returncode != 0:
        raise UnsupportedInputError("Input is corrupt or could not be inspected")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe returned invalid metadata") from exc

    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise UnsupportedInputError("Input contains no video stream")
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)

    width = _parse_positive_int(video.get("width"))
    height = _parse_positive_int(video.get("height"))
    if width is None or height is None:
        raise UnsupportedInputError("Input video has invalid dimensions")
    _validate_dimensions(width, height, limits or MediaLimits())

    average_rate = _parse_rate(video.get("avg_frame_rate"))
    nominal_rate = _parse_rate(video.get("r_frame_rate"))
    frame_rate = average_rate or nominal_rate
    if frame_rate <= 0:
        raise UnsupportedInputError("Input video has an invalid frame rate")
    if nominal_rate <= 0:
        nominal_rate = frame_rate

    format_info = payload.get("format", {})
    video_duration_ms = _parse_duration_ms(
        video.get("duration"), format_info.get("duration")
    )
    if video_duration_ms is None:
        frame_count_for_duration = _parse_positive_int(video.get("nb_frames"))
        if frame_count_for_duration:
            video_duration_ms = round(frame_count_for_duration / frame_rate * 1000)
    if video_duration_ms is None or video_duration_ms <= 0:
        raise UnsupportedInputError("Input video has zero or unknown duration")

    audio_duration_ms = None
    if audio is not None:
        audio_duration_ms = _parse_duration_ms(
            audio.get("duration"), format_info.get("duration")
        )
    duration_ms = _parse_duration_ms(format_info.get("duration")) or video_duration_ms
    rate_delta = abs(average_rate - nominal_rate) if average_rate else 0.0
    is_vfr = bool(average_rate and rate_delta > max(0.05, average_rate * 0.01))

    return MediaInfo(
        path=input_path,
        width=width,
        height=height,
        duration_ms=duration_ms,
        video_duration_ms=video_duration_ms,
        audio_duration_ms=audio_duration_ms,
        frame_rate=frame_rate,
        nominal_frame_rate=nominal_rate,
        frame_count=_parse_positive_int(video.get("nb_frames")),
        rotation=_rotation_for(video),
        video_codec=str(video.get("codec_name") or "unknown"),
        pixel_format=video.get("pix_fmt"),
        has_audio=audio is not None,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        is_variable_frame_rate=is_vfr,
    )
