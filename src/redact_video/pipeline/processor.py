"""Phase 1 inspect/decode/mask/encode/mux pipeline."""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from redact_video.config import AppConfig
from redact_video.exceptions import ConfigurationError, DecodeError
from redact_video.media.decoder import FrameDecoder
from redact_video.media.encoder import (
    FrameEncoder,
    mux_original_audio,
    validate_encoded_media,
)
from redact_video.media.probe import probe_media
from redact_video.models import BoundingBox, JobStatus
from redact_video.redaction.renderer import BlurRenderer

ProgressCallback = Callable[[str], None]


class ProcessResult(BaseModel):
    """Sanitized outcome for the explicit Phase 1 manual-mask operation."""

    model_config = ConfigDict(frozen=True)

    status: JobStatus = JobStatus.NEEDS_REVIEW
    output_path: Path
    input_duration_ms: int = Field(gt=0)
    output_duration_ms: int = Field(gt=0)
    frames_processed: int = Field(gt=0)
    manual_regions: int = Field(gt=0)
    audio_preserved: bool
    variable_frame_rate_normalized: bool = False
    warnings: tuple[str, ...] = ()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.redacted.mp4")


def _validate_request(
    input_path: Path,
    output_path: Path,
    manual_boxes: Sequence[BoundingBox],
    *,
    overwrite: bool,
) -> None:
    if not manual_boxes:
        raise ConfigurationError("Manual rendering requires at least one --manual-box")
    if output_path.suffix.lower() != ".mp4":
        raise ConfigurationError("Output must use the .mp4 extension")
    try:
        same_path = input_path.resolve() == output_path.resolve()
    except OSError:
        same_path = input_path.absolute() == output_path.absolute()
    if same_path:
        raise ConfigurationError("Output path must not overwrite the input video")
    if output_path.exists() and not overwrite:
        raise ConfigurationError("Output already exists; use --overwrite to replace it")
    if output_path.exists() and not output_path.is_file():
        raise ConfigurationError("Output path exists and is not a file")


def _validate_boxes(
    boxes: Sequence[BoundingBox], frame_width: int, frame_height: int
) -> None:
    if any(box.clipped(frame_width, frame_height) is None for box in boxes):
        raise ConfigurationError(
            "Every manual box must intersect the displayed video frame"
        )


def _stage_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.partial.mp4")


def process_video(
    input_path: Path,
    *,
    output_path: Path | None = None,
    manual_boxes: Sequence[BoundingBox],
    config: AppConfig | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
) -> ProcessResult:
    """Render static manual masks while preserving media properties.

    This explicit development path proves Phase 1 media integrity. It reports
    ``NEEDS_REVIEW`` because automatic timeline rendering and verification are
    not yet implemented.
    """

    app_config = config or AppConfig()
    source = Path(input_path)
    destination = Path(output_path) if output_path else default_output_path(source)
    _validate_request(source, destination, manual_boxes, overwrite=overwrite)
    if progress:
        progress("Inspecting video...")
    source_info = probe_media(
        source,
        ffprobe_binary=app_config.ffprobe_binary,
        limits=app_config.media_limits,
    )
    _validate_boxes(manual_boxes, source_info.display_width, source_info.display_height)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged_output = _stage_output_path(destination)
    renderer = BlurRenderer(app_config.blur)
    frames_processed = 0
    audio_preserved = False
    output_published = False

    try:
        job_root = Path(tempfile.gettempdir()) / "video-redactor"
        job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{uuid.uuid4().hex}.", dir=job_root
        ) as job_dir_string:
            job_dir = Path(job_dir_string)
            job_dir.chmod(0o700)
            silent_video = job_dir / "rendered-video.mp4"
            if progress:
                progress("Decoding and rendering privacy masks...")
            with FrameDecoder(source, source_info.frame_rate) as decoder:
                frame_iterator = iter(decoder)
                try:
                    first_frame = next(frame_iterator)
                except StopIteration as exc:  # defensive; decoder raises on empty input
                    raise DecodeError(
                        "Input video contains no decodable frames"
                    ) from exc
                actual_height, actual_width = first_frame.image.shape[:2]
                _validate_boxes(manual_boxes, actual_width, actual_height)
                with FrameEncoder(
                    silent_video,
                    width=actual_width,
                    height=actual_height,
                    frame_rate=source_info.frame_rate,
                    ffmpeg_binary=app_config.ffmpeg_binary,
                    config=app_config.output,
                ) as encoder:
                    encoder.write(
                        renderer.apply(
                            first_frame.image,
                            list(manual_boxes),
                            in_place=True,
                        )
                    )
                    frames_processed = 1
                    for frame in frame_iterator:
                        encoder.write(
                            renderer.apply(
                                frame.image,
                                list(manual_boxes),
                                in_place=True,
                            )
                        )
                        frames_processed += 1
            if progress:
                progress("Muxing original audio...")
            audio_preserved = mux_original_audio(
                silent_video=silent_video,
                source_video=source,
                output_path=staged_output,
                source_info=source_info,
                ffmpeg_binary=app_config.ffmpeg_binary,
                config=app_config.output,
            )
        output_info = probe_media(
            staged_output,
            ffprobe_binary=app_config.ffprobe_binary,
            limits=app_config.media_limits,
        )
        validate_encoded_media(
            output_info,
            source_info,
            preserve_audio=app_config.output.preserve_audio,
        )
        os.replace(staged_output, destination)
        output_published = True
    finally:
        if not output_published:
            staged_output.unlink(missing_ok=True)

    warnings: list[str] = []
    if source_info.is_variable_frame_rate:
        warnings.append("Variable frame rate was normalized to deterministic CFR")
    if progress:
        progress(
            "Manual render finished; status is NEEDS_REVIEW because manual mode "
            "does not enter automatic output verification."
        )
    return ProcessResult(
        output_path=destination,
        input_duration_ms=source_info.duration_ms,
        output_duration_ms=output_info.duration_ms,
        frames_processed=frames_processed,
        manual_regions=len(manual_boxes),
        audio_preserved=audio_preserved,
        variable_frame_rate_normalized=source_info.is_variable_frame_rate,
        warnings=tuple(warnings),
    )
