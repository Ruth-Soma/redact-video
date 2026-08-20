"""Automatic detection, rendering, verification, and repair pipeline."""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from collections.abc import Callable
from itertools import chain
from math import ceil
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field

from redact_video.config import AppConfig
from redact_video.detection import DetectionEngine
from redact_video.exceptions import ConfigurationError, DecodeError
from redact_video.media.decoder import FrameDecoder
from redact_video.media.encoder import (
    FrameEncoder,
    TimedRepairOverlay,
    apply_opaque_repair_overlays,
    mux_original_audio,
    validate_encoded_media,
)
from redact_video.media.probe import MediaInfo, probe_media
from redact_video.models import (
    BoundingBox,
    JobStatus,
    ProcessingTimings,
    SensitiveType,
)
from redact_video.ocr.apple_vision import AppleVisionOcrEngine
from redact_video.ocr.base import OcrEngine
from redact_video.ocr.paddle import PaddleOcrEngine
from redact_video.pipeline.processor import default_output_path
from redact_video.redaction.renderer import BlurRenderer
from redact_video.redaction.timeline import MaskTimeline, MaskTimelineBuilder
from redact_video.tracking import AdaptiveTrackingEngine
from redact_video.verification import (
    RepairEngine,
    RepairWindow,
    SafetyVerifier,
    VerificationLeak,
    VerificationSummary,
    build_repair_windows,
)

ProgressCallback = Callable[[str], None]


class AutomatedProcessResult(BaseModel):
    """Sanitized automatic redaction and verification result."""

    model_config = ConfigDict(frozen=True)

    status: JobStatus = JobStatus.NEEDS_REVIEW
    output_path: Path
    input_duration_ms: int = Field(gt=0)
    output_duration_ms: int = Field(gt=0)
    frames_processed: int = Field(gt=0)
    masked_frames: int = Field(ge=0)
    mask_regions: int = Field(ge=0)
    tracks_by_type: dict[SensitiveType, int] = Field(default_factory=dict)
    verification: VerificationSummary
    audio_preserved: bool
    variable_frame_rate_normalized: bool = False
    warnings: tuple[str, ...] = ()
    timings: ProcessingTimings
    mask_timeline: MaskTimeline = Field(exclude=True, repr=False)


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _validate_request(
    input_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
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


def _stage_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.partial.mp4")


def _source_detection_pre_roll_frames(
    tracker: AdaptiveTrackingEngine,
    *,
    frame_rate: float,
) -> int:
    if tracker.fast_ocr_engine is None:
        return 0
    slowest_scan_fps = min(
        tracker.detection_config.scan_fps,
        tracker.detection_config.safe_scan_fps,
    )
    coverage_ms = (
        tracker.tracking_config.authoritative_audit_interval_ms
        + 1000.0 / slowest_scan_fps
    )
    return ceil(frame_rate * coverage_ms / 1000)


def _analyze_source(
    source: Path,
    source_info: MediaInfo,
    tracker: AdaptiveTrackingEngine,
) -> tuple[MaskTimeline, int]:
    tracker.reset()
    detection_pre_roll_frames = _source_detection_pre_roll_frames(
        tracker,
        frame_rate=source_info.frame_rate,
    )
    timeline_builder = MaskTimelineBuilder(
        detection_pre_roll_frames=detection_pre_roll_frames
    )
    frames_processed = 0
    with FrameDecoder(source, source_info.frame_rate) as decoder:
        for frame in decoder:
            tracking = tracker.process_frame(
                frame.image,
                frame_index=frame.index,
                timestamp_ms=frame.timestamp_ms,
            )
            timeline_builder.add(tracking)
            frames_processed += 1
    return timeline_builder.build(), frames_processed


def _render_candidate(
    source: Path,
    source_info: MediaInfo,
    timeline: MaskTimeline,
    *,
    silent_video: Path,
    staged_output: Path,
    renderer: BlurRenderer,
    config: AppConfig,
) -> tuple[int, bool]:
    rendered_frames = 0
    with FrameDecoder(source, source_info.frame_rate) as decoder:
        frame_iterator = iter(decoder)
        try:
            first_frame = next(frame_iterator)
        except StopIteration as exc:  # defensive; decoder raises on empty input
            raise DecodeError("Input video contains no decodable frames") from exc
        actual_height, actual_width = first_frame.image.shape[:2]
        with FrameEncoder(
            silent_video,
            width=actual_width,
            height=actual_height,
            frame_rate=source_info.frame_rate,
            ffmpeg_binary=config.ffmpeg_binary,
            config=config.output,
        ) as encoder:
            for frame in chain((first_frame,), frame_iterator):
                encoder.write(
                    renderer.apply_timeline(
                        frame.image,
                        timeline,
                        frame_index=frame.index,
                        in_place=True,
                    )
                )
                rendered_frames += 1
    audio_preserved = mux_original_audio(
        silent_video=silent_video,
        source_video=source,
        output_path=staged_output,
        source_info=source_info,
        ffmpeg_binary=config.ffmpeg_binary,
        config=config.output,
    )
    return rendered_frames, audio_preserved


def _validate_candidate(
    candidate: Path,
    source_info: MediaInfo,
    config: AppConfig,
) -> MediaInfo:
    output_info = probe_media(
        candidate,
        ffprobe_binary=config.ffprobe_binary,
        limits=config.media_limits,
    )
    validate_encoded_media(
        output_info,
        source_info,
        preserve_audio=config.output.preserve_audio,
    )
    return output_info


def _build_opaque_repair_overlays(
    leaks: tuple[VerificationLeak, ...],
    windows: tuple[RepairWindow, ...],
    *,
    frame_width: int,
    frame_height: int,
    padding_px: int,
) -> tuple[TimedRepairOverlay, ...]:
    overlays: list[TimedRepairOverlay] = []
    for window in windows:
        boxes = [
            leak.bbox.expanded(padding_px, padding_px)
            for leak in leaks
            if window.start_ms <= leak.timestamp_ms <= window.end_ms
        ]
        if not boxes:
            continue
        left = min(box.x for box in boxes)
        top = min(box.y for box in boxes)
        right = max(box.right for box in boxes)
        bottom = max(box.bottom for box in boxes)
        union = BoundingBox(
            x=left,
            y=top,
            width=right - left,
            height=bottom - top,
        ).clipped(frame_width, frame_height)
        if union is not None:
            overlays.append(
                TimedRepairOverlay(
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    bbox=union,
                )
            )
    return tuple(overlays)


def process_video_automated(
    input_path: Path,
    *,
    output_path: Path | None = None,
    config: AppConfig | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
    ocr_engine: OcrEngine | None = None,
    verification_ocr_engine: OcrEngine | None = None,
    tracking_engine: AdaptiveTrackingEngine | None = None,
) -> AutomatedProcessResult:
    """Automatically detect, render, verify, and repair card-region masks."""

    total_started = time.perf_counter()
    inspection_started = time.perf_counter()
    app_config = config or AppConfig()
    source = Path(input_path)
    destination = Path(output_path) if output_path else default_output_path(source)
    _validate_request(source, destination, overwrite=overwrite)
    if tracking_engine is not None and ocr_engine is not None:
        raise ConfigurationError(
            "Provide either a tracking engine or an OCR engine, not both"
        )
    if progress:
        progress("Inspecting video...")
    source_info = probe_media(
        source,
        ffprobe_binary=app_config.ffprobe_binary,
        limits=app_config.media_limits,
    )
    inspection_ms = _elapsed_ms(inspection_started)
    setup_started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged_output = _stage_output_path(destination)
    renderer = BlurRenderer(app_config.blur)
    active_tracker = tracking_engine
    native_ocr: AppleVisionOcrEngine | None = None
    secondary_native_ocr: AppleVisionOcrEngine | None = None
    if active_tracker is None:
        source_ocr = (
            PaddleOcrEngine(app_config.ocr) if ocr_engine is None else ocr_engine
        )
        if getattr(source_ocr, "backend_name", None) == "paddle" and (
            app_config.tracking.use_apple_vision_acceleration
            or (
                app_config.verification.enabled
                and verification_ocr_engine is None
                and app_config.verification.use_apple_vision_acceleration
            )
        ):
            native_ocr = AppleVisionOcrEngine.try_create(app_config.ocr)
        active_tracker = AdaptiveTrackingEngine(
            source_ocr,
            detection_engine=DetectionEngine(app_config.detection),
            detection_config=app_config.detection,
            tracking_config=app_config.tracking,
            fast_ocr_engine=(
                native_ocr
                if app_config.tracking.use_apple_vision_acceleration
                else None
            ),
            include_track_snapshots=False,
        )
    else:
        source_ocr = active_tracker.ocr_engine
    verifier_ocr = (
        source_ocr if verification_ocr_engine is None else verification_ocr_engine
    )
    fast_verifier_ocr: OcrEngine | None = None
    if (
        app_config.verification.enabled
        and verification_ocr_engine is None
        and app_config.verification.use_apple_vision_acceleration
        and getattr(source_ocr, "backend_name", None) == "paddle"
    ):
        if native_ocr is None:
            native_ocr = AppleVisionOcrEngine.try_create(app_config.ocr)
        fast_verifier_ocr = native_ocr
        if fast_verifier_ocr is not None:
            if isinstance(fast_verifier_ocr, AppleVisionOcrEngine):
                secondary_candidate = fast_verifier_ocr.try_create_peer()
                if secondary_candidate is None:
                    secondary_candidate = AppleVisionOcrEngine.try_create(
                        app_config.ocr
                    )
            else:
                secondary_candidate = AppleVisionOcrEngine.try_create(app_config.ocr)
            if secondary_candidate is not fast_verifier_ocr:
                secondary_native_ocr = secondary_candidate
    setup_ms = _elapsed_ms(setup_started)
    if app_config.tracking.optical_flow_workers > 1:
        cv2.setNumThreads(0)
    frames_processed = 0
    audio_preserved = False
    status = JobStatus.NEEDS_REVIEW
    verification_passes = 0
    repairs = 0
    remaining_leaks = 0
    potential_issues = ()
    output_published = False
    source_analysis_ms = 0
    initial_render_ms = 0
    initial_validation_ms = 0
    verification_passes_ms: list[int] = []
    repair_passes_ms: list[int] = []
    publication_ms = 0

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
                progress("Scanning and tracking sensitive regions...")
            source_analysis_started = time.perf_counter()
            timeline, frames_processed = _analyze_source(
                source,
                source_info,
                active_tracker,
            )
            source_analysis_ms = _elapsed_ms(source_analysis_started)
            if progress:
                progress("Rendering privacy mask timeline...")
            initial_render_started = time.perf_counter()
            rendered_frames, audio_preserved = _render_candidate(
                source,
                source_info,
                timeline,
                silent_video=silent_video,
                staged_output=staged_output,
                renderer=renderer,
                config=app_config,
            )
            initial_render_ms = _elapsed_ms(initial_render_started)
            if rendered_frames != frames_processed:
                raise DecodeError(
                    "Frame count changed between automatic analysis and rendering"
                )
            initial_validation_started = time.perf_counter()
            _validate_candidate(staged_output, source_info, app_config)
            initial_validation_ms = _elapsed_ms(initial_validation_started)

            if app_config.verification.enabled:
                verifier = SafetyVerifier(
                    verifier_ocr,
                    detection_config=app_config.detection,
                    verification_config=app_config.verification,
                    fast_ocr_engine=fast_verifier_ocr,
                    secondary_fast_ocr_engine=secondary_native_ocr,
                )
                repair_engine = RepairEngine(
                    source_ocr,
                    detection_config=app_config.detection,
                    tracking_config=app_config.tracking,
                    verification_config=app_config.verification,
                    fast_ocr_engine=fast_verifier_ocr,
                )
                for pass_number in range(1, app_config.verification.max_passes + 1):
                    if progress:
                        progress(
                            f"Verification pass {pass_number}/"
                            f"{app_config.verification.max_passes}..."
                        )
                    verification_started = time.perf_counter()
                    pass_result = verifier.verify(
                        staged_output,
                        frame_rate=source_info.frame_rate,
                        pass_number=pass_number,
                    )
                    verification_passes_ms.append(_elapsed_ms(verification_started))
                    verification_passes = pass_number
                    remaining_leaks = len(pass_result.leaks)
                    if pass_result.passed:
                        status = JobStatus.COMPLETED
                        potential_issues = ()
                        break

                    windows = build_repair_windows(
                        pass_result.leaks,
                        duration_ms=source_info.duration_ms,
                        config=app_config.verification,
                    )
                    if pass_number == app_config.verification.max_passes:
                        potential_issues = windows
                        break
                    repair_started = time.perf_counter()
                    repairs += len(windows)
                    if progress:
                        progress(f"Repairing {len(windows)} temporal window(s)...")
                    timeline = repair_engine.patch_leak_seeds(
                        timeline,
                        pass_result.leaks,
                        windows,
                        frame_rate=source_info.frame_rate,
                        frame_count=frames_processed,
                        pass_number=pass_number,
                    )
                    overlays = _build_opaque_repair_overlays(
                        pass_result.leaks,
                        windows,
                        frame_width=source_info.display_width,
                        frame_height=source_info.display_height,
                        padding_px=(
                            app_config.verification.repair_padding_px
                            + app_config.blur.padding_for(
                                source_info.display_width,
                                source_info.display_height,
                            )
                        ),
                    )
                    repaired_candidate = job_dir / f"repair-pass-{pass_number}.mp4"
                    apply_opaque_repair_overlays(
                        candidate=staged_output,
                        output_path=repaired_candidate,
                        overlays=overlays,
                        ffmpeg_binary=app_config.ffmpeg_binary,
                        config=app_config.output,
                    )
                    _validate_candidate(repaired_candidate, source_info, app_config)
                    os.replace(repaired_candidate, staged_output)
                    repair_passes_ms.append(_elapsed_ms(repair_started))

        publication_started = time.perf_counter()
        output_info = _validate_candidate(staged_output, source_info, app_config)
        os.replace(staged_output, destination)
        output_published = True
        publication_ms = _elapsed_ms(publication_started)
    finally:
        if not output_published:
            staged_output.unlink(missing_ok=True)
        active_tracker.close()
        if secondary_native_ocr is not None:
            secondary_native_ocr.close()
        if native_ocr is not None:
            native_ocr.close()

    masked_frames = len(timeline.frames)
    mask_regions = sum(len(regions) for regions in timeline.frames.values())
    tracks_by_type: dict[SensitiveType, int] = {}
    for track in active_tracker.tracker.tracks:
        tracks_by_type[track.type] = tracks_by_type.get(track.type, 0) + 1
    warnings: list[str] = []
    if not app_config.verification.enabled:
        warnings.append("Automatic output verification is disabled")
    elif status is JobStatus.NEEDS_REVIEW:
        warnings.append(
            "Automatic verification found possible leaks after the maximum passes"
        )
    if source_info.is_variable_frame_rate:
        warnings.append("Variable frame rate was normalized to deterministic CFR")
    if progress:
        progress(
            "Automatic redaction completed and verified."
            if status is JobStatus.COMPLETED
            else "Automatic redaction finished with status NEEDS_REVIEW."
        )
    timings = ProcessingTimings(
        inspection_ms=inspection_ms,
        setup_ms=setup_ms,
        source_analysis_ms=source_analysis_ms,
        initial_render_ms=initial_render_ms,
        initial_validation_ms=initial_validation_ms,
        verification_passes_ms=tuple(verification_passes_ms),
        repair_passes_ms=tuple(repair_passes_ms),
        publication_ms=publication_ms,
        total_ms=_elapsed_ms(total_started),
    )
    return AutomatedProcessResult(
        status=status,
        output_path=destination,
        input_duration_ms=source_info.duration_ms,
        output_duration_ms=output_info.duration_ms,
        frames_processed=frames_processed,
        masked_frames=masked_frames,
        mask_regions=mask_regions,
        tracks_by_type=tracks_by_type,
        verification=VerificationSummary(
            passes=verification_passes,
            repairs=repairs,
            remaining_leaks=remaining_leaks,
            potential_issues=potential_issues,
        ),
        audio_preserved=audio_preserved,
        variable_frame_rate_normalized=source_info.is_variable_frame_rate,
        warnings=tuple(warnings),
        timings=timings,
        mask_timeline=timeline,
    )
