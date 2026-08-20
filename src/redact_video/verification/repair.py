"""Temporal repair-window construction and mask-timeline patching."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path

from redact_video.config import DetectionConfig, TrackingConfig, VerificationConfig
from redact_video.detection import DetectionEngine
from redact_video.media.decoder import FrameDecoder
from redact_video.models import SensitiveType
from redact_video.ocr.base import OcrEngine
from redact_video.redaction.timeline import MaskRegion, MaskTimeline
from redact_video.tracking import AdaptiveTrackingEngine
from redact_video.verification.models import RepairWindow, VerificationLeak
from redact_video.verification.verifier import sensitive_detection_config


def build_repair_windows(
    leaks: tuple[VerificationLeak, ...],
    *,
    duration_ms: int,
    config: VerificationConfig,
) -> tuple[RepairWindow, ...]:
    """Expand leak timestamps and deterministically merge overlapping windows."""

    if not leaks:
        return ()
    raw = sorted(
        (
            max(0, leak.timestamp_ms - config.repair_pre_roll_ms),
            min(duration_ms, leak.timestamp_ms + config.repair_post_roll_ms),
            leak.type,
        )
        for leak in leaks
    )
    merged: list[tuple[int, int, set[SensitiveType]]] = []
    for start_ms, end_ms, sensitive_type in raw:
        if merged and start_ms <= merged[-1][1]:
            previous_start, previous_end, previous_types = merged[-1]
            previous_types.add(sensitive_type)
            merged[-1] = (
                previous_start,
                max(previous_end, end_ms),
                previous_types,
            )
        else:
            merged.append((start_ms, end_ms, {sensitive_type}))
    return tuple(
        RepairWindow(
            start_ms=start_ms,
            end_ms=end_ms,
            types=tuple(sorted(types, key=lambda item: item.value)),
        )
        for start_ms, end_ms, types in merged
    )


class RepairEngine:
    """Rescan source windows at every frame and patch a mask timeline."""

    def __init__(
        self,
        ocr_engine: OcrEngine,
        *,
        detection_config: DetectionConfig,
        tracking_config: TrackingConfig,
        verification_config: VerificationConfig,
        fast_ocr_engine: OcrEngine | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.fast_ocr_engine = fast_ocr_engine
        self.detection_config = sensitive_detection_config(
            detection_config,
            verification_config,
        )
        self.tracking_config = tracking_config
        self.verification_config = verification_config

    def patch_leak_seeds(
        self,
        timeline: MaskTimeline,
        leaks: tuple[VerificationLeak, ...],
        windows: tuple[RepairWindow, ...],
        *,
        frame_rate: float,
        frame_count: int,
        pass_number: int,
    ) -> MaskTimeline:
        """Patch every repair-window frame with conservative verified leak seeds."""

        patched: dict[int, list[MaskRegion]] = {
            frame_index: list(regions)
            for frame_index, regions in timeline.frames.items()
        }
        for window_index, window in enumerate(windows):
            window_leaks = tuple(
                leak
                for leak in leaks
                if window.start_ms <= leak.timestamp_ms <= window.end_ms
            )
            first_frame = max(0, floor(window.start_ms * frame_rate / 1000))
            last_frame = min(
                frame_count - 1,
                ceil(window.end_ms * frame_rate / 1000),
            )
            for frame_index in range(first_frame, last_frame + 1):
                additions = patched.setdefault(frame_index, [])
                for leak_index, leak in enumerate(window_leaks):
                    additions.append(
                        MaskRegion(
                            track_id=(
                                f"repair-p{pass_number:02d}-w{window_index:04d}-"
                                f"leak-{leak_index:04d}"
                            ),
                            type=leak.type,
                            bbox=leak.bbox.expanded(
                                self.verification_config.repair_padding_px,
                                self.verification_config.repair_padding_px,
                            ),
                        )
                    )
        return _finalize_timeline(patched)

    def patch_timeline(
        self,
        source: Path,
        timeline: MaskTimeline,
        leaks: tuple[VerificationLeak, ...],
        windows: tuple[RepairWindow, ...],
        *,
        frame_rate: float,
        pass_number: int,
    ) -> MaskTimeline:
        """Return a new timeline with direct leak seeds and rescanned regions."""

        patched: dict[int, list[MaskRegion]] = {
            frame_index: list(regions)
            for frame_index, regions in timeline.frames.items()
        }
        tracker = AdaptiveTrackingEngine(
            self.ocr_engine,
            detection_engine=DetectionEngine(self.detection_config),
            detection_config=self.detection_config,
            tracking_config=self.tracking_config,
            fast_ocr_engine=self.fast_ocr_engine,
            include_track_snapshots=False,
        )
        active_window_index: int | None = None
        try:
            with FrameDecoder(source, frame_rate) as decoder:
                for frame in decoder:
                    window_index = next(
                        (
                            index
                            for index, window in enumerate(windows)
                            if window.start_ms <= frame.timestamp_ms <= window.end_ms
                        ),
                        None,
                    )
                    if window_index is None:
                        continue
                    if window_index != active_window_index:
                        tracker.reset()
                        active_window_index = window_index
                    tracking = tracker.process_frame(
                        frame.image,
                        frame_index=frame.index,
                        timestamp_ms=frame.timestamp_ms,
                    )
                    additions = patched.setdefault(frame.index, [])
                    for region_index, region in enumerate(tracking.regions):
                        additions.append(
                            MaskRegion(
                                track_id=(
                                    f"repair-p{pass_number:02d}-"
                                    f"w{window_index:04d}-"
                                    f"track-{region_index:06d}"
                                ),
                                type=region.type,
                                bbox=region.bbox,
                            )
                        )
                    window = windows[window_index]
                    for leak_index, leak in enumerate(leaks):
                        if not (window.start_ms <= leak.timestamp_ms <= window.end_ms):
                            continue
                        additions.append(
                            MaskRegion(
                                track_id=(
                                    f"repair-p{pass_number:02d}-"
                                    f"w{window_index:04d}-"
                                    f"leak-{leak_index:04d}"
                                ),
                                type=leak.type,
                                bbox=leak.bbox.expanded(
                                    self.verification_config.repair_padding_px,
                                    self.verification_config.repair_padding_px,
                                ),
                            ),
                        )
        finally:
            tracker.close()

        return _finalize_timeline(patched)


def _finalize_timeline(patched: dict[int, list[MaskRegion]]) -> MaskTimeline:
    return MaskTimeline(
        frames={
            frame_index: tuple(
                sorted(
                    _deduplicate(regions),
                    key=lambda item: (
                        item.bbox.y,
                        item.bbox.x,
                        item.type.value,
                        item.track_id,
                    ),
                )
            )
            for frame_index, regions in sorted(patched.items())
            if regions
        }
    )


def _deduplicate(regions: list[MaskRegion]) -> list[MaskRegion]:
    selected: list[MaskRegion] = []
    seen: set[tuple[SensitiveType, int, int, int, int]] = set()
    for region in regions:
        key = (
            region.type,
            region.bbox.x,
            region.bbox.y,
            region.bbox.width,
            region.bbox.height,
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(region)
    return selected
