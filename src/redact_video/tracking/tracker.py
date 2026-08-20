"""Sensitive-region state machine and adaptive OCR reacquisition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import DetectionConfig, TrackingConfig
from redact_video.detection import DetectionEngine
from redact_video.detection.geometry import box_distance, intersection_over_union
from redact_video.exceptions import OcrProcessingError, OcrUnavailableError
from redact_video.models import (
    BoundingBox,
    OcrToken,
    SensitiveDetection,
    SensitiveTrack,
    SensitiveType,
    TrackedRegion,
    TrackingFrameResult,
    TrackState,
)
from redact_video.ocr.base import OcrEngine
from redact_video.tracking.optical_flow import MotionEstimate, OpticalFlowEstimator
from redact_video.tracking.scene_change import SceneChangeDetector
from redact_video.tracking.template_match import TemplateMatcher


class DetectionProvider(Protocol):
    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]: ...


@dataclass(slots=True)
class _MutableTrack:
    id: str
    type: SensitiveType
    start_frame: int
    confidence: float
    regions: dict[int, BoundingBox] = field(default_factory=dict)
    state: TrackState = TrackState.DETECTED
    end_frame: int | None = None
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    sticky_since_ms: int | None = None

    @property
    def active(self) -> bool:
        return self.state in {
            TrackState.DETECTED,
            TrackState.TRACKING,
            TrackState.STICKY,
        }

    @property
    def current_frame(self) -> int:
        return next(reversed(self.regions))

    @property
    def current_bbox(self) -> BoundingBox:
        return self.regions[self.current_frame]

    def bbox_before(self, frame_index: int) -> BoundingBox | None:
        previous_frame = next(
            (index for index in reversed(self.regions) if index < frame_index),
            None,
        )
        return self.regions[previous_frame] if previous_frame is not None else None

    def snapshot(self) -> SensitiveTrack:
        return SensitiveTrack(
            id=self.id,
            type=self.type,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            confidence=self.confidence,
            regions=dict(self.regions),
            state=self.state,
        )


def _validate_frame(frame: NDArray[np.uint8]) -> None:
    if (
        frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        raise ValueError("Tracking expects a non-empty 8-bit, three-channel BGR frame")


def _move_box(
    bbox: BoundingBox,
    dx: float,
    dy: float,
    *,
    frame_width: int,
    frame_height: int,
) -> BoundingBox | None:
    moved_x = round(bbox.x + dx)
    moved_y = round(bbox.y + dy)
    right = moved_x + bbox.width
    bottom = moved_y + bbox.height
    if right <= 0 or bottom <= 0 or moved_x >= frame_width or moved_y >= frame_height:
        return None
    left = max(0, moved_x)
    top = max(0, moved_y)
    return BoundingBox(
        x=left,
        y=top,
        width=min(frame_width, right) - left,
        height=min(frame_height, bottom) - top,
    )


class RegionTracker:
    """Track sanitized detections across a strictly ordered frame stream."""

    def __init__(
        self,
        config: TrackingConfig | None = None,
        *,
        optical_flow: OpticalFlowEstimator | None = None,
        template_matcher: TemplateMatcher | None = None,
        scene_change_detector: SceneChangeDetector | None = None,
    ) -> None:
        self.config = config or TrackingConfig()
        self.optical_flow = (
            OpticalFlowEstimator(self.config) if optical_flow is None else optical_flow
        )
        self.template_matcher = (
            TemplateMatcher(self.config)
            if template_matcher is None
            else template_matcher
        )
        self.scene_change_detector = (
            SceneChangeDetector(self.config)
            if scene_change_detector is None
            else scene_change_detector
        )
        self._tracks: list[_MutableTrack] = []
        self._next_track_id = 1
        self._previous_gray: NDArray[np.uint8] | None = None
        self._previous_scene_gray: NDArray[np.uint8] | None = None
        self._frame_index: int | None = None
        self._timestamp_ms: int | None = None
        self._scene_changed = False
        self._flow_executor: ThreadPoolExecutor | None = None

    @property
    def tracks(self) -> tuple[SensitiveTrack, ...]:
        return tuple(track.snapshot() for track in self._tracks)

    @property
    def has_active_tracks(self) -> bool:
        return any(track.active for track in self._tracks)

    @property
    def has_sticky_tracks(self) -> bool:
        return any(track.state is TrackState.STICKY for track in self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._previous_gray = None
        self._previous_scene_gray = None
        self._frame_index = None
        self._timestamp_ms = None
        self._scene_changed = False

    def close(self) -> None:
        executor = self._flow_executor
        self._flow_executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def update(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
        detections: list[SensitiveDetection] | None = None,
        include_track_snapshots: bool = True,
    ) -> TrackingFrameResult:
        _validate_frame(frame)
        self._validate_order(frame_index, timestamp_ms)
        self._validate_detections(detections, frame_index, timestamp_ms)
        had_active_tracks = self.has_active_tracks
        current_scene_gray = self.scene_change_detector.prepare(frame)

        scene_changed = False
        ocr_required = False
        if self._previous_scene_gray is not None:
            scene_changed = self.scene_change_detector.compare_gray(
                self._previous_scene_gray,
                current_scene_gray,
            ).changed
        needs_full_gray = had_active_tracks or bool(detections) or scene_changed
        current_gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if needs_full_gray else None
        )
        if had_active_tracks:
            if self._previous_gray is None or current_gray is None:
                raise RuntimeError("Active tracking frame cache is unavailable")
            ocr_required = self._advance_tracks(
                current_gray,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                scene_changed=scene_changed,
            )
        if detections is not None:
            self._apply_detections(
                detections,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )

        self._previous_gray = current_gray
        self._previous_scene_gray = current_scene_gray
        self._frame_index = frame_index
        self._timestamp_ms = timestamp_ms
        self._scene_changed = scene_changed
        if scene_changed:
            ocr_required = True
        if detections is not None and not self.has_sticky_tracks:
            ocr_required = False
        return self._result(
            ocr_required=ocr_required,
            ocr_performed=detections is not None,
            include_track_snapshots=include_track_snapshots,
        )

    def reacquire(
        self,
        detections: list[SensitiveDetection],
        *,
        include_track_snapshots: bool = True,
    ) -> TrackingFrameResult:
        """Apply an immediate OCR/detection result to the current frame."""

        if (
            self._frame_index is None
            or self._timestamp_ms is None
            or self._previous_gray is None
        ):
            raise RuntimeError("A frame must be updated before OCR reacquisition")
        self._validate_detections(
            detections,
            self._frame_index,
            self._timestamp_ms,
        )
        self._apply_detections(
            detections,
            frame_index=self._frame_index,
            timestamp_ms=self._timestamp_ms,
            frame_width=self._previous_gray.shape[1],
            frame_height=self._previous_gray.shape[0],
        )
        return self._result(
            ocr_required=self.has_sticky_tracks,
            ocr_performed=True,
            include_track_snapshots=include_track_snapshots,
        )

    def _validate_order(self, frame_index: int, timestamp_ms: int) -> None:
        if frame_index < 0 or timestamp_ms < 0:
            raise ValueError("Frame index and timestamp must be non-negative")
        if self._frame_index is not None and frame_index <= self._frame_index:
            raise ValueError("Tracking frames must have strictly increasing indices")
        if self._timestamp_ms is not None and timestamp_ms < self._timestamp_ms:
            raise ValueError("Tracking timestamps must be monotonic")

    @staticmethod
    def _validate_detections(
        detections: list[SensitiveDetection] | None,
        frame_index: int,
        timestamp_ms: int,
    ) -> None:
        if detections is None:
            return
        if any(
            item.frame_index != frame_index or item.timestamp_ms != timestamp_ms
            for item in detections
        ):
            raise ValueError("Detections must belong to the frame being tracked")

    def _advance_tracks(
        self,
        current_gray: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
        scene_changed: bool,
    ) -> bool:
        assert self._previous_gray is not None
        frame_height, frame_width = current_gray.shape
        ocr_required = False
        active_tracks = [track for track in self._tracks if track.active]
        predictions = {
            track.id: _move_box(
                track.current_bbox,
                track.velocity_x,
                track.velocity_y,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for track in active_tracks
        }
        flow_estimates = (
            {}
            if scene_changed
            else self._estimate_active_motion(
                current_gray,
                [
                    (track.id, track.current_bbox)
                    for track in active_tracks
                    if predictions[track.id] is not None
                ],
            )
        )
        for track in active_tracks:
            previous_bbox = track.current_bbox
            predicted = predictions[track.id]
            if predicted is None:
                track.state = TrackState.ENDED
                track.end_frame = track.current_frame
                ocr_required = True
                continue

            if not scene_changed:
                estimate = flow_estimates[track.id]
                if (
                    estimate is not None
                    and estimate.confidence >= self.config.minimum_flow_confidence
                ):
                    moved = _move_box(
                        previous_bbox,
                        estimate.dx,
                        estimate.dy,
                        frame_width=frame_width,
                        frame_height=frame_height,
                    )
                    if moved is not None:
                        self._record_motion(
                            track,
                            moved,
                            frame_index=frame_index,
                            confidence=estimate.confidence,
                        )
                        continue

                template = self.template_matcher.match_gray(
                    self._previous_gray,
                    current_gray,
                    previous_bbox,
                    predicted,
                )
                if template is not None:
                    self._record_motion(
                        track,
                        template.bbox,
                        frame_index=frame_index,
                        confidence=template.confidence,
                    )
                    continue

            if track.sticky_since_ms is None:
                track.sticky_since_ms = timestamp_ms
            elapsed = timestamp_ms - track.sticky_since_ms
            if elapsed <= self.config.sticky_duration_ms:
                track.regions[frame_index] = predicted
                track.state = TrackState.STICKY
                track.confidence = max(0.0, track.confidence * 0.85)
                ocr_required = True
            else:
                track.state = TrackState.LOST
                track.end_frame = track.current_frame
        return ocr_required

    def _estimate_active_motion(
        self,
        current_gray: NDArray[np.uint8],
        candidates: list[tuple[str, BoundingBox]],
    ) -> dict[str, MotionEstimate | None]:
        assert self._previous_gray is not None
        if (
            len(candidates) <= 1
            or self.config.optical_flow_workers == 1
            or current_gray.size < self.config.parallel_flow_min_frame_pixels
        ):
            return {
                track_id: self.optical_flow.estimate_gray(
                    self._previous_gray,
                    current_gray,
                    bbox,
                )
                for track_id, bbox in candidates
            }
        if self._flow_executor is None:
            self._flow_executor = ThreadPoolExecutor(
                max_workers=self.config.optical_flow_workers,
                thread_name_prefix="redact-flow",
            )
        pending = [
            (
                track_id,
                self._flow_executor.submit(
                    self.optical_flow.estimate_gray,
                    self._previous_gray,
                    current_gray,
                    bbox,
                ),
            )
            for track_id, bbox in candidates
        ]
        return {track_id: future.result() for track_id, future in pending}

    def _record_motion(
        self,
        track: _MutableTrack,
        bbox: BoundingBox,
        *,
        frame_index: int,
        confidence: float,
    ) -> None:
        previous = track.current_bbox
        observed_x = bbox.x - previous.x
        observed_y = bbox.y - previous.y
        alpha = self.config.velocity_smoothing
        track.velocity_x = alpha * observed_x + (1 - alpha) * track.velocity_x
        track.velocity_y = alpha * observed_y + (1 - alpha) * track.velocity_y
        track.regions[frame_index] = bbox
        track.confidence = min(
            1.0,
            0.65 * track.confidence + 0.35 * confidence,
        )
        track.state = TrackState.TRACKING
        track.sticky_since_ms = None

    def _apply_detections(
        self,
        detections: list[SensitiveDetection],
        *,
        frame_index: int,
        timestamp_ms: int,
        frame_width: int,
        frame_height: int,
    ) -> None:
        unmatched_track_ids = {track.id for track in self._tracks if track.active}
        for detection in sorted(
            detections,
            key=lambda item: (-item.confidence, item.bbox.y, item.bbox.x),
        ):
            clipped = detection.bbox.clipped(frame_width, frame_height)
            if clipped is None:
                continue
            match = self._best_detection_match(
                detection.type,
                clipped,
                unmatched_track_ids,
            )
            if match is None:
                track = _MutableTrack(
                    id=f"track-{self._next_track_id:06d}",
                    type=detection.type,
                    start_frame=frame_index,
                    confidence=detection.confidence,
                    regions={frame_index: clipped},
                )
                self._next_track_id += 1
                self._tracks.append(track)
                continue

            unmatched_track_ids.remove(match.id)
            previous = match.bbox_before(frame_index)
            if previous is not None:
                observed_x = clipped.x - previous.x
                observed_y = clipped.y - previous.y
                alpha = self.config.velocity_smoothing
                match.velocity_x = alpha * observed_x + (1 - alpha) * match.velocity_x
                match.velocity_y = alpha * observed_y + (1 - alpha) * match.velocity_y
            match.regions[frame_index] = clipped
            match.confidence = detection.confidence
            match.state = TrackState.TRACKING
            match.end_frame = None
            match.sticky_since_ms = None

    def _best_detection_match(
        self,
        sensitive_type: SensitiveType,
        bbox: BoundingBox,
        candidates: set[str],
    ) -> _MutableTrack | None:
        best: _MutableTrack | None = None
        best_score = 0.0
        for track in self._tracks:
            if track.id not in candidates or track.type is not sensitive_type:
                continue
            current = track.current_bbox
            overlap = intersection_over_union(current, bbox)
            distance = box_distance(current, bbox)
            scale = max(current.width, current.height, bbox.width, bbox.height)
            if (
                overlap < self.config.detection_match_iou
                and distance > scale * self.config.detection_match_distance_ratio
            ):
                continue
            score = overlap + 1.0 / (1.0 + distance / max(1, scale))
            if score > best_score:
                best = track
                best_score = score
        return best

    def _result(
        self,
        *,
        ocr_required: bool,
        ocr_performed: bool,
        include_track_snapshots: bool,
    ) -> TrackingFrameResult:
        assert self._frame_index is not None
        assert self._timestamp_ms is not None
        regions = tuple(
            TrackedRegion(
                track_id=track.id,
                type=track.type,
                bbox=track.regions[self._frame_index],
                confidence=track.confidence,
                state=track.state,
            )
            for track in self._tracks
            if track.active and self._frame_index in track.regions
        )
        return TrackingFrameResult(
            frame_index=self._frame_index,
            timestamp_ms=self._timestamp_ms,
            regions=regions,
            tracks=self.tracks if include_track_snapshots else (),
            scene_changed=self._scene_changed,
            ocr_required=ocr_required,
            ocr_performed=ocr_performed,
        )


class AdaptiveTrackingEngine:
    """Schedule local OCR periodically and immediately on tracking risk."""

    def __init__(
        self,
        ocr_engine: OcrEngine,
        *,
        detection_engine: DetectionProvider | None = None,
        detection_config: DetectionConfig | None = None,
        tracking_config: TrackingConfig | None = None,
        tracker: RegionTracker | None = None,
        fast_ocr_engine: OcrEngine | None = None,
        include_track_snapshots: bool = True,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.fast_ocr_engine = fast_ocr_engine
        self.detection_config = detection_config or DetectionConfig()
        self.tracking_config = tracking_config or TrackingConfig()
        self.detection_engine = (
            DetectionEngine(self.detection_config)
            if detection_engine is None
            else detection_engine
        )
        self.tracker = (
            RegionTracker(self.tracking_config) if tracker is None else tracker
        )
        self.include_track_snapshots = include_track_snapshots
        self._source_executor: ThreadPoolExecutor | None = None
        self._last_scan_ms: int | None = None
        self._last_authoritative_scan_ms: int | None = None
        self._last_authoritative_frame: NDArray[np.uint8] | None = None
        self._last_authoritative_detections: tuple[SensitiveDetection, ...] | None = (
            None
        )

    def reset(self) -> None:
        self.tracker.reset()
        self._last_scan_ms = None
        self._last_authoritative_scan_ms = None
        self._last_authoritative_frame = None
        self._last_authoritative_detections = None

    def close(self) -> None:
        source_executor = self._source_executor
        self._source_executor = None
        if source_executor is not None:
            source_executor.shutdown(wait=True, cancel_futures=True)
        self.tracker.close()

    def process_frame(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> TrackingFrameResult:
        _validate_frame(frame)
        had_sticky_tracks = self.tracker.has_sticky_tracks
        scan_rate = (
            self.detection_config.safe_scan_fps
            if had_sticky_tracks
            else self.detection_config.scan_fps
        )
        interval_ms = 1000.0 / scan_rate
        if self.tracker.has_active_tracks:
            interval_ms = min(
                interval_ms,
                float(self.tracking_config.reacquire_interval_ms),
            )
        scheduled = (
            self._last_scan_ms is None
            or timestamp_ms - self._last_scan_ms >= interval_ms
        )
        overlap_scheduled_scan = (
            scheduled
            and self.tracking_config.overlap_source_ocr_and_tracking
            and self.tracker.has_active_tracks
        )
        if overlap_scheduled_scan:
            if self._source_executor is None:
                self._source_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="redact-source-ocr",
                )
            scan_future = self._source_executor.submit(
                self._scan,
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            try:
                result = self.tracker.update(
                    frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    detections=None,
                    include_track_snapshots=self.include_track_snapshots,
                )
            except Exception:
                scan_future.cancel()
                raise
            detections = scan_future.result()
            result = self.tracker.reacquire(
                detections,
                include_track_snapshots=self.include_track_snapshots,
            )
        else:
            detections = (
                self._scan(
                    frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
                if scheduled
                else None
            )
            result = self.tracker.update(
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                detections=detections,
                include_track_snapshots=self.include_track_snapshots,
            )
        if scheduled:
            self._last_scan_ms = timestamp_ms
        tracking_risk = result.scene_changed or (
            result.ocr_required and not had_sticky_tracks
        )
        authoritative_reacquisition_needed = tracking_risk and (
            not result.ocr_performed
            or (
                self.fast_ocr_engine is not None
                and self._last_authoritative_scan_ms != timestamp_ms
            )
        )
        if authoritative_reacquisition_needed:
            if result.ocr_performed and self.fast_ocr_engine is not None:
                detections = self._scan_with_engine(
                    self.ocr_engine,
                    frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
                self._remember_authoritative_scan(
                    frame,
                    detections,
                    timestamp_ms=timestamp_ms,
                )
            else:
                detections = self._scan(
                    frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    force_authoritative=True,
                )
            self._last_scan_ms = timestamp_ms
            result = self.tracker.reacquire(
                detections,
                include_track_snapshots=self.include_track_snapshots,
            )
        return result

    def _scan(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
        force_authoritative: bool = False,
    ) -> list[SensitiveDetection]:
        if self.fast_ocr_engine is None:
            detections = self._scan_with_engine(
                self.ocr_engine,
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            self._remember_authoritative_scan(
                frame,
                detections,
                timestamp_ms=timestamp_ms,
            )
            return detections

        authoritative_due = (
            force_authoritative
            or self._last_authoritative_scan_ms is None
            or timestamp_ms < self._last_authoritative_scan_ms
            or timestamp_ms - self._last_authoritative_scan_ms
            >= self.tracking_config.authoritative_audit_interval_ms
        )
        if authoritative_due:
            reused_authoritative = (
                None
                if force_authoritative
                else self._reuse_authoritative_scan(
                    frame,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            )
            if reused_authoritative is not None:
                try:
                    accelerated = self._scan_with_engine(
                        self.fast_ocr_engine,
                        frame,
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                    )
                except (OcrProcessingError, OcrUnavailableError):
                    self.fast_ocr_engine = None
                    return reused_authoritative
                return self._merge_detections(
                    reused_authoritative,
                    accelerated,
                )

            fast_ocr_engine = self.fast_ocr_engine
            with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="redact-source-audit",
            ) as executor:
                accelerated_future = executor.submit(fast_ocr_engine.scan, frame)
                authoritative_tokens = self.ocr_engine.scan(frame)
                try:
                    accelerated_tokens = accelerated_future.result()
                except (OcrProcessingError, OcrUnavailableError):
                    self.fast_ocr_engine = None
                    accelerated_tokens = None

            authoritative = self._detect_tokens(
                frame,
                authoritative_tokens,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            self._remember_authoritative_scan(
                frame,
                authoritative,
                timestamp_ms=timestamp_ms,
            )
            if accelerated_tokens is None:
                return authoritative
            accelerated = self._detect_tokens(
                frame,
                accelerated_tokens,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            return self._merge_detections(authoritative, accelerated)

        try:
            accelerated = self._scan_with_engine(
                self.fast_ocr_engine,
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
        except (OcrProcessingError, OcrUnavailableError):
            self.fast_ocr_engine = None
            detections = self._scan_with_engine(
                self.ocr_engine,
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            self._remember_authoritative_scan(
                frame,
                detections,
                timestamp_ms=timestamp_ms,
            )
            return detections
        return accelerated

    def _reuse_authoritative_scan(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection] | None:
        reference = self._last_authoritative_frame
        cached = self._last_authoritative_detections
        if (
            not self.tracking_config.reuse_authoritative_frames
            or reference is None
            or cached is None
            or reference.shape != frame.shape
        ):
            return None

        difference = cv2.absdiff(reference, frame)
        delta = self.tracking_config.authoritative_reuse_pixel_delta
        within_delta = cv2.inRange(
            difference,
            (0, 0, 0),
            (delta, delta, delta),
        )
        total_pixels = frame.shape[0] * frame.shape[1]
        changed_pixels = total_pixels - cv2.countNonZero(within_delta)
        resolution_limit = int(
            total_pixels * self.tracking_config.authoritative_reuse_max_changed_fraction
        )
        changed_pixel_limit = min(
            self.tracking_config.authoritative_reuse_max_changed_pixels,
            resolution_limit,
        )
        if changed_pixels > changed_pixel_limit:
            return None

        reused = [
            detection.model_copy(
                update={
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                }
            )
            for detection in cached
        ]
        self._last_authoritative_scan_ms = timestamp_ms
        return reused

    def _remember_authoritative_scan(
        self,
        frame: NDArray[np.uint8],
        detections: list[SensitiveDetection],
        *,
        timestamp_ms: int,
    ) -> None:
        self._last_authoritative_scan_ms = timestamp_ms
        self._last_authoritative_frame = frame.copy()
        self._last_authoritative_detections = tuple(detections)

    def _scan_with_engine(
        self,
        ocr_engine: OcrEngine,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]:
        tokens = ocr_engine.scan(frame)
        return self._detect_tokens(
            frame,
            tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def _detect_tokens(
        self,
        frame: NDArray[np.uint8],
        tokens: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]:
        return self.detection_engine.detect(
            frame,
            tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _merge_detections(
        authoritative: list[SensitiveDetection],
        accelerated: list[SensitiveDetection],
    ) -> list[SensitiveDetection]:
        """Merge native findings while preferring overlapping Paddle geometry."""

        merged = list(authoritative)
        for candidate in accelerated:
            duplicate = any(
                existing.type is candidate.type
                and intersection_over_union(existing.bbox, candidate.bbox) >= 0.5
                for existing in merged
            )
            if not duplicate:
                merged.append(candidate)
        return merged
