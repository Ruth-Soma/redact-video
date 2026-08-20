"""Rendered-output safety scanner using local OCR and card detectors."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Condition, Event, Lock, Thread
from typing import Protocol, Self

import cv2
import numpy as np
from numpy.typing import NDArray

from redact_video.config import DetectionConfig, VerificationConfig
from redact_video.detection import DetectionEngine
from redact_video.detection.geometry import intersection_over_union
from redact_video.exceptions import DecodeError, OcrProcessingError, OcrUnavailableError
from redact_video.media.decoder import (
    CompactFrameDecoder,
    FrameDecoder,
    SelectedFrameDecoder,
    VideoFrame,
)
from redact_video.models import OcrToken, SensitiveDetection
from redact_video.ocr.base import OcrEngine
from redact_video.verification.models import (
    VerificationLeak,
    VerificationPassResult,
)


class VerificationDetectionProvider(Protocol):
    def detect(
        self,
        frame: NDArray[np.uint8],
        ocr: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> list[SensitiveDetection]: ...


@dataclass(frozen=True, slots=True)
class _PrefetchedFastScan:
    engine: OcrEngine
    future: Future[list[OcrToken]]


@dataclass(frozen=True, slots=True)
class _PrefetchedAuthoritativeScan:
    frame_index: int
    future: Future[list[OcrToken]]


@dataclass(frozen=True, slots=True)
class _CleanFrameEvidence:
    timestamp_ms: int
    guard: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _DecoderFailure:
    error: Exception


_FRAME_QUEUE_END = object()


class _PrefetchingVerificationDecoder:
    """Bounded decoder that overlaps future audits with ordered consumption."""

    def __init__(
        self,
        video_path: Path,
        *,
        frame_rate: float,
        ocr_engine: OcrEngine,
        max_buffered_frames: int,
        max_buffered_bytes: int,
    ) -> None:
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.ocr_engine = ocr_engine
        self._frames: Queue[VideoFrame | _DecoderFailure | object] = Queue(
            maxsize=max_buffered_frames
        )
        self._state_lock = Lock()
        self._ocr_lock = Lock()
        self._buffer_condition = Condition()
        self._buffered_bytes = 0
        self._max_buffered_bytes = max_buffered_bytes
        self._stop = Event()
        self._audit: _PrefetchedAuthoritativeScan | None = None
        self._closed = False
        self._audit_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="redact-authoritative-audit",
        )
        self._thread = Thread(
            target=self._run,
            name="redact-verification-decoder",
            daemon=True,
        )

    def __enter__(self) -> Self:
        self.schedule_audit(0)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    def __iter__(self):  # type: ignore[no-untyped-def]
        while True:
            item = self._frames.get()
            if item is _FRAME_QUEUE_END:
                return
            if isinstance(item, _DecoderFailure):
                raise item.error
            self._release_buffered_frame(item)
            yield item

    def schedule_audit(self, frame_index: int) -> _PrefetchedAuthoritativeScan:
        future: Future[list[OcrToken]] = Future()
        scan = _PrefetchedAuthoritativeScan(
            frame_index=frame_index,
            future=future,
        )
        with self._state_lock:
            if self._audit is not None:
                self._audit.future.cancel()
            self._audit = scan
        return scan

    def audit_for(self, frame_index: int) -> _PrefetchedAuthoritativeScan | None:
        with self._state_lock:
            if self._audit is not None and self._audit.frame_index == frame_index:
                return self._audit
        return None

    def cancel_audit(self) -> None:
        with self._state_lock:
            if self._audit is not None:
                self._audit.future.cancel()
                self._audit = None

    def scan(self, frame: NDArray[np.uint8]) -> list[OcrToken]:
        """Serialize immediate scans with any audit already in flight."""

        with self._ocr_lock:
            return self.ocr_engine.scan(frame)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        with self._buffer_condition:
            self._buffer_condition.notify_all()
        while self._thread.is_alive():
            try:
                item = self._frames.get(timeout=0.05)
            except Empty:
                pass
            else:
                self._release_buffered_frame(item)
        self._thread.join()
        self._audit_executor.shutdown(wait=True, cancel_futures=False)

    def _run(self) -> None:
        try:
            with FrameDecoder(self.video_path, self.frame_rate) as decoder:
                for frame in decoder:
                    with self._state_lock:
                        audit = (
                            self._audit
                            if (
                                self._audit is not None
                                and self._audit.frame_index == frame.index
                            )
                            else None
                        )
                    if audit is None or not audit.future.set_running_or_notify_cancel():
                        audit = None
                    if audit is not None:
                        self._audit_executor.submit(
                            self._complete_audit,
                            audit,
                            frame.image,
                        )
                    if not self._put_frame(frame):
                        return
        except Exception as exc:  # noqa: BLE001 - optimization falls back safely
            with self._state_lock:
                if self._audit is not None and not self._audit.future.done():
                    self._audit.future.set_exception(exc)
            self._put_frame(_DecoderFailure(exc))
        finally:
            self._put_frame(_FRAME_QUEUE_END)

    def _complete_audit(
        self,
        audit: _PrefetchedAuthoritativeScan,
        frame: NDArray[np.uint8],
    ) -> None:
        try:
            tokens = self.scan(frame)
        except Exception as exc:  # noqa: BLE001 - deliver to ordered consumer
            audit.future.set_exception(exc)
        else:
            audit.future.set_result(tokens)

    def _put_frame(self, item: VideoFrame | _DecoderFailure | object) -> bool:
        reserved_bytes = 0
        if isinstance(item, VideoFrame):
            reserved_bytes = item.image.nbytes
            with self._buffer_condition:
                while (
                    not self._stop.is_set()
                    and self._buffered_bytes > 0
                    and self._buffered_bytes + reserved_bytes > self._max_buffered_bytes
                ):
                    self._buffer_condition.wait(timeout=0.05)
                if self._stop.is_set():
                    return False
                self._buffered_bytes += reserved_bytes
        while not self._stop.is_set():
            try:
                self._frames.put(item, timeout=0.05)
            except Full:
                continue
            return True
        if reserved_bytes:
            with self._buffer_condition:
                self._buffered_bytes -= reserved_bytes
                self._buffer_condition.notify_all()
        return False

    def _release_buffered_frame(
        self,
        item: VideoFrame | _DecoderFailure | object,
    ) -> None:
        if not isinstance(item, VideoFrame):
            return
        with self._buffer_condition:
            self._buffered_bytes -= item.image.nbytes
            self._buffer_condition.notify_all()


def sensitive_detection_config(
    base: DetectionConfig,
    verification: VerificationConfig,
) -> DetectionConfig:
    """Return lower detector thresholds and per-frame repair scan rates."""

    thresholds = base.thresholds.model_copy(
        update={
            field: getattr(base.thresholds, field) * verification.threshold_scale
            for field in base.thresholds.__class__.model_fields
        }
    )
    return base.model_copy(
        update={
            "thresholds": thresholds,
            "scan_fps": 120.0,
            "safe_scan_fps": 120.0,
        }
    )


class SafetyVerifier:
    """Rescan every rendered frame and report value-bearing detections."""

    def __init__(
        self,
        ocr_engine: OcrEngine,
        *,
        detection_config: DetectionConfig | None = None,
        verification_config: VerificationConfig | None = None,
        detection_engine: VerificationDetectionProvider | None = None,
        fast_ocr_engine: OcrEngine | None = None,
        secondary_fast_ocr_engine: OcrEngine | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.fast_ocr_engine = fast_ocr_engine
        self.secondary_fast_ocr_engine = (
            None
            if secondary_fast_ocr_engine is fast_ocr_engine
            else secondary_fast_ocr_engine
        )
        self.verification_config = verification_config or VerificationConfig()
        self.detection_config = sensitive_detection_config(
            detection_config or DetectionConfig(),
            self.verification_config,
        )
        self.detection_engine = (
            DetectionEngine(self.detection_config)
            if detection_engine is None
            else detection_engine
        )
        self._clean_evidence_pass_number: int | None = None
        self._clean_frame_evidence: dict[int, _CleanFrameEvidence] = {}

    def inspect_frame(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> tuple[VerificationLeak, ...]:
        """Return readable-value evidence while ignoring label-only anchors."""

        return self._inspect_frame_with_engine(
            self.ocr_engine,
            frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def _inspect_frame_with_engine(
        self,
        ocr_engine: OcrEngine,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> tuple[VerificationLeak, ...]:
        tokens = ocr_engine.scan(frame)
        return self._inspect_tokens(
            frame,
            tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def _inspect_tokens(
        self,
        frame: NDArray[np.uint8],
        tokens: list[OcrToken],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> tuple[VerificationLeak, ...]:
        detections = self.detection_engine.detect(
            frame,
            tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        return tuple(
            VerificationLeak.from_detection(detection)
            for detection in detections
            if detection.source != "semantic_anchor"
        )

    def _inspect_authoritative_audit(
        self,
        frame: NDArray[np.uint8],
        *,
        frame_index: int,
        timestamp_ms: int,
    ) -> tuple[VerificationLeak, ...]:
        """Run independent OCR engines together, then inspect their tokens safely."""

        fast_ocr_engine = self.fast_ocr_engine
        if fast_ocr_engine is None:
            return self.inspect_frame(
                frame,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )

        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="redact-verification-audit",
        ) as executor:
            accelerated_future = executor.submit(fast_ocr_engine.scan, frame)
            authoritative_tokens = self.ocr_engine.scan(frame)
            try:
                accelerated_tokens = accelerated_future.result()
            except (OcrProcessingError, OcrUnavailableError):
                self._disable_fast_ocr()
                accelerated_tokens = None

        authoritative_leaks = self._inspect_tokens(
            frame,
            authoritative_tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        if accelerated_tokens is None:
            return authoritative_leaks
        accelerated_leaks = self._inspect_tokens(
            frame,
            accelerated_tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        return self._merge_leaks(authoritative_leaks, accelerated_leaks)

    @staticmethod
    def _merge_leaks(
        authoritative: tuple[VerificationLeak, ...],
        accelerated: tuple[VerificationLeak, ...],
    ) -> tuple[VerificationLeak, ...]:
        merged = list(authoritative)
        for candidate in accelerated:
            duplicate = any(
                existing.type is candidate.type
                and intersection_over_union(existing.bbox, candidate.bbox) >= 0.5
                for existing in merged
            )
            if not duplicate:
                merged.append(candidate)
        return tuple(merged)

    def _disable_fast_ocr(self) -> None:
        self.fast_ocr_engine = None
        self.secondary_fast_ocr_engine = None

    def _inspect_prefetched_frame(
        self,
        frame: NDArray[np.uint8],
        scan: _PrefetchedFastScan,
        *,
        authoritative_due: bool,
        authoritative_scan: _PrefetchedAuthoritativeScan | None = None,
        authoritative_ocr_engine: OcrEngine | None = None,
        frame_index: int,
        timestamp_ms: int,
    ) -> tuple[tuple[VerificationLeak, ...], bool]:
        """Inspect one ordered frame whose accelerated OCR may already be running."""

        active_authoritative_ocr = authoritative_ocr_engine or self.ocr_engine
        authoritative_tokens = None
        if authoritative_due:
            if authoritative_scan is not None:
                audit_future = authoritative_scan.future
                if not audit_future.done() and audit_future.cancel():
                    authoritative_tokens = None
                else:
                    try:
                        authoritative_tokens = audit_future.result()
                    except Exception:  # noqa: BLE001 - synchronized retry below
                        authoritative_tokens = None
            if authoritative_tokens is None:
                authoritative_tokens = active_authoritative_ocr.scan(frame)
        try:
            accelerated_tokens = scan.future.result()
        except (OcrProcessingError, OcrUnavailableError):
            self._disable_fast_ocr()
            if authoritative_tokens is None:
                authoritative_tokens = active_authoritative_ocr.scan(frame)

        if authoritative_tokens is not None:
            authoritative_leaks = self._inspect_tokens(
                frame,
                authoritative_tokens,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )
            if self.fast_ocr_engine is None:
                return authoritative_leaks, True
        else:
            authoritative_leaks = ()

        accelerated_leaks = self._inspect_tokens(
            frame,
            accelerated_tokens,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        if authoritative_tokens is not None:
            return self._merge_leaks(authoritative_leaks, accelerated_leaks), True
        if not accelerated_leaks:
            return accelerated_leaks, False

        authoritative_leaks = self._inspect_frame_with_engine(
            active_authoritative_ocr,
            frame,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )
        return self._merge_leaks(authoritative_leaks, accelerated_leaks), True

    def _discard_prefetch(self, scan: _PrefetchedFastScan | None) -> None:
        if scan is None or scan.future.cancel():
            return
        try:
            scan.future.result()
        except (OcrProcessingError, OcrUnavailableError):
            self._disable_fast_ocr()

    def _can_reuse_clean_scan(
        self,
        reference: NDArray[np.uint8],
        candidate: NDArray[np.uint8],
    ) -> bool:
        """Return whether a candidate differs only by bounded codec noise."""

        if reference.shape != candidate.shape:
            return False
        return self._can_reuse_guards(
            self._guard_image(reference),
            self._guard_image(candidate),
        )

    def _guard_image(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return a compact luminance image for inexpensive visual change checks."""

        height, width = frame.shape[:2]
        max_dimension = self.verification_config.reuse_guard_max_dimension
        scale = min(1.0, max_dimension / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _scene_guard_image(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Return a sparsely sampled guard for broad, immediate scene cuts."""

        height, width = frame.shape[:2]
        max_dimension = self.verification_config.reuse_guard_max_dimension
        scale = min(1.0, max_dimension / max(width, height))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _can_reuse_guards(
        self,
        reference: NDArray[np.uint8],
        candidate: NDArray[np.uint8],
    ) -> bool:
        if reference.shape != candidate.shape:
            return False
        changed_pixels = self._changed_pixel_count(reference, candidate)
        total_pixels = reference.size
        resolution_limit = int(
            total_pixels * self.verification_config.reuse_max_changed_fraction
        )
        changed_pixel_limit = min(
            self.verification_config.reuse_max_changed_pixels,
            resolution_limit,
        )
        return changed_pixels <= changed_pixel_limit

    def _prior_clean_evidence(
        self,
        pass_number: int,
    ) -> dict[int, _CleanFrameEvidence]:
        """Return sanitized evidence only for the immediately preceding pass."""

        if (
            not self.verification_config.reuse_clean_frames
            or not self.verification_config.reuse_clean_frames_across_passes
            or self._clean_evidence_pass_number != pass_number - 1
        ):
            return {}
        return self._clean_frame_evidence

    def _matching_prior_guard(
        self,
        frame: VideoFrame,
        prior_evidence: dict[int, _CleanFrameEvidence],
    ) -> NDArray[np.uint8] | None:
        """Return the current guard when prior clean evidence still matches."""

        if not prior_evidence:
            return None
        current_guard = self._guard_image(frame.image)
        if not self._guard_matches_prior_evidence(
            frame_index=frame.index,
            timestamp_ms=frame.timestamp_ms,
            current_guard=current_guard,
            prior_evidence=prior_evidence,
        ):
            return None
        return current_guard

    def _guard_matches_prior_evidence(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        current_guard: NDArray[np.uint8],
        prior_evidence: dict[int, _CleanFrameEvidence],
    ) -> bool:
        evidence = prior_evidence.get(frame_index)
        return (
            evidence is not None
            and evidence.timestamp_ms == timestamp_ms
            and self._can_reuse_guards(evidence.guard, current_guard)
        )

    def _prepare_selective_cross_pass_decode(
        self,
        video_path: Path,
        *,
        frame_rate: float,
        prior_evidence: dict[int, _CleanFrameEvidence],
        current_evidence: dict[int, _CleanFrameEvidence],
    ) -> tuple[tuple[int, ...], int]:
        """Guard every frame cheaply and return indices requiring full decode."""

        selected_indices: list[int] = []
        frames_scanned = 0
        with CompactFrameDecoder(
            video_path,
            frame_rate,
            max_dimension=self.verification_config.reuse_guard_max_dimension,
        ) as decoder:
            for frame in decoder:
                frames_scanned += 1
                current_guard = (
                    frame.image
                    if frame.image.ndim == 2
                    else self._guard_image(frame.image)
                )
                if self._guard_matches_prior_evidence(
                    frame_index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    current_guard=current_guard,
                    prior_evidence=prior_evidence,
                ):
                    self._record_clean_evidence(
                        current_evidence,
                        frame,
                        current_guard,
                    )
                else:
                    selected_indices.append(frame.index)
        if frames_scanned == 0:
            raise DecodeError("Video decoder returned no frames")
        return tuple(selected_indices), frames_scanned - len(selected_indices)

    @staticmethod
    def _record_clean_evidence(
        evidence: dict[int, _CleanFrameEvidence],
        frame: VideoFrame,
        guard: NDArray[np.uint8],
    ) -> None:
        evidence[frame.index] = _CleanFrameEvidence(
            timestamp_ms=frame.timestamp_ms,
            guard=guard,
        )

    def _commit_clean_evidence(
        self,
        pass_number: int,
        evidence: dict[int, _CleanFrameEvidence],
    ) -> None:
        self._clean_evidence_pass_number = pass_number
        self._clean_frame_evidence = evidence

    def _changed_pixel_count(
        self,
        reference: NDArray[np.uint8],
        candidate: NDArray[np.uint8],
    ) -> int:
        if reference.shape != candidate.shape:
            return max(reference.size, candidate.size)
        difference = cv2.absdiff(reference, candidate)
        delta = self.verification_config.reuse_pixel_delta
        return cv2.countNonZero(
            cv2.compare(difference, delta, cv2.CMP_GT),
        )

    def _is_immediate_scene_change(
        self,
        previous: NDArray[np.uint8] | None,
        current: NDArray[np.uint8],
    ) -> bool:
        if previous is None or previous.shape != current.shape:
            return previous is not None
        changed_fraction = self._changed_pixel_count(previous, current) / current.size
        return (
            changed_fraction >= self.verification_config.immediate_scene_change_fraction
        )

    def _minimum_scan_interval_elapsed(
        self,
        *,
        timestamp_ms: int,
        last_ocr_timestamp_ms: int | None,
    ) -> bool:
        return (
            last_ocr_timestamp_ms is None
            or timestamp_ms < last_ocr_timestamp_ms
            or timestamp_ms - last_ocr_timestamp_ms
            >= self.verification_config.minimum_scan_interval_ms
        )

    def verify(
        self,
        video_path: Path,
        *,
        frame_rate: float,
        pass_number: int,
    ) -> VerificationPassResult:
        """Decode and inspect every frame of one rendered candidate."""

        prior_evidence = self._prior_clean_evidence(pass_number)
        current_evidence: dict[int, _CleanFrameEvidence] = {}
        selected_frame_indices: tuple[int, ...] | None = None
        initial_frames_scanned = 0
        if prior_evidence and self.verification_config.selective_cross_pass_decode:
            try:
                (
                    selected_frame_indices,
                    initial_frames_scanned,
                ) = self._prepare_selective_cross_pass_decode(
                    video_path,
                    frame_rate=frame_rate,
                    prior_evidence=prior_evidence,
                    current_evidence=current_evidence,
                )
            except DecodeError:
                current_evidence.clear()
                selected_frame_indices = None
                initial_frames_scanned = 0
            if selected_frame_indices == ():
                result = VerificationPassResult(
                    pass_number=pass_number,
                    frames_scanned=initial_frames_scanned,
                )
                self._commit_clean_evidence(pass_number, current_evidence)
                return result

        if (
            self.fast_ocr_engine is not None
            and self.secondary_fast_ocr_engine is not None
        ):
            result = self._verify_with_fast_prefetch(
                video_path,
                frame_rate=frame_rate,
                pass_number=pass_number,
                prior_evidence=prior_evidence,
                current_evidence=current_evidence,
                selected_frame_indices=selected_frame_indices,
                initial_frames_scanned=initial_frames_scanned,
            )
            for leak in result.leaks:
                current_evidence.pop(leak.frame_index, None)
            self._commit_clean_evidence(pass_number, current_evidence)
            return result

        leaks: list[VerificationLeak] = []
        frames_scanned = initial_frames_scanned
        reference_guard: NDArray[np.uint8] | None = None
        previous_scene_guard: NDArray[np.uint8] | None = None
        last_ocr_frame_index: int | None = None
        last_ocr_timestamp_ms: int | None = None
        last_guard_check_timestamp_ms: int | None = None
        last_scan_was_clean = False
        last_authoritative_frame_index: int | None = None
        last_authoritative_timestamp_ms: int | None = None
        max_reuse_frames = max(
            1,
            int(frame_rate * self.verification_config.reuse_max_interval_ms / 1000),
        )
        max_audit_frames = max(
            1,
            int(
                frame_rate
                * self.verification_config.authoritative_audit_interval_ms
                / 1000
            ),
        )
        decoder_context = (
            SelectedFrameDecoder(
                video_path,
                frame_rate,
                frame_indices=selected_frame_indices,
            )
            if selected_frame_indices is not None
            else FrameDecoder(video_path, frame_rate)
        )
        with decoder_context as decoder:
            for frame in decoder:
                current_guard: NDArray[np.uint8] | None = None
                current_scene_guard = self._scene_guard_image(frame.image)
                prior_guard = self._matching_prior_guard(frame, prior_evidence)
                if prior_guard is not None:
                    self._record_clean_evidence(
                        current_evidence,
                        frame,
                        prior_guard,
                    )
                    frames_scanned += 1
                    previous_scene_guard = current_scene_guard
                    continue
                reuse_enabled = self.verification_config.reuse_clean_frames
                forced_by_frame_count = (
                    last_ocr_frame_index is None
                    or frame.index - last_ocr_frame_index >= max_reuse_frames
                )
                forced_by_timestamp = (
                    last_ocr_timestamp_ms is None
                    or frame.timestamp_ms < last_ocr_timestamp_ms
                    or frame.timestamp_ms - last_ocr_timestamp_ms
                    >= self.verification_config.reuse_max_interval_ms
                )
                authoritative_due = self.fast_ocr_engine is not None and (
                    last_authoritative_frame_index is None
                    or frame.index - last_authoritative_frame_index >= max_audit_frames
                    or last_authoritative_timestamp_ms is None
                    or frame.timestamp_ms < last_authoritative_timestamp_ms
                    or frame.timestamp_ms - last_authoritative_timestamp_ms
                    >= self.verification_config.authoritative_audit_interval_ms
                )
                minimum_scan_interval_elapsed = self._minimum_scan_interval_elapsed(
                    timestamp_ms=frame.timestamp_ms,
                    last_ocr_timestamp_ms=last_guard_check_timestamp_ms,
                )
                immediate_scene_change = self._is_immediate_scene_change(
                    previous_scene_guard,
                    current_scene_guard,
                )
                reuse_prerequisites_met = (
                    reuse_enabled
                    and last_scan_was_clean
                    and reference_guard is not None
                    and not forced_by_frame_count
                    and not forced_by_timestamp
                    and not authoritative_due
                    and not immediate_scene_change
                )
                material_reuse_allowed = True
                if reuse_prerequisites_met and minimum_scan_interval_elapsed:
                    current_guard = self._guard_image(frame.image)
                    last_guard_check_timestamp_ms = frame.timestamp_ms
                    material_reuse_allowed = self._can_reuse_guards(
                        reference_guard,
                        current_guard,
                    )
                can_reuse = reuse_prerequisites_met and material_reuse_allowed
                frame_leaks: tuple[VerificationLeak, ...] = ()
                if not can_reuse:
                    if self.fast_ocr_engine is None:
                        frame_leaks = self.inspect_frame(
                            frame.image,
                            frame_index=frame.index,
                            timestamp_ms=frame.timestamp_ms,
                        )
                        last_authoritative_frame_index = frame.index
                        last_authoritative_timestamp_ms = frame.timestamp_ms
                    elif authoritative_due:
                        frame_leaks = self._inspect_authoritative_audit(
                            frame.image,
                            frame_index=frame.index,
                            timestamp_ms=frame.timestamp_ms,
                        )
                        last_authoritative_frame_index = frame.index
                        last_authoritative_timestamp_ms = frame.timestamp_ms
                    else:
                        try:
                            accelerated_leaks = self._inspect_frame_with_engine(
                                self.fast_ocr_engine,
                                frame.image,
                                frame_index=frame.index,
                                timestamp_ms=frame.timestamp_ms,
                            )
                        except (OcrProcessingError, OcrUnavailableError):
                            self._disable_fast_ocr()
                            frame_leaks = self.inspect_frame(
                                frame.image,
                                frame_index=frame.index,
                                timestamp_ms=frame.timestamp_ms,
                            )
                            last_authoritative_frame_index = frame.index
                            last_authoritative_timestamp_ms = frame.timestamp_ms
                            leaks.extend(frame_leaks)
                            reference_guard = self._guard_image(frame.image)
                            last_guard_check_timestamp_ms = frame.timestamp_ms
                            last_ocr_frame_index = frame.index
                            last_ocr_timestamp_ms = frame.timestamp_ms
                            last_scan_was_clean = not frame_leaks
                            if not frame_leaks:
                                self._record_clean_evidence(
                                    current_evidence,
                                    frame,
                                    reference_guard,
                                )
                            frames_scanned += 1
                            previous_scene_guard = current_scene_guard
                            continue
                        if accelerated_leaks:
                            authoritative_leaks = self.inspect_frame(
                                frame.image,
                                frame_index=frame.index,
                                timestamp_ms=frame.timestamp_ms,
                            )
                            frame_leaks = self._merge_leaks(
                                authoritative_leaks,
                                accelerated_leaks,
                            )
                            last_authoritative_frame_index = frame.index
                            last_authoritative_timestamp_ms = frame.timestamp_ms
                        else:
                            frame_leaks = accelerated_leaks
                    leaks.extend(frame_leaks)
                    if current_guard is None:
                        current_guard = self._guard_image(frame.image)
                    reference_guard = current_guard
                    last_guard_check_timestamp_ms = frame.timestamp_ms
                    last_ocr_frame_index = frame.index
                    last_ocr_timestamp_ms = frame.timestamp_ms
                    last_scan_was_clean = not frame_leaks
                if not frame_leaks:
                    clean_guard = (
                        current_guard
                        if current_guard is not None
                        else self._guard_image(frame.image)
                    )
                    self._record_clean_evidence(
                        current_evidence,
                        frame,
                        clean_guard,
                    )
                frames_scanned += 1
                previous_scene_guard = current_scene_guard
        result = VerificationPassResult(
            pass_number=pass_number,
            frames_scanned=frames_scanned,
            leaks=tuple(
                sorted(
                    leaks,
                    key=lambda item: (
                        item.frame_index,
                        item.bbox.y,
                        item.bbox.x,
                        item.type.value,
                    ),
                )
            ),
        )
        for leak in result.leaks:
            current_evidence.pop(leak.frame_index, None)
        self._commit_clean_evidence(pass_number, current_evidence)
        return result

    def _verify_with_fast_prefetch(
        self,
        video_path: Path,
        *,
        frame_rate: float,
        pass_number: int,
        prior_evidence: dict[int, _CleanFrameEvidence],
        current_evidence: dict[int, _CleanFrameEvidence],
        selected_frame_indices: tuple[int, ...] | None,
        initial_frames_scanned: int,
    ) -> VerificationPassResult:
        """Verify in order while fast and authoritative engines scan ahead."""

        leaks: list[VerificationLeak] = []
        frames_scanned = initial_frames_scanned
        reference_guard: NDArray[np.uint8] | None = None
        previous_scene_guard: NDArray[np.uint8] | None = None
        last_ocr_frame_index: int | None = None
        last_ocr_timestamp_ms: int | None = None
        last_guard_check_timestamp_ms: int | None = None
        last_scan_was_clean = False
        last_authoritative_frame_index: int | None = None
        last_authoritative_timestamp_ms: int | None = None
        max_reuse_frames = max(
            1,
            int(frame_rate * self.verification_config.reuse_max_interval_ms / 1000),
        )
        max_audit_frames = max(
            1,
            int(
                frame_rate
                * self.verification_config.authoritative_audit_interval_ms
                / 1000
            ),
        )

        decoder_context = (
            SelectedFrameDecoder(
                video_path,
                frame_rate,
                frame_indices=selected_frame_indices,
            )
            if selected_frame_indices is not None
            else _PrefetchingVerificationDecoder(
                video_path,
                frame_rate=frame_rate,
                ocr_engine=self.ocr_engine,
                max_buffered_frames=(
                    self.verification_config.authoritative_prefetch_frames
                ),
                max_buffered_bytes=(
                    self.verification_config.authoritative_prefetch_max_bytes
                ),
            )
            if self.verification_config.prefetch_authoritative_audits
            else FrameDecoder(video_path, frame_rate)
        )
        with (
            decoder_context as decoder,
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="redact-verification-prefetch",
            ) as executor,
        ):
            authoritative_ocr_engine = (
                decoder
                if isinstance(decoder, _PrefetchingVerificationDecoder)
                else self.ocr_engine
            )
            frame_iterator = iter(decoder)
            current = next(frame_iterator, None)
            upcoming = next(frame_iterator, None)
            prefetched: _PrefetchedFastScan | None = None

            while current is not None:
                current_guard: NDArray[np.uint8] | None = None
                current_scene_guard = self._scene_guard_image(current.image)
                prior_guard = self._matching_prior_guard(current, prior_evidence)
                if prior_guard is not None:
                    self._discard_prefetch(prefetched)
                    if isinstance(decoder, _PrefetchingVerificationDecoder):
                        decoder.cancel_audit()
                    self._record_clean_evidence(
                        current_evidence,
                        current,
                        prior_guard,
                    )
                    frames_scanned += 1
                    previous_scene_guard = current_scene_guard
                    current = upcoming
                    upcoming = next(frame_iterator, None)
                    prefetched = None
                    continue
                reuse_enabled = self.verification_config.reuse_clean_frames
                forced_by_frame_count = (
                    last_ocr_frame_index is None
                    or current.index - last_ocr_frame_index >= max_reuse_frames
                )
                forced_by_timestamp = (
                    last_ocr_timestamp_ms is None
                    or current.timestamp_ms < last_ocr_timestamp_ms
                    or current.timestamp_ms - last_ocr_timestamp_ms
                    >= self.verification_config.reuse_max_interval_ms
                )
                authoritative_due = self.fast_ocr_engine is not None and (
                    last_authoritative_frame_index is None
                    or current.index - last_authoritative_frame_index
                    >= max_audit_frames
                    or last_authoritative_timestamp_ms is None
                    or current.timestamp_ms < last_authoritative_timestamp_ms
                    or current.timestamp_ms - last_authoritative_timestamp_ms
                    >= self.verification_config.authoritative_audit_interval_ms
                )
                minimum_scan_interval_elapsed = self._minimum_scan_interval_elapsed(
                    timestamp_ms=current.timestamp_ms,
                    last_ocr_timestamp_ms=last_guard_check_timestamp_ms,
                )
                immediate_scene_change = self._is_immediate_scene_change(
                    previous_scene_guard,
                    current_scene_guard,
                )
                reuse_prerequisites_met = (
                    reuse_enabled
                    and last_scan_was_clean
                    and reference_guard is not None
                    and not forced_by_frame_count
                    and not forced_by_timestamp
                    and not authoritative_due
                    and not immediate_scene_change
                )
                material_reuse_allowed = True
                if reuse_prerequisites_met and minimum_scan_interval_elapsed:
                    current_guard = self._guard_image(current.image)
                    last_guard_check_timestamp_ms = current.timestamp_ms
                    material_reuse_allowed = self._can_reuse_guards(
                        reference_guard,
                        current_guard,
                    )
                can_reuse = reuse_prerequisites_met and material_reuse_allowed
                frame_leaks: tuple[VerificationLeak, ...] = ()

                next_prefetch: _PrefetchedFastScan | None = None
                if can_reuse:
                    self._discard_prefetch(prefetched)
                elif self.fast_ocr_engine is None:
                    self._discard_prefetch(prefetched)
                    if isinstance(decoder, _PrefetchingVerificationDecoder):
                        decoder.cancel_audit()
                    frame_leaks = self._inspect_frame_with_engine(
                        authoritative_ocr_engine,
                        current.image,
                        frame_index=current.index,
                        timestamp_ms=current.timestamp_ms,
                    )
                    last_authoritative_frame_index = current.index
                    last_authoritative_timestamp_ms = current.timestamp_ms
                else:
                    if prefetched is None:
                        active_fast_engine = self.fast_ocr_engine
                        prefetched = _PrefetchedFastScan(
                            engine=active_fast_engine,
                            future=executor.submit(
                                active_fast_engine.scan,
                                current.image,
                            ),
                        )

                    if not authoritative_due and upcoming is not None:
                        upcoming_scene_guard = self._scene_guard_image(upcoming.image)
                        upcoming_forced_by_frame_count = (
                            upcoming.index - current.index >= max_reuse_frames
                        )
                        upcoming_forced_by_timestamp = (
                            upcoming.timestamp_ms < current.timestamp_ms
                            or upcoming.timestamp_ms - current.timestamp_ms
                            >= self.verification_config.reuse_max_interval_ms
                        )
                        upcoming_authoritative_due = (
                            last_authoritative_frame_index is None
                            or upcoming.index - last_authoritative_frame_index
                            >= max_audit_frames
                            or last_authoritative_timestamp_ms is None
                            or upcoming.timestamp_ms < last_authoritative_timestamp_ms
                            or upcoming.timestamp_ms - last_authoritative_timestamp_ms
                            >= self.verification_config.authoritative_audit_interval_ms
                        )
                        upcoming_minimum_scan_interval_elapsed = (
                            self._minimum_scan_interval_elapsed(
                                timestamp_ms=upcoming.timestamp_ms,
                                last_ocr_timestamp_ms=current.timestamp_ms,
                            )
                        )
                        upcoming_immediate_scene_change = (
                            self._is_immediate_scene_change(
                                current_scene_guard,
                                upcoming_scene_guard,
                            )
                        )
                        upcoming_can_reuse_if_clean = (
                            reuse_enabled
                            and not upcoming_forced_by_frame_count
                            and not upcoming_forced_by_timestamp
                            and not upcoming_authoritative_due
                            and not upcoming_immediate_scene_change
                            and (
                                not upcoming_minimum_scan_interval_elapsed
                                or self._can_reuse_guards(
                                    (
                                        current_guard
                                        if current_guard is not None
                                        else self._guard_image(current.image)
                                    ),
                                    self._guard_image(upcoming.image),
                                )
                            )
                        )
                        if not upcoming_can_reuse_if_clean:
                            other_fast_engine = (
                                self.secondary_fast_ocr_engine
                                if prefetched.engine is self.fast_ocr_engine
                                else self.fast_ocr_engine
                            )
                            if other_fast_engine is not None:
                                next_prefetch = _PrefetchedFastScan(
                                    engine=other_fast_engine,
                                    future=executor.submit(
                                        other_fast_engine.scan,
                                        upcoming.image,
                                    ),
                                )

                    authoritative_scan = (
                        decoder.audit_for(current.index)
                        if (
                            authoritative_due
                            and isinstance(
                                decoder,
                                _PrefetchingVerificationDecoder,
                            )
                        )
                        else None
                    )
                    frame_leaks, authoritative_ran = self._inspect_prefetched_frame(
                        current.image,
                        prefetched,
                        authoritative_due=authoritative_due,
                        authoritative_scan=authoritative_scan,
                        authoritative_ocr_engine=authoritative_ocr_engine,
                        frame_index=current.index,
                        timestamp_ms=current.timestamp_ms,
                    )
                    if authoritative_ran:
                        last_authoritative_frame_index = current.index
                        last_authoritative_timestamp_ms = current.timestamp_ms
                        if self.fast_ocr_engine is not None and isinstance(
                            decoder,
                            _PrefetchingVerificationDecoder,
                        ):
                            decoder.schedule_audit(current.index + max_audit_frames)
                    if self.fast_ocr_engine is None:
                        self._discard_prefetch(next_prefetch)
                        next_prefetch = None

                if not can_reuse:
                    leaks.extend(frame_leaks)
                    if current_guard is None:
                        current_guard = self._guard_image(current.image)
                    reference_guard = current_guard
                    last_guard_check_timestamp_ms = current.timestamp_ms
                    last_ocr_frame_index = current.index
                    last_ocr_timestamp_ms = current.timestamp_ms
                    last_scan_was_clean = not frame_leaks
                if not frame_leaks:
                    clean_guard = (
                        current_guard
                        if current_guard is not None
                        else self._guard_image(current.image)
                    )
                    self._record_clean_evidence(
                        current_evidence,
                        current,
                        clean_guard,
                    )
                frames_scanned += 1
                previous_scene_guard = current_scene_guard
                current = upcoming
                upcoming = next(frame_iterator, None)
                prefetched = next_prefetch

        return VerificationPassResult(
            pass_number=pass_number,
            frames_scanned=frames_scanned,
            leaks=tuple(
                sorted(
                    leaks,
                    key=lambda item: (
                        item.frame_index,
                        item.bbox.y,
                        item.bbox.x,
                        item.type.value,
                    ),
                )
            ),
        )
