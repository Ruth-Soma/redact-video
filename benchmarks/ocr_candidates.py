"""Value-free OCR candidate benchmark over private video frames.

Recognized text is held only in memory for the existing sensitive-data detectors.
The persisted report contains aggregate timing and sanitized detection geometry
comparisons, never source paths, pixels, OCR strings, or detected values.
"""

from __future__ import annotations

import argparse
import math
import platform
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from redact_video.config import DetectionConfig, OcrConfig
from redact_video.detection.engine import DetectionEngine
from redact_video.exceptions import DecodeError, OcrUnavailableError
from redact_video.media.probe import probe_media
from redact_video.models import BoundingBox, SensitiveDetection, SensitiveType
from redact_video.ocr.apple_vision import AppleVisionOcrEngine
from redact_video.ocr.base import OcrEngine
from redact_video.ocr.paddle import PaddleOcrEngine
from redact_video.reporting import write_private_json_artifact

REFERENCE_CANDIDATE = "paddle_v5"
DEFAULT_CANDIDATES = (
    REFERENCE_CANDIDATE,
    "paddle_v6_tiny",
    "paddle_v6_tiny_onnx",
    "apple_vision_accurate",
    "apple_vision_fast",
)
MINIMUM_REFERENCE_COVERAGE = 0.65
MINIMUM_INTERSECTION_OVER_UNION = 0.20


class CandidateStatus(str, Enum):
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class TimingSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int = Field(gt=0)
    total_ms: float = Field(ge=0)
    mean_ms: float = Field(ge=0)
    median_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    minimum_ms: float = Field(ge=0)
    maximum_ms: float = Field(ge=0)


class DetectionEquivalence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_findings: int = Field(ge=0)
    matched_findings: int = Field(ge=0)
    missing_findings: int = Field(ge=0)
    additional_findings: int = Field(ge=0)
    exact_frames: int = Field(ge=0)
    compared_frames: int = Field(ge=0)
    finding_recall: float = Field(ge=0, le=1)
    preserved_all_findings: bool


class CandidateBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    status: CandidateStatus
    initialization_ms: float = Field(ge=0)
    frames_processed: int = Field(ge=0)
    tokens_observed: int = Field(ge=0)
    findings: dict[SensitiveType, int] = Field(default_factory=dict)
    cold_scan_ms: float | None = Field(default=None, ge=0)
    warm_scan_timing: TimingSummary | None = None
    all_scan_timing: TimingSummary | None = None
    projected_scan_seconds: float | None = Field(default=None, ge=0)
    speedup_vs_reference: float | None = Field(default=None, ge=0)
    equivalence: DetectionEquivalence | None = None
    eligible: bool = False


class OcrCandidateBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    created_at: datetime
    python_version: str
    operating_system: str
    architecture: str
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    sampled_frames: int = Field(gt=0)
    source_duration_ms: int | None = Field(default=None, gt=0)
    projected_scan_count: int | None = Field(default=None, gt=0)
    reference_candidate: str
    reference_findings: int = Field(ge=0)
    has_sensitive_coverage: bool
    candidates: tuple[CandidateBenchmark, ...]
    recommended_candidate: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkFrame:
    """Private in-memory frame selected for a candidate comparison."""

    frame_index: int
    timestamp_ms: int
    image: NDArray[np.uint8]


EngineFactory = Callable[[], OcrEngine]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    name: str
    factory: EngineFactory


@dataclass(frozen=True, slots=True)
class _CompletedRun:
    timings_ms: tuple[float, ...]
    detections: tuple[tuple[SensitiveDetection, ...], ...]
    token_count: int


def _timing_summary(values: Sequence[float]) -> TimingSummary:
    if not values:
        raise ValueError("At least one timing sample is required")
    ordered = sorted(values)
    sample_count = len(ordered)
    midpoint = sample_count // 2
    median = (
        ordered[midpoint]
        if sample_count % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    p95_index = max(0, math.ceil(sample_count * 0.95) - 1)
    total = sum(ordered)
    return TimingSummary(
        samples=sample_count,
        total_ms=round(total, 3),
        mean_ms=round(total / sample_count, 3),
        median_ms=round(median, 3),
        p95_ms=round(ordered[p95_index], 3),
        minimum_ms=round(ordered[0], 3),
        maximum_ms=round(ordered[-1], 3),
    )


def _sensitive_family(value: SensitiveType) -> str:
    if value in {SensitiveType.CARD_NUMBER, SensitiveType.SAVED_CARD_NUMBER}:
        return "number"
    if value in {SensitiveType.CARD_EXPIRY, SensitiveType.SAVED_CARD_EXPIRY}:
        return "expiry"
    return "cvv"


def _intersection_area(first: BoundingBox, second: BoundingBox) -> int:
    width = max(0, min(first.right, second.right) - max(first.x, second.x))
    height = max(0, min(first.bottom, second.bottom) - max(first.y, second.y))
    return width * height


def _geometry_score(reference: BoundingBox, candidate: BoundingBox) -> float:
    intersection = _intersection_area(reference, candidate)
    if intersection <= 0:
        return 0
    reference_area = reference.width * reference.height
    candidate_area = candidate.width * candidate.height
    union = reference_area + candidate_area - intersection
    reference_coverage = intersection / reference_area
    intersection_over_union = intersection / union
    if (
        reference_coverage < MINIMUM_REFERENCE_COVERAGE
        and intersection_over_union < MINIMUM_INTERSECTION_OVER_UNION
    ):
        return 0
    return max(reference_coverage, intersection_over_union)


def _equivalence(
    reference_frames: Sequence[Sequence[SensitiveDetection]],
    candidate_frames: Sequence[Sequence[SensitiveDetection]],
) -> DetectionEquivalence:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError("Candidate and reference frame counts must match")
    reference_findings = 0
    matched_findings = 0
    additional_findings = 0
    exact_frames = 0
    for reference, candidate in zip(reference_frames, candidate_frames, strict=True):
        reference_findings += len(reference)
        available = set(range(len(candidate)))
        frame_matches = 0
        for finding in reference:
            matches = [
                (index, _geometry_score(finding.bbox, candidate[index].bbox))
                for index in available
                if _sensitive_family(candidate[index].type)
                == _sensitive_family(finding.type)
            ]
            best = max(matches, key=lambda item: item[1], default=None)
            if best is None or best[1] <= 0:
                continue
            available.remove(best[0])
            frame_matches += 1
        matched_findings += frame_matches
        additional_findings += len(available)
        if frame_matches == len(reference) and not available:
            exact_frames += 1
    missing_findings = reference_findings - matched_findings
    finding_recall = (
        matched_findings / reference_findings if reference_findings else 1.0
    )
    return DetectionEquivalence(
        reference_findings=reference_findings,
        matched_findings=matched_findings,
        missing_findings=missing_findings,
        additional_findings=additional_findings,
        exact_frames=exact_frames,
        compared_frames=len(reference_frames),
        finding_recall=round(finding_recall, 6),
        preserved_all_findings=missing_findings == 0,
    )


def _finding_counts(
    frames: Sequence[Sequence[SensitiveDetection]],
) -> dict[SensitiveType, int]:
    counts = Counter(finding.type for frame in frames for finding in frame)
    return {sensitive_type: counts[sensitive_type] for sensitive_type in counts}


def _projected_seconds(
    timings_ms: Sequence[float],
    projected_scan_count: int | None,
) -> float | None:
    if projected_scan_count is None or not timings_ms:
        return None
    if len(timings_ms) == 1 or projected_scan_count == 1:
        milliseconds = timings_ms[0] * projected_scan_count
    else:
        warm_mean = sum(timings_ms[1:]) / len(timings_ms[1:])
        milliseconds = timings_ms[0] + warm_mean * (projected_scan_count - 1)
    return round(milliseconds / 1000, 3)


def _run_candidate(
    spec: CandidateSpec,
    frames: Sequence[BenchmarkFrame],
    detector: DetectionEngine,
) -> tuple[CandidateBenchmark, _CompletedRun | None]:
    initialized_at = time.perf_counter()
    try:
        engine = spec.factory()
    except Exception:  # noqa: BLE001 - candidate details must not enter the report
        return (
            CandidateBenchmark(
                name=spec.name,
                status=CandidateStatus.UNAVAILABLE,
                initialization_ms=round(
                    (time.perf_counter() - initialized_at) * 1000,
                    3,
                ),
                frames_processed=0,
                tokens_observed=0,
            ),
            None,
        )
    initialization_ms = round((time.perf_counter() - initialized_at) * 1000, 3)
    timings: list[float] = []
    detections_by_frame: list[tuple[SensitiveDetection, ...]] = []
    token_count = 0
    try:
        for frame in frames:
            started = time.perf_counter()
            tokens = engine.scan(frame.image)
            findings = detector.detect(
                frame.image,
                tokens,
                frame_index=frame.frame_index,
                timestamp_ms=frame.timestamp_ms,
            )
            timings.append((time.perf_counter() - started) * 1000)
            token_count += len(tokens)
            detections_by_frame.append(tuple(findings))
    except OcrUnavailableError:
        return (
            CandidateBenchmark(
                name=spec.name,
                status=CandidateStatus.UNAVAILABLE,
                initialization_ms=initialization_ms,
                frames_processed=0,
                tokens_observed=0,
            ),
            None,
        )
    except Exception:  # noqa: BLE001 - candidate details must not enter the report
        return (
            CandidateBenchmark(
                name=spec.name,
                status=CandidateStatus.FAILED,
                initialization_ms=initialization_ms,
                frames_processed=0,
                tokens_observed=0,
            ),
            None,
        )
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    completed = _CompletedRun(
        timings_ms=tuple(timings),
        detections=tuple(detections_by_frame),
        token_count=token_count,
    )
    warm_timings = timings[1:] if len(timings) > 1 else timings
    return (
        CandidateBenchmark(
            name=spec.name,
            status=CandidateStatus.COMPLETED,
            initialization_ms=initialization_ms,
            frames_processed=len(frames),
            tokens_observed=token_count,
            findings=_finding_counts(detections_by_frame),
            cold_scan_ms=round(timings[0], 3),
            warm_scan_timing=_timing_summary(warm_timings),
            all_scan_timing=_timing_summary(timings),
        ),
        completed,
    )


def run_ocr_candidate_benchmark(
    frames: Sequence[BenchmarkFrame],
    candidates: Sequence[CandidateSpec],
    *,
    report_path: Path,
    overwrite: bool = False,
    reference_candidate: str = REFERENCE_CANDIDATE,
    source_duration_ms: int | None = None,
    projected_scan_count: int | None = None,
    detection_config: DetectionConfig | None = None,
) -> OcrCandidateBenchmarkReport:
    """Compare OCR engines without allowing recognized values into the report."""

    if not frames:
        raise ValueError("At least one private benchmark frame is required")
    if not candidates:
        raise ValueError("At least one OCR candidate is required")
    if projected_scan_count is not None and projected_scan_count <= 0:
        raise ValueError("Projected scan count must be positive")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("OCR candidate names must be unique")
    if reference_candidate not in {candidate.name for candidate in candidates}:
        raise ValueError("Reference OCR candidate is missing")

    detector = DetectionEngine(detection_config)
    base_results: list[CandidateBenchmark] = []
    completed_runs: dict[str, _CompletedRun] = {}
    for candidate in candidates:
        result, completed = _run_candidate(candidate, frames, detector)
        base_results.append(result)
        if completed is not None:
            completed_runs[candidate.name] = completed

    reference_run = completed_runs.get(reference_candidate)
    reference_findings = (
        sum(len(findings) for findings in reference_run.detections)
        if reference_run is not None
        else 0
    )
    has_sensitive_coverage = reference_findings > 0
    reference_projection = (
        _projected_seconds(reference_run.timings_ms, projected_scan_count)
        if reference_run is not None
        else None
    )

    finalized: list[CandidateBenchmark] = []
    for base_result in base_results:
        completed = completed_runs.get(base_result.name)
        if completed is None or reference_run is None:
            finalized.append(base_result)
            continue
        equivalence = _equivalence(reference_run.detections, completed.detections)
        projection = _projected_seconds(
            completed.timings_ms,
            projected_scan_count,
        )
        reference_timing = reference_run.timings_ms[1:] or reference_run.timings_ms
        candidate_timing = completed.timings_ms[1:] or completed.timings_ms
        reference_mean = sum(reference_timing) / len(reference_timing)
        candidate_mean = sum(candidate_timing) / len(candidate_timing)
        speedup = reference_mean / candidate_mean if candidate_mean > 0 else None
        if (
            reference_projection is not None
            and projection is not None
            and projection > 0
        ):
            speedup = reference_projection / projection
        finalized.append(
            base_result.model_copy(
                update={
                    "projected_scan_seconds": projection,
                    "speedup_vs_reference": (
                        round(speedup, 3) if speedup is not None else None
                    ),
                    "equivalence": equivalence,
                    "eligible": (
                        has_sensitive_coverage and equivalence.preserved_all_findings
                    ),
                }
            )
        )

    eligible = [
        candidate
        for candidate in finalized
        if candidate.eligible and candidate.warm_scan_timing is not None
    ]
    recommended = min(
        eligible,
        key=lambda candidate: candidate.warm_scan_timing.mean_ms,  # type: ignore[union-attr]
        default=None,
    )
    first_image = frames[0].image
    report = OcrCandidateBenchmarkReport(
        created_at=datetime.now(UTC),
        python_version=platform.python_version(),
        operating_system=platform.system().lower(),
        architecture=platform.machine().lower(),
        frame_width=int(first_image.shape[1]),
        frame_height=int(first_image.shape[0]),
        sampled_frames=len(frames),
        source_duration_ms=source_duration_ms,
        projected_scan_count=projected_scan_count,
        reference_candidate=reference_candidate,
        reference_findings=reference_findings,
        has_sensitive_coverage=has_sensitive_coverage,
        candidates=tuple(finalized),
        recommended_candidate=recommended.name if recommended is not None else None,
    )
    write_private_json_artifact(report, report_path, overwrite=overwrite)
    return report


def _even_frame_indices(frame_count: int, sample_count: int) -> tuple[int, ...]:
    if frame_count <= 0 or sample_count <= 0:
        raise ValueError("Frame and sample counts must be positive")
    if frame_count <= sample_count:
        return tuple(range(frame_count))
    return tuple(
        sorted(
            {
                min(
                    frame_count - 1,
                    round((offset + 1) * (frame_count - 1) / (sample_count + 1)),
                )
                for offset in range(sample_count)
            }
        )
    )


def load_private_benchmark_frames(
    input_path: Path,
    *,
    frame_indices: Sequence[int] | None = None,
    sample_count: int = 7,
) -> tuple[tuple[BenchmarkFrame, ...], int]:
    """Decode selected frames without writing pixels or source metadata."""

    media = probe_media(input_path)
    estimated_frame_count = media.frame_count or max(
        1,
        round(media.video_duration_ms / 1000 * media.frame_rate),
    )
    selected = (
        tuple(sorted(set(frame_indices)))
        if frame_indices
        else _even_frame_indices(estimated_frame_count, sample_count)
    )
    if not selected or selected[0] < 0 or selected[-1] >= estimated_frame_count:
        raise ValueError("Selected benchmark frame is outside the video")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise DecodeError("OpenCV could not open the benchmark video")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    frames: list[BenchmarkFrame] = []
    try:
        for index in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            decoded, image = capture.read()
            if (
                not decoded
                or image is None
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 3
            ):
                raise DecodeError("A selected benchmark frame could not be decoded")
            frames.append(
                BenchmarkFrame(
                    frame_index=index,
                    timestamp_ms=round(index / media.frame_rate * 1000),
                    image=image,
                )
            )
    finally:
        capture.release()
    return tuple(frames), media.video_duration_ms


def _paddle_v5() -> OcrEngine:
    return PaddleOcrEngine(
        OcrConfig(
            ocr_version="PP-OCRv5",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            inference_engine=None,
        )
    )


def _paddle_v6_tiny(*, onnx: bool = False) -> OcrEngine:
    return PaddleOcrEngine(
        OcrConfig(
            ocr_version="PP-OCRv6",
            text_detection_model_name="PP-OCRv6_tiny_det",
            text_recognition_model_name="PP-OCRv6_tiny_rec",
            inference_engine="onnxruntime" if onnx else None,
        )
    )


def _apple_vision(level: str) -> OcrEngine:
    config = OcrConfig(apple_vision_recognition_level=level)  # type: ignore[arg-type]
    engine = AppleVisionOcrEngine.try_create(config)
    if engine is None:
        raise OcrUnavailableError("Apple Vision OCR is unavailable")
    return engine


def candidate_specs(names: Sequence[str]) -> tuple[CandidateSpec, ...]:
    factories: dict[str, EngineFactory] = {
        REFERENCE_CANDIDATE: _paddle_v5,
        "paddle_v6_tiny": _paddle_v6_tiny,
        "paddle_v6_tiny_onnx": lambda: _paddle_v6_tiny(onnx=True),
        "apple_vision_accurate": lambda: _apple_vision("accurate"),
        "apple_vision_fast": lambda: _apple_vision("fast"),
    }
    unknown = set(names) - factories.keys()
    if unknown:
        raise ValueError("Unknown OCR candidate")
    ordered_names = tuple(dict.fromkeys((REFERENCE_CANDIDATE, *names)))
    return tuple(CandidateSpec(name, factories[name]) for name in ordered_names)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare OCR candidates on private frames with a value-free report."
    )
    parser.add_argument("input", type=Path, help="Private MP4 or MOV input.")
    parser.add_argument(
        "--candidate",
        action="append",
        choices=DEFAULT_CANDIDATES,
        help="Candidate to run; repeat for more than one. The v5 reference is automatic.",
    )
    parser.add_argument(
        "--frame-index",
        action="append",
        type=int,
        help="Exact private frame to sample; repeat for more than one.",
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--projected-scans", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ocr-candidate-benchmark.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    frames, duration_ms = load_private_benchmark_frames(
        args.input,
        frame_indices=args.frame_index,
        sample_count=args.samples,
    )
    report = run_ocr_candidate_benchmark(
        frames,
        candidate_specs(args.candidate or DEFAULT_CANDIDATES),
        report_path=args.output,
        overwrite=args.overwrite,
        source_duration_ms=duration_ms,
        projected_scan_count=args.projected_scans,
    )
    for candidate in report.candidates:
        if candidate.status is not CandidateStatus.COMPLETED:
            print(f"{candidate.name}: {candidate.status.value}")
            continue
        assert candidate.warm_scan_timing is not None
        assert candidate.equivalence is not None
        print(
            f"{candidate.name}: {candidate.warm_scan_timing.mean_ms:.1f} ms warm, "
            f"{candidate.equivalence.matched_findings}/"
            f"{candidate.equivalence.reference_findings} reference findings"
        )
    if not report.has_sensitive_coverage:
        print("No reference findings were present; select known-sensitive frames.")
    elif report.recommended_candidate is not None:
        print(f"Fastest privacy-eligible candidate: {report.recommended_candidate}")
    print("Sanitized benchmark report written.")
    reference = next(
        candidate
        for candidate in report.candidates
        if candidate.name == report.reference_candidate
    )
    return 0 if reference.status is CandidateStatus.COMPLETED else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
