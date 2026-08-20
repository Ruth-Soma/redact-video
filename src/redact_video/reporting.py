"""Value-free report construction and atomic JSON persistence."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from redact_video.exceptions import ConfigurationError, ReportWriteError
from redact_video.models import JobStatus, ProcessingTimings, SensitiveType
from redact_video.verification import VerificationSummary

if TYPE_CHECKING:
    from redact_video.pipeline.automated import AutomatedProcessResult


class SanitizedResultReport(BaseModel):
    """Core result metadata with no paths, pixels, timelines, or OCR text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_duration_ms: int = Field(gt=0)
    output_duration_ms: int = Field(gt=0)
    frames_processed: int = Field(gt=0)
    masked_frames: int = Field(ge=0)
    mask_regions: int = Field(ge=0)
    detections: dict[SensitiveType, int] = Field(default_factory=dict)
    verification: VerificationSummary
    audio_preserved: bool
    variable_frame_rate_normalized: bool = False
    warnings: tuple[str, ...] = ()
    timings: ProcessingTimings

    @classmethod
    def from_result(cls, result: AutomatedProcessResult) -> SanitizedResultReport:
        return cls(
            input_duration_ms=result.input_duration_ms,
            output_duration_ms=result.output_duration_ms,
            frames_processed=result.frames_processed,
            masked_frames=result.masked_frames,
            mask_regions=result.mask_regions,
            detections=result.tracks_by_type,
            verification=result.verification,
            audio_preserved=result.audio_preserved,
            variable_frame_rate_normalized=result.variable_frame_rate_normalized,
            warnings=result.warnings,
            timings=result.timings,
        )


class SanitizedProcessReport(SanitizedResultReport):
    status: JobStatus

    @classmethod
    def from_result(cls, result: AutomatedProcessResult) -> SanitizedProcessReport:
        details = SanitizedResultReport.from_result(result)
        return cls(status=result.status, **details.model_dump())


def _same_path(first: Path, second: Path) -> bool:
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return first.absolute() == second.absolute()


def validate_report_destination(
    report_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    destination = Path(report_path)
    if _same_path(destination, input_path) or _same_path(destination, output_path):
        raise ConfigurationError("Report path must differ from input and video output")
    if destination.suffix.lower() != ".json":
        raise ConfigurationError("Report output must use the .json extension")
    if destination.exists() and not overwrite:
        raise ConfigurationError("Report already exists; use --overwrite to replace it")
    if destination.exists() and not destination.is_file():
        raise ConfigurationError("Report path exists and is not a file")


def write_sanitized_report(
    report: SanitizedProcessReport,
    report_path: Path,
    *,
    overwrite: bool,
) -> None:
    write_private_json_artifact(report, report_path, overwrite=overwrite)


def write_private_json_artifact(
    document: BaseModel,
    destination_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Atomically persist an approved Pydantic model with private permissions."""

    destination = Path(destination_path)
    if destination.suffix.lower() != ".json":
        raise ConfigurationError("Report output must use the .json extension")
    if destination.exists() and not overwrite:
        raise ConfigurationError("Report already exists; use --overwrite to replace it")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.partial.json"
        )
        descriptor = os.open(
            staged,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(document.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if overwrite:
                os.replace(staged, destination)
            else:
                os.link(staged, destination)
                staged.unlink()
        except Exception:
            staged.unlink(missing_ok=True)
            raise
    except Exception:  # noqa: BLE001 -- replace filesystem details at public boundary
        raise ReportWriteError("Sanitized report could not be written") from None
