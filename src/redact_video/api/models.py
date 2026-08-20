"""Public, value-free API request and response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from redact_video.config import RedactionPolicy
from redact_video.models import JobStatus, ProcessingTimings, SensitiveType
from redact_video.verification import VerificationSummary


class RedactionMode(str, Enum):
    STANDARD = "standard"
    SAFE = "safe"
    FAST = "fast"


class CreateRedactionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    status: JobStatus = JobStatus.QUEUED


class RedactionStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    created_at: datetime
    finished_at: datetime | None = None


class ApiError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(pattern=r"^[a-z0-9_]+$", max_length=64)
    message: str = Field(max_length=160)


class RedactionResultReport(BaseModel):
    """Sanitized core result; paths and mask timelines are intentionally absent."""

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


class RedactionReportResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    status: JobStatus
    policy: RedactionPolicy
    mode: RedactionMode
    result: RedactionResultReport | None = None
    error: ApiError | None = None
