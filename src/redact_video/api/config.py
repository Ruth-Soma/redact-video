"""Validated settings for the optional local HTTP service."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ApiConfig(BaseModel):
    """Resource and retention limits enforced by the API adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_root: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "REDACT_VIDEO_API_JOB_ROOT",
                str(Path(tempfile.gettempdir()) / "video-redactor"),
            )
        )
    )
    engine_config_path: Path | None = Field(
        default_factory=lambda: (
            Path(value) if (value := os.getenv("REDACT_VIDEO_CONFIG")) else None
        )
    )
    max_upload_bytes: int = Field(
        default_factory=lambda: int(
            os.getenv("REDACT_VIDEO_API_MAX_UPLOAD_BYTES", str(2 * 1024**3))
        ),
        ge=1,
    )
    multipart_overhead_bytes: int = Field(default=64 * 1024, ge=0, le=16 * 1024**2)
    upload_chunk_bytes: int = Field(default=1024 * 1024, ge=4096, le=16 * 1024**2)
    max_workers: int = Field(
        default_factory=lambda: int(os.getenv("REDACT_VIDEO_API_WORKERS", "1")),
        ge=1,
        le=32,
    )
    max_queued_jobs: int = Field(
        default_factory=lambda: int(os.getenv("REDACT_VIDEO_API_QUEUE_SIZE", "4")),
        ge=0,
        le=1024,
    )
    retention_seconds: int = Field(
        default_factory=lambda: int(
            os.getenv("REDACT_VIDEO_API_RETENTION_SECONDS", "3600")
        ),
        ge=1,
        le=7 * 24 * 60 * 60,
    )
    cleanup_interval_seconds: int = Field(default=60, ge=1, le=3600)
