"""Bounded local job execution over the verified core pipeline."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, Self

from redact_video.api.config import ApiConfig
from redact_video.api.models import (
    ApiError,
    RedactionMode,
    RedactionPolicy,
    RedactionReportResponse,
    RedactionResultReport,
    RedactionStatusResponse,
)
from redact_video.config import AppConfig, ProcessingMode
from redact_video.config_loader import apply_processing_mode, load_app_config
from redact_video.exceptions import (
    ConfigurationError,
    DecodeError,
    EncodeError,
    MediaProbeError,
    OcrProcessingError,
    OcrUnavailableError,
    RedactVideoError,
    UnsupportedInputError,
)
from redact_video.models import JobStatus
from redact_video.pipeline.automated import process_video_automated
from redact_video.reporting import SanitizedResultReport


class JobNotFoundError(KeyError):
    """The requested identifier does not belong to a live local job."""


class JobQueueFullError(RuntimeError):
    """The bounded worker and pending queue have no available slot."""


class JobWorker(Protocol):
    def __call__(self, request: WorkerRequest) -> WorkerOutcome: ...


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    job_id: str
    input_path: str
    output_path: str
    status_path: str
    mode: str
    config_path: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    status: str
    result: dict[str, Any] | None = None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedJob:
    job_id: str
    input_path: Path


@dataclass(slots=True)
class _JobRecord:
    job_id: str
    job_dir: Path
    input_path: Path
    output_path: Path
    status_path: Path
    policy: RedactionPolicy
    mode: RedactionMode
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    finished_at: datetime | None = None
    result: RedactionResultReport | None = None
    error: ApiError | None = None
    future: Future[WorkerOutcome] | None = None
    slot_reserved: bool = True


_FAILURES: dict[type[BaseException], tuple[str, str]] = {
    UnsupportedInputError: (
        "unsupported_input",
        "The uploaded video is unsupported or could not be inspected safely.",
    ),
    MediaProbeError: (
        "unsupported_input",
        "The uploaded video is unsupported or could not be inspected safely.",
    ),
    DecodeError: ("decode_failed", "The uploaded video could not be decoded safely."),
    EncodeError: ("encode_failed", "The redacted output could not be encoded."),
    ConfigurationError: (
        "invalid_configuration",
        "The redaction job configuration is invalid.",
    ),
    OcrUnavailableError: (
        "ocr_unavailable",
        "The local OCR runtime is unavailable.",
    ),
    OcrProcessingError: (
        "ocr_failed",
        "Local OCR could not safely process the uploaded video.",
    ),
    RedactVideoError: ("processing_failed", "The redaction job failed."),
}
_PUBLIC_ERROR_MESSAGES = {code: message for code, message in _FAILURES.values()} | {
    "internal_error": "The redaction job failed unexpectedly.",
    "worker_failed": "The local redaction worker stopped unexpectedly.",
}


def _config_for_mode(
    mode: RedactionMode, base_config: AppConfig | None = None
) -> AppConfig:
    config = base_config or AppConfig()
    return apply_processing_mode(config, ProcessingMode(mode.value))


def _write_progress(path: Path, status: JobStatus, progress: int) -> None:
    payload = json.dumps(
        {"status": status.value, "progress": progress},
        separators=(",", ":"),
    )
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _progress_for(message: str) -> tuple[JobStatus, int] | None:
    if message.startswith("Inspecting"):
        return JobStatus.INSPECTING, 5
    if message.startswith("Scanning"):
        return JobStatus.DETECTING, 20
    if message.startswith("Rendering"):
        return JobStatus.RENDERING, 55
    if message.startswith("Verification pass"):
        return JobStatus.VERIFYING, 75
    if message.startswith("Repairing"):
        return JobStatus.PATCHING, 85
    return None


def _failure_for(exc: BaseException) -> tuple[str, str]:
    for exception_type, failure in _FAILURES.items():
        if isinstance(exc, exception_type):
            return failure
    return "internal_error", "The redaction job failed unexpectedly."


def _public_error(code: str | None) -> ApiError:
    safe_code = code if code in _PUBLIC_ERROR_MESSAGES else "worker_failed"
    return ApiError(code=safe_code, message=_PUBLIC_ERROR_MESSAGES[safe_code])


def run_redaction_job(request: WorkerRequest) -> WorkerOutcome:
    """Process one job in a child process and return only sanitized metadata."""

    status_path = Path(request.status_path)

    def progress(message: str) -> None:
        state = _progress_for(message)
        if state is not None:
            _write_progress(status_path, *state)

    try:
        mode = RedactionMode(request.mode)
        base_config = (
            load_app_config(Path(request.config_path))
            if request.config_path is not None
            else AppConfig()
        )
        result = process_video_automated(
            Path(request.input_path),
            output_path=Path(request.output_path),
            config=_config_for_mode(mode, base_config),
            overwrite=False,
            progress=progress,
        )
        report = SanitizedResultReport.from_result(result).model_dump(mode="json")
        return WorkerOutcome(status=result.status.value, result=report)
    except Exception as exc:  # noqa: BLE001 -- sanitize the child-process boundary
        code, _ = _failure_for(exc)
        return WorkerOutcome(status=JobStatus.FAILED.value, error_code=code)


class LocalJobManager:
    """In-memory job registry backed by a bounded executor.

    The default executor is a local process pool. An executor can be injected for
    deterministic tests or replaced later by an external queue adapter.
    """

    def __init__(
        self,
        config: ApiConfig | None = None,
        *,
        executor: Executor | None = None,
        worker: JobWorker = run_redaction_job,
        start_cleanup_thread: bool = True,
    ) -> None:
        self.config = config or ApiConfig()
        if self.config.engine_config_path is not None:
            load_app_config(self.config.engine_config_path)
        self.config.job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.config.job_root.chmod(0o700)
        self._executor = executor or ProcessPoolExecutor(
            max_workers=self.config.max_workers
        )
        self._owns_executor = executor is None
        self._worker = worker
        self._slots = threading.BoundedSemaphore(
            self.config.max_workers + self.config.max_queued_jobs
        )
        self._records: dict[str, _JobRecord] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._stop_cleanup = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        if start_cleanup_thread:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="redact-video-job-cleanup",
                daemon=True,
            )
            self._cleanup_thread.start()

    def prepare_job(
        self,
        *,
        suffix: str,
        policy: RedactionPolicy,
        mode: RedactionMode,
    ) -> PreparedJob:
        if self._closed:
            raise RuntimeError("The local job manager is closed")
        if not self._slots.acquire(blocking=False):
            raise JobQueueFullError("The local redaction queue is full")
        normalized_suffix = suffix.lower()
        if normalized_suffix not in {".mp4", ".mov"}:
            self._slots.release()
            raise ValueError("Only MP4 and MOV uploads are supported")
        job_id = f"red_{uuid.uuid4().hex}"
        job_dir = self.config.job_root / job_id
        try:
            job_dir.mkdir(mode=0o700)
            record = _JobRecord(
                job_id=job_id,
                job_dir=job_dir,
                input_path=job_dir / f"input{normalized_suffix}",
                output_path=job_dir / "output.mp4",
                status_path=job_dir / "worker-status.json",
                policy=policy,
                mode=mode,
            )
            with self._lock:
                self._records[job_id] = record
            return PreparedJob(job_id=job_id, input_path=record.input_path)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            self._slots.release()
            raise

    def submit(self, job_id: str) -> None:
        with self._lock:
            record = self._record(job_id)
            if record.future is not None:
                raise RuntimeError("The job has already been submitted")
            request = WorkerRequest(
                job_id=record.job_id,
                input_path=str(record.input_path),
                output_path=str(record.output_path),
                status_path=str(record.status_path),
                mode=record.mode.value,
                config_path=(
                    str(self.config.engine_config_path)
                    if self.config.engine_config_path is not None
                    else None
                ),
            )
            try:
                future = self._executor.submit(self._worker, request)
            except Exception:
                self._release_slot(record)
                raise
            record.future = future
        future.add_done_callback(lambda completed: self._finish(job_id, completed))

    def discard(self, job_id: str) -> None:
        with self._lock:
            record = self._records.pop(job_id, None)
            if record is None:
                return
            if record.future is not None and not record.future.done():
                record.future.cancel()
            self._release_slot(record)
        shutil.rmtree(record.job_dir, ignore_errors=True)

    def status(self, job_id: str) -> RedactionStatusResponse:
        with self._lock:
            record = self._record(job_id)
            self._refresh_progress(record)
            return RedactionStatusResponse(
                job_id=record.job_id,
                status=record.status,
                progress=record.progress,
                created_at=record.created_at,
                finished_at=record.finished_at,
            )

    def report(self, job_id: str) -> RedactionReportResponse:
        with self._lock:
            record = self._record(job_id)
            return RedactionReportResponse(
                job_id=record.job_id,
                status=record.status,
                policy=record.policy,
                mode=record.mode,
                result=record.result,
                error=record.error,
            )

    def output_path(self, job_id: str) -> Path | None:
        with self._lock:
            record = self._record(job_id)
            if record.status not in {JobStatus.COMPLETED, JobStatus.NEEDS_REVIEW}:
                return None
            return record.output_path if record.output_path.is_file() else None

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        current = now or datetime.now(UTC)
        cutoff = current - timedelta(seconds=self.config.retention_seconds)
        expired: list[_JobRecord] = []
        with self._lock:
            for job_id, record in tuple(self._records.items()):
                if record.finished_at is not None and record.finished_at <= cutoff:
                    expired.append(self._records.pop(job_id))
        for record in expired:
            shutil.rmtree(record.job_dir, ignore_errors=True)
        return len(expired)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_cleanup.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2)
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            records = tuple(self._records.values())
            self._records.clear()
        for record in records:
            shutil.rmtree(record.job_dir, ignore_errors=True)

    def _record(self, job_id: str) -> _JobRecord:
        try:
            return self._records[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def _refresh_progress(self, record: _JobRecord) -> None:
        if record.finished_at is not None or not record.status_path.is_file():
            return
        try:
            payload = json.loads(record.status_path.read_text(encoding="utf-8"))
            status = JobStatus(payload["status"])
            progress = int(payload["progress"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return
        if (
            status
            in {
                JobStatus.INSPECTING,
                JobStatus.DETECTING,
                JobStatus.TRACKING,
                JobStatus.RENDERING,
                JobStatus.VERIFYING,
                JobStatus.PATCHING,
            }
            and 0 <= progress <= 99
        ):
            record.status = status
            record.progress = max(record.progress, progress)

    def _finish(self, job_id: str, future: Future[WorkerOutcome]) -> None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return
            try:
                outcome = future.result()
                status = JobStatus(outcome.status)
                if status not in {JobStatus.COMPLETED, JobStatus.NEEDS_REVIEW}:
                    raise ValueError("Worker returned a non-terminal success status")
                if outcome.result is None or not record.output_path.is_file():
                    raise ValueError("Worker returned no usable output")
                record.result = RedactionResultReport.model_validate(outcome.result)
                record.status = status
                record.progress = 100
            except Exception:  # noqa: BLE001 -- sanitize worker failures
                code = (
                    outcome.error_code
                    if "outcome" in locals() and outcome.error_code
                    else "worker_failed"
                )
                record.status = JobStatus.FAILED
                record.progress = 100
                record.error = _public_error(code)
                record.output_path.unlink(missing_ok=True)
            record.finished_at = datetime.now(UTC)
            record.input_path.unlink(missing_ok=True)
            record.status_path.unlink(missing_ok=True)
            self._release_slot(record)

    def _release_slot(self, record: _JobRecord) -> None:
        if record.slot_reserved:
            record.slot_reserved = False
            self._slots.release()

    def _cleanup_loop(self) -> None:
        while not self._stop_cleanup.wait(self.config.cleanup_interval_seconds):
            self.cleanup_expired()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
