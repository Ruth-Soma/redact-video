"""FastAPI routes for asynchronous redaction jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from redact_video.api.jobs import (
    JobNotFoundError,
    JobQueueFullError,
    LocalJobManager,
)
from redact_video.api.models import (
    CreateRedactionResponse,
    RedactionMode,
    RedactionPolicy,
    RedactionReportResponse,
    RedactionStatusResponse,
)
from redact_video.models import JobStatus

router = APIRouter(prefix="/v1/redactions", tags=["redactions"])


def _manager(request: Request) -> LocalJobManager:
    return request.app.state.job_manager


async def _store_upload(
    upload: UploadFile,
    destination: Path,
    *,
    maximum_bytes: int,
    chunk_bytes: int,
) -> None:
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    received = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while chunk := await upload.read(chunk_bytes):
                received += len(chunk)
                if received > maximum_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail="The uploaded video exceeds the configured size limit.",
                    )
                handle.write(chunk)
        if received == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded video is empty.",
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@router.post(
    "",
    response_model=CreateRedactionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_redaction(
    request: Request,
    file: Annotated[UploadFile, File(description="MP4 or MOV screen recording")],
    policy: Annotated[RedactionPolicy, Form()] = RedactionPolicy.CARD_DETAILS,
    mode: Annotated[RedactionMode, Form()] = RedactionMode.STANDARD,
) -> CreateRedactionResponse:
    manager = _manager(request)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only MP4 and MOV uploads are supported.",
        )
    try:
        prepared = manager.prepare_job(suffix=suffix, policy=policy, mode=mode)
    except JobQueueFullError as exc:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local redaction queue is full; retry later.",
            headers={"Retry-After": "5"},
        ) from exc
    try:
        await _store_upload(
            file,
            prepared.input_path,
            maximum_bytes=manager.config.max_upload_bytes,
            chunk_bytes=manager.config.upload_chunk_bytes,
        )
        manager.submit(prepared.job_id)
    except HTTPException:
        manager.discard(prepared.job_id)
        raise
    except Exception as exc:
        manager.discard(prepared.job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local redaction worker is unavailable; retry later.",
            headers={"Retry-After": "5"},
        ) from exc
    return CreateRedactionResponse(job_id=prepared.job_id)


@router.get("/{job_id}", response_model=RedactionStatusResponse)
def redaction_status(request: Request, job_id: str) -> RedactionStatusResponse:
    try:
        return _manager(request).status(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Redaction job not found.") from exc


@router.get(
    "/{job_id}/output",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Redacted MP4 output",
            "content": {"video/mp4": {}},
        },
        409: {"description": "Output is not ready or the job failed"},
    },
)
def redaction_output(request: Request, job_id: str) -> FileResponse:
    manager = _manager(request)
    try:
        job_status = manager.status(job_id)
        output_path = manager.output_path(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Redaction job not found.") from exc
    if output_path is None:
        if job_status.status is JobStatus.FAILED:
            detail = "The redaction job failed and has no output."
        else:
            detail = "The redacted output is not ready."
        raise HTTPException(status_code=409, detail=detail)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{job_id}.redacted.mp4",
    )


@router.get("/{job_id}/report", response_model=RedactionReportResponse)
def redaction_report(request: Request, job_id: str) -> RedactionReportResponse:
    manager = _manager(request)
    try:
        job_status = manager.status(job_id)
        report = manager.report(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Redaction job not found.") from exc
    if job_status.finished_at is None:
        raise HTTPException(
            status_code=409, detail="The redaction report is not ready."
        )
    return report
