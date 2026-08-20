"""FastAPI application factory and local server entry point."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from redact_video import __version__
from redact_video.api.config import ApiConfig
from redact_video.api.jobs import LocalJobManager
from redact_video.api.middleware import UploadBodyLimitMiddleware
from redact_video.api.routes import router


def create_app(
    config: ApiConfig | None = None,
    *,
    job_manager: LocalJobManager | None = None,
) -> FastAPI:
    api_config = config or ApiConfig()
    owns_manager = job_manager is None

    @asynccontextmanager
    async def lifespan(running_app: FastAPI) -> AsyncIterator[None]:
        manager = job_manager or LocalJobManager(api_config)
        running_app.state.job_manager = manager
        try:
            yield
        finally:
            if owns_manager:
                manager.close()

    application = FastAPI(
        title="Video Redactor API",
        version=__version__,
        lifespan=lifespan,
    )
    if job_manager is not None:
        application.state.job_manager = job_manager
    application.add_middleware(
        UploadBodyLimitMiddleware,
        maximum_bytes=(
            api_config.max_upload_bytes + api_config.multipart_overhead_bytes
        ),
    )
    application.include_router(router)

    @application.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()


def run() -> None:
    """Run the local API server using environment-controlled bind settings."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - packaging/install guard
        raise RuntimeError("Install video-redactor with the 'api' extra") from exc
    uvicorn.run(
        "redact_video.api.app:app",
        host=os.getenv("REDACT_VIDEO_API_HOST", "127.0.0.1"),
        port=int(os.getenv("REDACT_VIDEO_API_PORT", "8000")),
        reload=False,
        access_log=False,
    )
