"""Shared processing pipeline used by command and HTTP interfaces."""

from redact_video.pipeline.automated import (
    AutomatedProcessResult,
    process_video_automated,
)
from redact_video.pipeline.processor import ProcessResult, process_video

__all__ = [
    "AutomatedProcessResult",
    "ProcessResult",
    "process_video",
    "process_video_automated",
]
