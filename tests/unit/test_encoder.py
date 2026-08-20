from __future__ import annotations

from pathlib import Path

import pytest

from redact_video.exceptions import EncodeError
from redact_video.media.encoder import validate_encoded_media
from redact_video.media.probe import MediaInfo


def _media_info(
    *,
    video_duration_ms: int = 2000,
    audio_duration_ms: int | None = 2000,
) -> MediaInfo:
    return MediaInfo(
        path=Path("video.mp4"),
        width=320,
        height=240,
        duration_ms=max(video_duration_ms, audio_duration_ms or 0),
        video_duration_ms=video_duration_ms,
        audio_duration_ms=audio_duration_ms,
        frame_rate=10,
        nominal_frame_rate=10,
        frame_count=round(video_duration_ms / 100),
        video_codec="h264",
        pixel_format="yuv420p",
        has_audio=audio_duration_ms is not None,
        audio_codec="aac" if audio_duration_ms is not None else None,
    )


def test_encoded_media_rejects_video_duration_drift() -> None:
    source = _media_info()
    truncated = _media_info(video_duration_ms=1000, audio_duration_ms=1000)

    with pytest.raises(EncodeError, match="video duration"):
        validate_encoded_media(truncated, source, preserve_audio=True)


def test_encoded_media_rejects_audio_video_drift() -> None:
    source = _media_info()
    desynchronized = _media_info(video_duration_ms=2000, audio_duration_ms=1000)

    with pytest.raises(EncodeError, match="not synchronized"):
        validate_encoded_media(desynchronized, source, preserve_audio=True)
