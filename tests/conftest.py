from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from redact_video.models import BoundingBox, OcrToken


class MockCardFieldOcr:
    def __init__(self) -> None:
        self.calls = 0

    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        self.calls += 1
        return [
            OcrToken(
                text="Card Number",
                confidence=0.99,
                bbox=BoundingBox(x=35, y=30, width=120, height=22),
            )
        ]


@pytest.fixture
def card_field_ocr() -> MockCardFieldOcr:
    return MockCardFieldOcr()


@pytest.fixture(scope="session")
def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def media_factory(tmp_path: Path, ffmpeg_available: bool) -> Callable[..., Path]:
    if not ffmpeg_available:
        pytest.skip("FFmpeg and ffprobe are required for media integration tests")

    def create(
        *,
        suffix: str = ".mp4",
        with_audio: bool = True,
        duration: float = 1.0,
        audio_duration: float | None = None,
        audio_codec: str = "aac",
    ) -> Path:
        path = tmp_path / f"source{suffix}"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x240:rate=10:duration={duration}",
        ]
        if with_audio:
            selected_audio_duration = (
                duration if audio_duration is None else audio_duration
            )
            audio_source = (
                f"sine=frequency=440:sample_rate=48000:duration="
                f"{selected_audio_duration}"
            )
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    audio_source,
                ]
            )
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        if with_audio:
            command.extend(["-c:a", audio_codec])
        else:
            command.append("-an")
        command.append(str(path))
        subprocess.run(command, check=True, capture_output=True)
        return path

    return create
