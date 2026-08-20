from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from redact_video.exceptions import UnsupportedInputError
from redact_video.media import probe as probe_module


def test_probe_parses_rotation_audio_and_vfr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "sample.mov"
    source.write_bytes(b"media")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "duration": "2.0",
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "60/1",
                "nb_frames": "60",
                "pix_fmt": "yuv420p",
                "side_data_list": [{"rotation": -90}],
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "duration": "2.0",
            },
        ],
        "format": {"duration": "2.0"},
    }
    monkeypatch.setattr(probe_module.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        probe_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    info = probe_module.probe_media(source)

    assert info.display_width == 1080
    assert info.display_height == 1920
    assert info.has_audio is True
    assert info.audio_codec == "aac"
    assert info.is_variable_frame_rate is True


def test_probe_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "sample.avi"
    source.write_bytes(b"media")

    with pytest.raises(UnsupportedInputError, match="MP4 and MOV"):
        probe_module.probe_media(source)


def test_probe_rejects_missing_video_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "sample.mp4"
    source.write_bytes(b"media")
    payload = {"streams": [{"codec_type": "audio", "codec_name": "aac"}]}
    monkeypatch.setattr(probe_module.shutil, "which", lambda _: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        probe_module.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(UnsupportedInputError, match="no video stream"):
        probe_module.probe_media(source)
