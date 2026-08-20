from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from redact_video.media.decoder import (
    CompactFrameDecoder,
    FrameDecoder,
    SelectedFrameDecoder,
)
from redact_video.media.probe import probe_media
from redact_video.models import BoundingBox, JobStatus
from redact_video.pipeline.processor import process_video

pytestmark = pytest.mark.integration


def test_compact_decoder_preserves_frame_count_and_guard_geometry(
    media_factory: Callable[..., Path],
) -> None:
    source = media_factory(with_audio=False)

    with CompactFrameDecoder(source, 10, max_dimension=64) as decoder:
        frames = list(decoder)

    assert len(frames) == 10
    assert [frame.index for frame in frames] == list(range(10))
    assert all(frame.image.shape == (48, 64) for frame in frames)


def test_selected_decoder_returns_only_requested_absolute_frames(
    media_factory: Callable[..., Path],
) -> None:
    source = media_factory(with_audio=False)
    selected_indices = (1, 4, 8)

    with FrameDecoder(source, 10) as decoder:
        all_frames = list(decoder)
    with SelectedFrameDecoder(
        source,
        10,
        frame_indices=selected_indices,
    ) as decoder:
        selected = list(decoder)

    assert [frame.index for frame in selected] == list(selected_indices)
    assert [frame.timestamp_ms for frame in selected] == [100, 400, 800]
    assert all(
        np.array_equal(frame.image, all_frames[frame.index].image) for frame in selected
    )


def test_selected_decoder_splits_large_noncontiguous_selections(
    media_factory: Callable[..., Path],
) -> None:
    source = media_factory(with_audio=False, duration=10.0)
    selected_indices = tuple(range(1, 100, 2))

    with SelectedFrameDecoder(
        source,
        10,
        frame_indices=selected_indices,
    ) as decoder:
        selected = list(decoder)

    assert [frame.index for frame in selected] == list(selected_indices)
    assert [frame.timestamp_ms for frame in selected] == [
        frame_index * 100 for frame_index in selected_indices
    ]


def test_mp4_pipeline_preserves_resolution_audio_and_duration(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    source = media_factory(suffix=".mp4", with_audio=True)
    output = tmp_path / "redacted.mp4"

    result = process_video(
        source,
        output_path=output,
        manual_boxes=[BoundingBox(x=40, y=40, width=180, height=70)],
    )
    source_info = probe_media(source)
    output_info = probe_media(output)

    assert result.status == JobStatus.NEEDS_REVIEW
    assert result.frames_processed == 10
    assert result.audio_preserved is True
    assert output_info.video_codec == "h264"
    assert output_info.pixel_format == "yuv420p"
    assert output_info.has_audio is True
    assert (output_info.width, output_info.height) == (
        source_info.width,
        source_info.height,
    )
    assert abs(output_info.video_duration_ms - output_info.audio_duration_ms) < 100
    assert abs(output_info.duration_ms - source_info.duration_ms) < 100


def test_pipeline_pads_short_audio_without_truncating_video(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    source = media_factory(
        suffix=".mp4",
        with_audio=True,
        duration=2.0,
        audio_duration=1.0,
    )
    output = tmp_path / "short-audio.redacted.mp4"

    result = process_video(
        source,
        output_path=output,
        manual_boxes=[BoundingBox(x=40, y=40, width=180, height=70)],
    )
    source_info = probe_media(source)
    output_info = probe_media(output)

    assert source_info.video_duration_ms == 2000
    assert source_info.audio_duration_ms == 1000
    assert abs(output_info.video_duration_ms - source_info.video_duration_ms) < 100
    assert output_info.audio_duration_ms is not None
    assert abs(output_info.video_duration_ms - output_info.audio_duration_ms) < 100
    assert abs(result.output_duration_ms - source_info.video_duration_ms) < 100


def test_mov_with_pcm_audio_is_transcoded_to_mp4_audio(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    source = media_factory(suffix=".mov", with_audio=True, audio_codec="pcm_s16le")
    output = tmp_path / "from-mov.redacted.mp4"

    process_video(
        source,
        output_path=output,
        manual_boxes=[BoundingBox(x=20, y=20, width=100, height=50)],
    )
    output_info = probe_media(output)

    assert output_info.has_audio is True
    assert output_info.audio_codec == "aac"
    assert output_info.video_codec == "h264"


def test_pipeline_handles_video_without_audio(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "silent.redacted.mp4"

    result = process_video(
        source,
        output_path=output,
        manual_boxes=[BoundingBox(x=20, y=20, width=100, height=50)],
    )

    assert result.audio_preserved is False
    assert probe_media(output).has_audio is False


def test_pipeline_visibly_changes_the_manual_region(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "visual.redacted.mp4"
    box = BoundingBox(x=40, y=40, width=180, height=70)
    process_video(source, output_path=output, manual_boxes=[box])

    source_capture = cv2.VideoCapture(str(source))
    output_capture = cv2.VideoCapture(str(output))
    source_ok, source_frame = source_capture.read()
    output_ok, output_frame = output_capture.read()
    source_capture.release()
    output_capture.release()

    assert source_ok and output_ok
    source_roi = source_frame[box.y : box.bottom, box.x : box.right]
    output_roi = output_frame[box.y : box.bottom, box.x : box.right]
    assert output_roi.std() < source_roi.std()


def test_pipeline_normalizes_rotation_without_changing_display_geometry(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    base = media_factory(suffix=".mp4", with_audio=False)
    rotated = tmp_path / "rotated.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation",
            "90",
            "-i",
            str(base),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
        capture_output=True,
    )
    source_info = probe_media(rotated)
    if source_info.rotation == 0:
        pytest.skip("Local FFmpeg did not write rotation metadata")
    output = tmp_path / "rotation.redacted.mp4"

    process_video(
        rotated,
        output_path=output,
        manual_boxes=[BoundingBox(x=20, y=20, width=100, height=50)],
    )
    output_info = probe_media(output)

    assert (output_info.display_width, output_info.display_height) == (
        source_info.display_width,
        source_info.display_height,
    )
    assert output_info.rotation == 0
