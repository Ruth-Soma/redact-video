from pathlib import Path

import pytest

from redact_video.exceptions import ConfigurationError
from redact_video.pipeline.processor import default_output_path, process_video


def test_default_output_path_is_always_mp4() -> None:
    assert default_output_path(Path("capture.mov")) == Path("capture.redacted.mp4")


def test_phase_one_requires_a_manual_box(tmp_path: Path) -> None:
    source = tmp_path / "capture.mp4"

    with pytest.raises(ConfigurationError, match="requires at least one"):
        process_video(source, manual_boxes=[])
