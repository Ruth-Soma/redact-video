from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from redact_video.cli import app
from redact_video.models import BoundingBox, OcrToken

pytestmark = pytest.mark.integration

runner = CliRunner()


def test_cli_renders_a_manual_box(
    media_factory: Callable[..., Path], tmp_path: Path
) -> None:
    source = media_factory(suffix=".mp4", with_audio=True)
    output = tmp_path / "cli-output.mp4"

    result = runner.invoke(
        app,
        [
            str(source),
            "--output",
            str(output),
            "--manual-box",
            "40,40,180,70",
            "--verbose",
        ],
    )

    assert result.exit_code == 3, result.output
    assert output.is_file()
    assert "Manual-mask render complete" in result.output
    assert "Status: NEEDS_REVIEW" in result.output
    assert "Frames processed: 10" in result.output


def test_cli_runs_automatic_mode_without_a_manual_box(
    media_factory: Callable[..., Path],
    card_field_ocr,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "automatic-cli.mp4"
    monkeypatch.setattr(
        "redact_video.pipeline.automated.PaddleOcrEngine",
        lambda config: card_field_ocr,
    )

    result = runner.invoke(app, [str(source), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert "Automatic detection/tracking render complete" in result.output
    assert "Masked frames: 10" in result.output
    assert "Verification passes: 1" in result.output
    assert "Status: COMPLETED" in result.output


def test_cli_rejects_safe_and_fast_together(tmp_path: Path) -> None:
    result = runner.invoke(app, [str(tmp_path / "input.mp4"), "--safe", "--fast"])

    assert result.exit_code == 4
    assert "cannot be used together" in result.output


def test_cli_loads_yaml_and_writes_sanitized_report(
    media_factory: Callable[..., Path],
    card_field_ocr,  # type: ignore[no-untyped-def]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "configured-output.mp4"
    report = tmp_path / "configured-report.json"
    config = tmp_path / "redactor.yaml"
    config.write_text(
        """
detection:
  scan_fps: 9
  safe_scan_fps: 15
output:
  codec: h264
  preset: ultrafast
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "redact_video.pipeline.automated.PaddleOcrEngine",
        lambda config: card_field_ocr,
    )

    result = runner.invoke(
        app,
        [
            str(source),
            "--output",
            str(output),
            "--config",
            str(config),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["verification"]["remaining_leaks"] == 0
    assert payload["detections"] == {"card_number": 1}
    report_text = report.read_text(encoding="utf-8")
    assert str(source) not in report_text
    assert str(output) not in report_text
    assert "Report:" in result.output


def test_cli_rejects_invalid_yaml_without_echoing_submitted_value(
    tmp_path: Path,
) -> None:
    config = tmp_path / "invalid.yaml"
    config.write_text("unknown_key: do-not-echo", encoding="utf-8")

    result = runner.invoke(
        app,
        [str(tmp_path / "missing.mp4"), "--config", str(config)],
    )

    assert result.exit_code == 4
    assert "unknown_key" in result.output
    assert "do-not-echo" not in result.output


def test_cli_rejects_report_for_unverified_manual_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            str(tmp_path / "input.mp4"),
            "--manual-box",
            "1,1,10,10",
            "--report",
            str(tmp_path / "report.json"),
        ],
    )

    assert result.exit_code == 4
    assert "automatic verified jobs" in result.output


class _PersistentLeakOcr:
    def scan(self, frame: np.ndarray) -> list[OcrToken]:
        del frame
        return [
            OcrToken(
                text="4242424242424242",
                confidence=0.99,
                bbox=BoundingBox(x=70, y=80, width=150, height=28),
            )
        ]


def test_cli_returns_needs_review_when_verification_exhausts_its_passes(
    media_factory: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = media_factory(suffix=".mp4", with_audio=False)
    output = tmp_path / "automatic-needs-review.mp4"
    persistent_ocr = _PersistentLeakOcr()
    monkeypatch.setattr(
        "redact_video.pipeline.automated.PaddleOcrEngine",
        lambda config: persistent_ocr,
    )

    result = runner.invoke(app, [str(source), "--output", str(output)])

    assert result.exit_code == 3, result.output
    assert output.is_file()
    assert "Verification passes: 3" in result.output
    assert "Remaining possible leaks: 10" in result.output
    assert "Potential issue: 00:00:00.000–00:00:01.000" in result.output
    assert "Status: NEEDS_REVIEW" in result.output
