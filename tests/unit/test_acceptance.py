from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from redact_video.acceptance import app, run_acceptance
from redact_video.config import AppConfig, ProcessingMode
from redact_video.exceptions import ConfigurationError, UnsupportedInputError
from redact_video.models import JobStatus, ProcessingTimings, SensitiveType
from redact_video.verification import RepairWindow, VerificationSummary


def _processor_result(status: JobStatus) -> SimpleNamespace:
    issues = (
        (
            RepairWindow(
                start_ms=100,
                end_ms=500,
                types=(SensitiveType.CARD_NUMBER,),
            ),
        )
        if status is JobStatus.NEEDS_REVIEW
        else ()
    )
    return SimpleNamespace(
        status=status,
        input_duration_ms=1000,
        output_duration_ms=1000,
        frames_processed=10,
        masked_frames=8,
        mask_regions=8,
        tracks_by_type={SensitiveType.CARD_NUMBER: 1},
        verification=VerificationSummary(
            passes=3 if issues else 1,
            repairs=1 if issues else 0,
            remaining_leaks=1 if issues else 0,
            potential_issues=issues,
        ),
        audio_preserved=True,
        variable_frame_rate_normalized=False,
        warnings=(),
        timings=ProcessingTimings(
            inspection_ms=1,
            setup_ms=1,
            source_analysis_ms=3,
            initial_render_ms=2,
            initial_validation_ms=1,
            verification_passes_ms=(2,),
            repair_passes_ms=(),
            publication_ms=1,
            total_ms=10,
        ),
    )


def test_external_acceptance_uses_anonymous_outputs_and_private_report(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "private-inputs"
    output_dir = tmp_path / "reviewed"
    source_dir.mkdir()
    (source_dir / "customer-4242424242424242.mp4").write_bytes(b"video one")
    (source_dir / "cvv-729.mov").write_bytes(b"video two")
    seen_scan_rates: list[float] = []

    def processor(source: Path, **kwargs: object) -> SimpleNamespace:
        del source
        config = kwargs["config"]
        seen_scan_rates.append(config.detection.scan_fps)
        Path(kwargs["output_path"]).write_bytes(b"redacted")
        return _processor_result(JobStatus.COMPLETED)

    report_path = output_dir / "acceptance.json"
    report = run_acceptance(
        source_dir,
        output_dir,
        report_path=report_path,
        config=AppConfig(),
        mode=ProcessingMode.SAFE,
        processor=processor,
    )

    assert report.acceptance_passed is True
    assert report.completed == 2
    assert seen_scan_rates == [12.0, 12.0]
    assert (output_dir / "sample_001.redacted.mp4").is_file()
    assert (output_dir / "sample_002.redacted.mp4").is_file()
    report_text = report_path.read_text(encoding="utf-8")
    assert "4242424242424242" not in report_text
    assert "cvv-729" not in report_text
    assert str(source_dir) not in report_text
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600


def test_acceptance_retains_sanitized_review_and_failure_states(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    (source_dir / "a-review.mp4").write_bytes(b"review")
    (source_dir / "b-fail.mp4").write_bytes(b"fail")

    def processor(source: Path, **kwargs: object) -> SimpleNamespace:
        if source.name.startswith("b-"):
            raise UnsupportedInputError("do-not-echo-4242424242424242")
        Path(kwargs["output_path"]).write_bytes(b"redacted")
        return _processor_result(JobStatus.NEEDS_REVIEW)

    report_path = output_dir / "acceptance.json"
    report = run_acceptance(
        source_dir,
        output_dir,
        report_path=report_path,
        config=AppConfig(),
        processor=processor,
    )

    assert report.acceptance_passed is False
    assert report.completed == 0
    assert report.needs_review == 1
    assert report.failed == 1
    assert report.performance_failures == 0
    assert report.samples[0].result is not None
    assert report.samples[0].result.verification.remaining_leaks == 1
    assert report.samples[1].error_code == "unsupported_input"
    assert "do-not-echo" not in report_path.read_text(encoding="utf-8")


def test_acceptance_preflight_rejects_unsafe_destinations(tmp_path: Path) -> None:
    source_dir = tmp_path / "inputs"
    source_dir.mkdir()
    (source_dir / "sample.mp4").write_bytes(b"video")

    with pytest.raises(ConfigurationError, match="must differ"):
        run_acceptance(
            source_dir,
            source_dir,
            report_path=tmp_path / "report.json",
            config=AppConfig(),
        )
    with pytest.raises(ConfigurationError, match=".json"):
        run_acceptance(
            source_dir,
            tmp_path / "outputs",
            report_path=tmp_path / "report.txt",
            config=AppConfig(),
        )


def test_acceptance_command_returns_success_without_printing_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    source_dir.mkdir()
    secret_name = "customer-4242424242424242.mp4"
    (source_dir / secret_name).write_bytes(b"video")

    def processor(source: Path, **kwargs: object) -> SimpleNamespace:
        del source
        Path(kwargs["output_path"]).write_bytes(b"redacted")
        return _processor_result(JobStatus.COMPLETED)

    monkeypatch.setattr(
        "redact_video.acceptance.process_video_automated",
        processor,
    )
    report_path = output_dir / "report.json"
    result = CliRunner().invoke(
        app,
        [str(source_dir), str(output_dir), "--report", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    assert "Completed: 1/1" in result.output
    assert secret_name not in result.output
    assert json.loads(report_path.read_text())["acceptance_passed"] is True
