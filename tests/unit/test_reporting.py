from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from redact_video.exceptions import ConfigurationError, ReportWriteError
from redact_video.models import JobStatus, ProcessingTimings, SensitiveType
from redact_video.reporting import (
    SanitizedProcessReport,
    SanitizedResultReport,
    validate_report_destination,
    write_sanitized_report,
)
from redact_video.verification import RepairWindow, VerificationSummary


def _result(*, status: JobStatus = JobStatus.COMPLETED) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        output_path=Path("private-output.mp4"),
        input_duration_ms=1000,
        output_duration_ms=1001,
        frames_processed=10,
        masked_frames=8,
        mask_regions=9,
        tracks_by_type={SensitiveType.CARD_NUMBER: 1},
        verification=VerificationSummary(
            passes=2,
            repairs=1,
            remaining_leaks=0,
        ),
        audio_preserved=True,
        variable_frame_rate_normalized=False,
        warnings=("sanitized warning",),
        timings=ProcessingTimings(
            inspection_ms=10,
            setup_ms=20,
            source_analysis_ms=300,
            initial_render_ms=100,
            initial_validation_ms=5,
            verification_passes_ms=(80, 70),
            repair_passes_ms=(30,),
            publication_ms=5,
            total_ms=620,
        ),
        mask_timeline="private timeline",
    )


def test_report_models_exclude_paths_timelines_and_recognized_values() -> None:
    details = SanitizedResultReport.from_result(_result())
    report = SanitizedProcessReport.from_result(_result())
    serialized = report.model_dump_json()

    assert details.detections == {SensitiveType.CARD_NUMBER: 1}
    assert report.status is JobStatus.COMPLETED
    assert "output_path" not in serialized
    assert "mask_timeline" not in serialized
    assert "private-output" not in serialized
    assert report.timings.verification_passes_ms == (80, 70)


def test_needs_review_report_retains_only_sanitized_issue_window() -> None:
    source = _result(status=JobStatus.NEEDS_REVIEW)
    source.verification = VerificationSummary(
        passes=3,
        repairs=2,
        remaining_leaks=1,
        potential_issues=(
            RepairWindow(
                start_ms=300,
                end_ms=900,
                types=(SensitiveType.CARD_CVV,),
            ),
        ),
    )

    payload = SanitizedProcessReport.from_result(source).model_dump(mode="json")

    assert payload["status"] == "needs_review"
    assert payload["verification"]["potential_issues"] == [
        {"start_ms": 300, "end_ms": 900, "types": ["card_cvv"]}
    ]


def test_report_destination_validation_is_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    existing = tmp_path / "existing.json"
    existing.write_text("existing", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=".json"):
        validate_report_destination(
            tmp_path / "report.txt",
            input_path=source,
            output_path=output,
            overwrite=False,
        )
    with pytest.raises(ConfigurationError, match="must differ"):
        validate_report_destination(
            output,
            input_path=source,
            output_path=output,
            overwrite=False,
        )
    with pytest.raises(ConfigurationError, match="already exists"):
        validate_report_destination(
            existing,
            input_path=source,
            output_path=output,
            overwrite=False,
        )


def test_writes_private_atomic_json_and_requires_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "reports" / "result.json"
    report = SanitizedProcessReport.from_result(_result())

    write_sanitized_report(report, destination, overwrite=False)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["detections"] == {"card_number": 1}
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(destination.parent.glob("*.partial.json"))
    with pytest.raises(ConfigurationError, match="already exists"):
        write_sanitized_report(report, destination, overwrite=False)

    write_sanitized_report(report, destination, overwrite=True)
    assert json.loads(destination.read_text())["verification"]["passes"] == 2


def test_report_write_failure_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "result.json"
    report = SanitizedProcessReport.from_result(_result())

    def fail_link(*_: object) -> None:
        raise OSError("private filesystem detail")

    monkeypatch.setattr("redact_video.reporting.os.link", fail_link)
    with pytest.raises(ReportWriteError, match="could not be written") as caught:
        write_sanitized_report(report, destination, overwrite=False)

    assert "private filesystem" not in str(caught.value)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.partial.json"))
