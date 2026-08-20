from __future__ import annotations

import shutil
import time
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from redact_video.api.config import ApiConfig
from redact_video.api.jobs import (
    JobNotFoundError,
    JobQueueFullError,
    LocalJobManager,
    WorkerOutcome,
    WorkerRequest,
    _config_for_mode,
    _failure_for,
    _progress_for,
    _public_error,
    run_redaction_job,
)
from redact_video.api.models import RedactionMode, RedactionPolicy
from redact_video.exceptions import OcrUnavailableError
from redact_video.models import JobStatus, ProcessingTimings


def _result_payload() -> dict[str, object]:
    return {
        "input_duration_ms": 1000,
        "output_duration_ms": 1000,
        "frames_processed": 10,
        "masked_frames": 8,
        "mask_regions": 9,
        "detections": {"card_number": 1},
        "verification": {
            "passes": 1,
            "repairs": 0,
            "remaining_leaks": 0,
            "potential_issues": [],
        },
        "audio_preserved": True,
        "variable_frame_rate_normalized": False,
        "warnings": [],
        "timings": {
            "inspection_ms": 1,
            "setup_ms": 1,
            "source_analysis_ms": 3,
            "initial_render_ms": 2,
            "initial_validation_ms": 1,
            "verification_passes_ms": [2],
            "repair_passes_ms": [],
            "publication_ms": 1,
            "total_ms": 10,
        },
    }


def _successful_worker(request: WorkerRequest) -> WorkerOutcome:
    shutil.copyfile(request.input_path, request.output_path)
    return WorkerOutcome(status="completed", result=_result_payload())


def _failed_worker(request: WorkerRequest) -> WorkerOutcome:
    del request
    return WorkerOutcome(status="failed", error_code="ocr_unavailable")


def _wait_for_terminal(manager: LocalJobManager, job_id: str) -> JobStatus:
    for _ in range(100):
        status = manager.status(job_id).status
        if status in {
            JobStatus.COMPLETED,
            JobStatus.NEEDS_REVIEW,
            JobStatus.FAILED,
        }:
            return status
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_local_manager_completes_job_and_deletes_raw_upload(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = LocalJobManager(
        ApiConfig(job_root=tmp_path, max_workers=1, max_queued_jobs=0),
        executor=executor,
        worker=_successful_worker,
        start_cleanup_thread=False,
    )
    try:
        prepared = manager.prepare_job(
            suffix=".mp4",
            policy=RedactionPolicy.CARD_DETAILS,
            mode=RedactionMode.SAFE,
        )
        prepared.input_path.write_bytes(b"synthetic video")
        manager.submit(prepared.job_id)

        assert _wait_for_terminal(manager, prepared.job_id) is JobStatus.COMPLETED
        assert not prepared.input_path.exists()
        assert manager.output_path(prepared.job_id) is not None
        report = manager.report(prepared.job_id)
        assert report.result is not None
        assert report.result.verification.remaining_leaks == 0
        assert "output_path" not in report.model_dump(mode="json")
    finally:
        manager.close()
        executor.shutdown(wait=True)


def test_local_manager_sanitizes_worker_failure(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = LocalJobManager(
        ApiConfig(job_root=tmp_path, max_workers=1),
        executor=executor,
        worker=_failed_worker,
        start_cleanup_thread=False,
    )
    try:
        prepared = manager.prepare_job(
            suffix=".mov",
            policy=RedactionPolicy.CARD_DETAILS,
            mode=RedactionMode.STANDARD,
        )
        prepared.input_path.write_bytes(b"not media")
        manager.submit(prepared.job_id)

        assert _wait_for_terminal(manager, prepared.job_id) is JobStatus.FAILED
        report = manager.report(prepared.job_id)
        assert report.result is None
        assert report.error is not None
        assert report.error.code == "ocr_unavailable"
        assert "not media" not in report.model_dump_json()
        assert not prepared.input_path.exists()
    finally:
        manager.close()
        executor.shutdown(wait=True)


def test_manager_rejects_work_beyond_bounded_capacity(tmp_path: Path) -> None:
    manager = LocalJobManager(
        ApiConfig(job_root=tmp_path, max_workers=1, max_queued_jobs=0),
        executor=_HoldingExecutor(),
        start_cleanup_thread=False,
    )
    first = manager.prepare_job(
        suffix=".mp4",
        policy=RedactionPolicy.CARD_DETAILS,
        mode=RedactionMode.STANDARD,
    )
    with pytest.raises(JobQueueFullError):
        manager.prepare_job(
            suffix=".mp4",
            policy=RedactionPolicy.CARD_DETAILS,
            mode=RedactionMode.STANDARD,
        )
    manager.discard(first.job_id)
    manager.close()


def test_manager_expires_terminal_artifacts(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    manager = LocalJobManager(
        ApiConfig(job_root=tmp_path, max_workers=1, retention_seconds=1),
        executor=executor,
        worker=_successful_worker,
        start_cleanup_thread=False,
    )
    try:
        prepared = manager.prepare_job(
            suffix=".mp4",
            policy=RedactionPolicy.CARD_DETAILS,
            mode=RedactionMode.STANDARD,
        )
        prepared.input_path.write_bytes(b"video")
        manager.submit(prepared.job_id)
        _wait_for_terminal(manager, prepared.job_id)
        finished_at = manager.status(prepared.job_id).finished_at
        assert finished_at is not None

        assert manager.cleanup_expired(now=finished_at + timedelta(seconds=2)) == 1
        with pytest.raises(JobNotFoundError):
            manager.status(prepared.job_id)
        assert not prepared.input_path.parent.exists()
    finally:
        manager.close()
        executor.shutdown(wait=True)


def test_progress_messages_map_to_sanitized_states() -> None:
    assert _progress_for("Inspecting video...") == (JobStatus.INSPECTING, 5)
    assert _progress_for("Verification pass 1/3...") == (JobStatus.VERIFYING, 75)
    assert _progress_for("unrecognized detail") is None


def test_modes_adjust_only_scan_frequency() -> None:
    standard = _config_for_mode(RedactionMode.STANDARD)
    safe = _config_for_mode(RedactionMode.SAFE)
    fast = _config_for_mode(RedactionMode.FAST)

    assert safe.detection.scan_fps == standard.detection.safe_scan_fps
    assert fast.detection.scan_fps == standard.detection.scan_fps / 2
    assert safe.verification == standard.verification == fast.verification


def test_expected_failures_have_fixed_public_messages() -> None:
    code, message = _failure_for(OcrUnavailableError("secret vendor detail"))
    assert code == "ocr_unavailable"
    assert "secret" not in message
    assert _public_error("../../unsafe").code == "worker_failed"


def test_child_worker_excludes_paths_and_status_from_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    status_path = tmp_path / "status.json"
    input_path.write_bytes(b"video")

    class _Result:
        status = JobStatus.COMPLETED
        input_duration_ms = 1000
        output_duration_ms = 1000
        frames_processed = 10
        masked_frames = 8
        mask_regions = 9
        audio_preserved = True
        variable_frame_rate_normalized = False
        warnings = ()
        timings = ProcessingTimings(
            inspection_ms=1,
            setup_ms=1,
            source_analysis_ms=3,
            initial_render_ms=2,
            initial_validation_ms=1,
            verification_passes_ms=(2,),
            repair_passes_ms=(),
            publication_ms=1,
            total_ms=10,
        )

        def __init__(self, path: Path) -> None:
            self.output_path = path
            self.tracks_by_type = {"card_number": 1}
            self.verification = {
                "passes": 1,
                "repairs": 0,
                "remaining_leaks": 0,
                "potential_issues": [],
            }

    def fake_process(*_: object, **kwargs: object) -> _Result:
        progress = kwargs["progress"]
        assert callable(progress)
        progress("Rendering privacy mask timeline...")
        output_path.write_bytes(b"redacted")
        return _Result(output_path)

    monkeypatch.setattr("redact_video.api.jobs.process_video_automated", fake_process)
    outcome = run_redaction_job(
        WorkerRequest(
            job_id="red_test",
            input_path=str(input_path),
            output_path=str(output_path),
            status_path=str(status_path),
            mode="safe",
        )
    )

    assert outcome.status == "completed"
    assert outcome.result is not None
    assert "output_path" not in outcome.result
    assert status_path.read_text() == '{"status":"rendering","progress":55}'


def test_child_worker_sanitizes_core_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_processing(*_: object, **__: object) -> None:
        raise OcrUnavailableError("vendor detail with recognized text")

    monkeypatch.setattr(
        "redact_video.api.jobs.process_video_automated", fail_processing
    )
    outcome = run_redaction_job(
        WorkerRequest(
            job_id="red_test",
            input_path=str(tmp_path / "input.mp4"),
            output_path=str(tmp_path / "output.mp4"),
            status_path=str(tmp_path / "status.json"),
            mode="standard",
        )
    )

    assert outcome == WorkerOutcome(status="failed", error_code="ocr_unavailable")


def test_child_worker_loads_yaml_before_applying_safe_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(
        "detection:\n  scan_fps: 3\n  safe_scan_fps: 17\n",
        encoding="utf-8",
    )

    def inspect_config(*_: object, **kwargs: object) -> None:
        config = kwargs["config"]
        assert config.detection.scan_fps == 17
        raise OcrUnavailableError("sanitized after assertion")

    monkeypatch.setattr("redact_video.api.jobs.process_video_automated", inspect_config)
    outcome = run_redaction_job(
        WorkerRequest(
            job_id="red_test",
            input_path=str(tmp_path / "input.mp4"),
            output_path=str(tmp_path / "output.mp4"),
            status_path=str(tmp_path / "status.json"),
            mode="safe",
            config_path=str(config_path),
        )
    )

    assert outcome.error_code == "ocr_unavailable"


class _HoldingExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        del fn, args, kwargs
        return Future()
