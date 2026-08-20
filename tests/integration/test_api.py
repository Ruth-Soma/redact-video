from __future__ import annotations

import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from redact_video.api.app import create_app
from redact_video.api.config import ApiConfig
from redact_video.api.jobs import LocalJobManager, WorkerOutcome, WorkerRequest
from redact_video.api.models import RedactionMode, RedactionPolicy


def _api_result() -> dict[str, object]:
    return {
        "input_duration_ms": 1000,
        "output_duration_ms": 1000,
        "frames_processed": 10,
        "masked_frames": 10,
        "mask_regions": 10,
        "detections": {"card_number": 1},
        "verification": {
            "passes": 1,
            "repairs": 0,
            "remaining_leaks": 0,
            "potential_issues": [],
        },
        "audio_preserved": False,
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


def _api_worker(request: WorkerRequest) -> WorkerOutcome:
    shutil.copyfile(request.input_path, request.output_path)
    return WorkerOutcome(status="completed", result=_api_result())


def _poll(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/v1/redactions/{job_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "needs_review", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("API job did not finish")


def test_api_job_lifecycle_output_and_sanitized_report(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    config = ApiConfig(job_root=tmp_path, max_workers=1, max_queued_jobs=1)
    manager = LocalJobManager(
        config,
        executor=executor,
        worker=_api_worker,
        start_cleanup_thread=False,
    )
    app = create_app(config, job_manager=manager)
    try:
        with TestClient(app) as client:
            assert client.get("/healthz").json() == {"status": "ok"}
            output_schema = client.get("/openapi.json").json()["paths"][
                "/v1/redactions/{job_id}/output"
            ]["get"]["responses"]["200"]["content"]
            assert "video/mp4" in output_schema
            created = client.post(
                "/v1/redactions",
                files={"file": ("payment.mp4", b"synthetic video", "video/mp4")},
                data={"policy": "card_details", "mode": "safe"},
            )
            assert created.status_code == 202, created.text
            assert created.json()["status"] == "queued"
            job_id = created.json()["job_id"]
            assert job_id.startswith("red_")

            status_payload = _poll(client, job_id)
            assert status_payload["status"] == "completed"
            assert status_payload["progress"] == 100

            report = client.get(f"/v1/redactions/{job_id}/report")
            assert report.status_code == 200
            report_text = report.text
            assert report.json()["mode"] == "safe"
            assert report.json()["result"]["verification"]["remaining_leaks"] == 0
            assert report.json()["result"]["detections"] == {"card_number": 1}
            assert "output_path" not in report_text
            assert "payment.mp4" not in report_text

            output = client.get(f"/v1/redactions/{job_id}/output")
            assert output.status_code == 200
            assert output.headers["content-type"].startswith("video/mp4")
            assert output.content == b"synthetic video"

            assert client.get("/v1/redactions/red_missing").status_code == 404
    finally:
        manager.close()
        executor.shutdown(wait=True)


def test_default_local_process_pool_executes_worker(tmp_path: Path) -> None:
    manager = LocalJobManager(
        ApiConfig(job_root=tmp_path, max_workers=1, max_queued_jobs=0),
        worker=_api_worker,
        start_cleanup_thread=False,
    )
    try:
        prepared = manager.prepare_job(
            suffix=".mp4",
            policy=RedactionPolicy.CARD_DETAILS,
            mode=RedactionMode.STANDARD,
        )
        prepared.input_path.write_bytes(b"process pool video")
        manager.submit(prepared.job_id)

        for _ in range(200):
            if manager.status(prepared.job_id).status.value == "completed":
                break
            time.sleep(0.01)
        else:
            raise AssertionError("process-pool job did not finish")
        assert (
            manager.output_path(prepared.job_id).read_bytes() == b"process pool video"
        )
    finally:
        manager.close()


def test_api_rejects_invalid_empty_and_oversized_uploads(tmp_path: Path) -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    config = ApiConfig(
        job_root=tmp_path,
        max_upload_bytes=4,
        upload_chunk_bytes=4096,
        max_workers=1,
    )
    manager = LocalJobManager(
        config,
        executor=executor,
        worker=_api_worker,
        start_cleanup_thread=False,
    )
    try:
        with TestClient(create_app(config, job_manager=manager)) as client:
            wrong_type = client.post(
                "/v1/redactions",
                files={"file": ("payment.txt", b"1234", "text/plain")},
            )
            assert wrong_type.status_code == 415

            empty = client.post(
                "/v1/redactions",
                files={"file": ("payment.mp4", b"", "video/mp4")},
            )
            assert empty.status_code == 400

            oversized = client.post(
                "/v1/redactions",
                files={"file": ("payment.mov", b"12345", "video/quicktime")},
            )
            assert oversized.status_code == 413

            invalid_mode = client.post(
                "/v1/redactions",
                files={"file": ("payment.mp4", b"1234", "video/mp4")},
                data={"mode": "unsafe"},
            )
            assert invalid_mode.status_code == 422
    finally:
        manager.close()
        executor.shutdown(wait=True)


def test_api_returns_conflict_while_output_and_report_are_pending(
    tmp_path: Path,
) -> None:
    executor = _HoldingExecutor()
    config = ApiConfig(job_root=tmp_path, max_workers=1, max_queued_jobs=0)
    manager = LocalJobManager(
        config,
        executor=executor,
        start_cleanup_thread=False,
    )
    try:
        with TestClient(create_app(config, job_manager=manager)) as client:
            created = client.post(
                "/v1/redactions",
                files={"file": ("payment.mp4", b"video", "video/mp4")},
            )
            assert created.status_code == 202
            job_id = created.json()["job_id"]
            assert client.get(f"/v1/redactions/{job_id}/output").status_code == 409
            assert client.get(f"/v1/redactions/{job_id}/report").status_code == 409

            full = client.post(
                "/v1/redactions",
                files={"file": ("second.mp4", b"video", "video/mp4")},
            )
            assert full.status_code == 503
            assert full.headers["retry-after"] == "5"
    finally:
        manager.close()


class _HoldingExecutor:
    def submit(self, fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        del fn, args, kwargs
        return Future()
