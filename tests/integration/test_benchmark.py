from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.synthetic import run_synthetic_benchmark
from tests.synthetic.generator import ScenarioKind

pytestmark = pytest.mark.integration


def test_synthetic_release_benchmark_emits_value_free_metrics(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark.json"

    report = run_synthetic_benchmark(
        (ScenarioKind.STATIC_LANDSCAPE,),
        report_path=report_path,
        maximum_realtime_factor=20,
    )

    assert report.benchmark_passed is True
    assert report.privacy_failures == 0
    assert report.performance_failures == 0
    scenario = report.scenarios[0]
    assert scenario.status.value == "completed"
    assert scenario.coverage == 1.0
    assert scenario.missed_sensitive_pixels == 0
    assert scenario.masked_visible_pixels == 0
    assert scenario.throughput_fps > 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert "output_path" not in serialized
    assert "mask_timeline" not in serialized
    assert "4242424242424242" not in serialized
