from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from redact_video.models import SensitiveType
from redact_video.redaction.timeline import MaskRegion, MaskTimeline
from tests.synthetic.generator import ScenarioKind, SyntheticCorpusGenerator
from tests.synthetic.metrics import measure_mask_timeline


def test_golden_metrics_report_covered_and_missed_sensitive_pixels(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required for synthetic video generation")
    scenario = SyntheticCorpusGenerator(tmp_path).generate(
        ScenarioKind.STATIC_LANDSCAPE
    )
    first_region = scenario.frames[0].sensitive_regions[0]
    timeline = MaskTimeline(
        frames={
            0: (
                MaskRegion(
                    track_id="golden-000001",
                    type=SensitiveType.CARD_NUMBER,
                    bbox=first_region.bbox,
                ),
            )
        }
    )

    metrics = measure_mask_timeline(scenario, timeline)

    assert metrics.covered_sensitive_pixels > 0
    assert metrics.missed_sensitive_pixels > 0
    assert metrics.coverage < 1
    assert metrics.overmask_pixels > 0
