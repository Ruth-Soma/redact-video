from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from redact_video.config import AppConfig, OutputConfig
from redact_video.media.probe import probe_media
from redact_video.models import JobStatus, SensitiveType
from redact_video.pipeline.automated import process_video_automated
from tests.synthetic.generator import (
    ScenarioKind,
    SyntheticCorpusGenerator,
    SyntheticScenario,
)
from tests.synthetic.metrics import measure_mask_timeline
from tests.synthetic.oracle import ScenarioOcrEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def scenario_factory(
    tmp_path: Path,
    ffmpeg_available: bool,
) -> Callable[[ScenarioKind], SyntheticScenario]:
    if not ffmpeg_available:
        pytest.skip("FFmpeg and ffprobe are required for synthetic corpus tests")
    generator = SyntheticCorpusGenerator(tmp_path / "corpus")
    return generator.generate


_REQUIRED_SCENARIOS = (
    ScenarioKind.STATIC_LANDSCAPE,
    ScenarioKind.STATIC_PORTRAIT,
    ScenarioKind.TYPING_PAN,
    ScenarioKind.TYPING_CVV,
    ScenarioKind.SLOW_SCROLL,
    ScenarioKind.FAST_SCROLL,
    ScenarioKind.DISAPPEAR_RETURN,
    ScenarioKind.MODAL_SLIDE,
    ScenarioKind.SAVED_CARD,
    ScenarioKind.BRIEF_APPEARANCE,
    ScenarioKind.FALSE_POSITIVES,
)

_OVERMASK_RATIO_LIMITS = {
    ScenarioKind.TYPING_PAN: 12.0,
    ScenarioKind.TYPING_CVV: 40.0,
}


@pytest.mark.parametrize("kind", _REQUIRED_SCENARIOS)
def test_required_synthetic_scenarios_have_zero_uncovered_sensitive_pixels(
    kind: ScenarioKind,
    scenario_factory: Callable[[ScenarioKind], SyntheticScenario],
    tmp_path: Path,
) -> None:
    scenario = scenario_factory(kind)
    output = tmp_path / f"{kind.value}.redacted.mp4"
    result = process_video_automated(
        scenario.path,
        output_path=output,
        ocr_engine=ScenarioOcrEngine(scenario),
        verification_ocr_engine=ScenarioOcrEngine(
            scenario,
            include_sensitive=False,
        ),
        config=AppConfig(output=OutputConfig(preset="ultrafast")),
    )
    metrics = measure_mask_timeline(scenario, result.mask_timeline)
    output_info = probe_media(output)
    expected_sensitive_frames = sum(
        bool(frame.sensitive_regions) for frame in scenario.frames
    )

    assert result.status is JobStatus.COMPLETED
    assert result.verification.passes == 1
    assert result.verification.remaining_leaks == 0
    assert output_info.display_width == scenario.width
    assert output_info.display_height == scenario.height
    assert metrics.missed_sensitive_pixels == 0
    assert metrics.coverage == 1.0
    assert metrics.masked_visible_pixels == 0
    if expected_sensitive_frames:
        assert result.masked_frames >= expected_sensitive_frames
        assert metrics.sensitive_pixels > 0
        assert metrics.overmask_ratio <= _OVERMASK_RATIO_LIMITS.get(kind, 6.0)
    else:
        assert result.masked_frames == 0
        assert metrics.mask_pixels == 0


def test_synthetic_saved_card_preserves_brand_and_masks_saved_details(
    scenario_factory: Callable[[ScenarioKind], SyntheticScenario],
    tmp_path: Path,
) -> None:
    scenario = scenario_factory(ScenarioKind.SAVED_CARD)
    result = process_video_automated(
        scenario.path,
        output_path=tmp_path / "saved-card.redacted.mp4",
        ocr_engine=ScenarioOcrEngine(scenario),
        verification_ocr_engine=ScenarioOcrEngine(
            scenario,
            include_sensitive=False,
        ),
        config=AppConfig(output=OutputConfig(preset="ultrafast")),
    )
    types = {
        region.type
        for regions in result.mask_timeline.frames.values()
        for region in regions
    }
    metrics = measure_mask_timeline(scenario, result.mask_timeline)

    assert SensitiveType.SAVED_CARD_NUMBER in types
    assert SensitiveType.SAVED_CARD_EXPIRY in types
    assert metrics.masked_visible_pixels == 0


def test_synthetic_false_positives_remain_completely_unmasked(
    scenario_factory: Callable[[ScenarioKind], SyntheticScenario],
    tmp_path: Path,
) -> None:
    scenario = scenario_factory(ScenarioKind.FALSE_POSITIVES)
    result = process_video_automated(
        scenario.path,
        output_path=tmp_path / "false-positives.redacted.mp4",
        ocr_engine=ScenarioOcrEngine(scenario),
        verification_ocr_engine=ScenarioOcrEngine(scenario),
        config=AppConfig(output=OutputConfig(preset="ultrafast")),
    )

    assert result.status is JobStatus.COMPLETED
    assert result.mask_timeline.frames == {}
    assert result.masked_frames == 0


def test_synthetic_intentional_leak_is_repaired_on_the_second_pass(
    scenario_factory: Callable[[ScenarioKind], SyntheticScenario],
    tmp_path: Path,
) -> None:
    scenario = scenario_factory(ScenarioKind.REPAIR_LEAK)
    source_ocr = ScenarioOcrEngine(
        scenario,
        excluded_types=frozenset({SensitiveType.CARD_CVV}),
        excluded_texts=frozenset({"CVV"}),
    )
    verification_ocr = ScenarioOcrEngine(
        scenario,
        excluded_types=frozenset(
            {SensitiveType.CARD_NUMBER, SensitiveType.CARD_EXPIRY}
        ),
        sensitive_passes=1,
    )

    result = process_video_automated(
        scenario.path,
        output_path=tmp_path / "repair-leak.redacted.mp4",
        ocr_engine=source_ocr,
        verification_ocr_engine=verification_ocr,
        config=AppConfig(output=OutputConfig(preset="ultrafast")),
    )
    metrics = measure_mask_timeline(scenario, result.mask_timeline)

    assert result.status is JobStatus.COMPLETED
    assert result.verification.passes == 2
    assert result.verification.repairs == 1
    assert result.verification.remaining_leaks == 0
    assert metrics.missed_sensitive_pixels == 0
    assert any(
        region.track_id.startswith("repair-p01")
        for regions in result.mask_timeline.frames.values()
        for region in regions
    )
