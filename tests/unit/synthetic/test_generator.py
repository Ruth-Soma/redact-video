from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from redact_video.media.decoder import FrameDecoder
from redact_video.models import SensitiveType
from tests.synthetic.generator import (
    ScenarioKind,
    SyntheticCorpusGenerator,
    decode_frame_marker,
)
from tests.synthetic.oracle import ScenarioOcrEngine


def test_encoded_scenario_retains_every_frame_marker(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required for synthetic video generation")
    scenario = SyntheticCorpusGenerator(tmp_path).generate(
        ScenarioKind.STATIC_LANDSCAPE
    )

    with FrameDecoder(scenario.path, scenario.frame_rate) as decoder:
        decoded_indices = [decode_frame_marker(frame.image) for frame in decoder]

    assert decoded_indices == list(range(len(scenario.frames)))


def test_oracle_can_hide_sensitive_tokens_without_hiding_labels(
    tmp_path: Path,
) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required for synthetic video generation")
    scenario = SyntheticCorpusGenerator(tmp_path).generate(
        ScenarioKind.STATIC_LANDSCAPE
    )
    source_ocr = ScenarioOcrEngine(scenario)
    verification_ocr = ScenarioOcrEngine(scenario, include_sensitive=False)
    with FrameDecoder(scenario.path, scenario.frame_rate) as decoder:
        first_frame = next(iter(decoder)).image

    source_tokens = source_ocr.scan(first_frame)
    verification_tokens = verification_ocr.scan(first_frame)

    assert len(source_tokens) > len(verification_tokens)
    assert any(
        token.sensitive_type is SensitiveType.CARD_NUMBER
        for token in scenario.frames[0].tokens
    )
    assert any(token.text == "Card Number" for token in scenario.frames[0].tokens)
    assert len(verification_tokens) >= 4
