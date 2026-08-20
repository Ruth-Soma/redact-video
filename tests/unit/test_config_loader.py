from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from redact_video.config import (
    BlurConfig,
    OutputConfig,
    RedactionPolicy,
    TrackingConfig,
    VerificationConfig,
)
from redact_video.config_loader import MAX_CONFIG_BYTES, load_app_config
from redact_video.exceptions import ConfigurationError


def test_loads_specification_shape_and_preserves_executable_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "redactor.yaml"
    path.write_text(
        """
redaction:
  policy: card_details
  blur:
    gaussian_sigma: 10
    min_padding_px: 8
detection:
  scan_fps: 9
  safe_scan_fps: 15
  thresholds:
    card_number: 0.75
tracking:
  sticky_duration_ms: 450
verification:
  max_passes: 4
output:
  codec: h264
  crf: 20
  preserve_audio: false
ocr:
  apple_vision_max_input_dimension: 1600
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("REDACT_VIDEO_FFMPEG", "custom-ffmpeg")

    config = load_app_config(path)

    assert config.policy is RedactionPolicy.CARD_DETAILS
    assert config.blur.gaussian_sigma == 10
    assert config.blur.min_padding_px == 8
    assert config.detection.scan_fps == 9
    assert config.detection.safe_scan_fps == 15
    assert config.detection.thresholds.card_number == 0.75
    assert config.tracking.sticky_duration_ms == 450
    assert config.verification.max_passes == 4
    assert config.output.codec == "libx264"
    assert config.output.crf == 20
    assert config.output.preserve_audio is False
    assert config.ocr.apple_vision_max_input_dimension == 1600
    assert config.ffmpeg_binary == "custom-ffmpeg"


def test_empty_configuration_uses_validated_defaults(tmp_path: Path) -> None:
    path = tmp_path / "defaults.yml"
    path.write_text("", encoding="utf-8")

    config = load_app_config(path)

    assert config.policy is RedactionPolicy.CARD_DETAILS
    assert config.verification.enabled is True
    assert config.verification.max_passes == 3
    assert config.verification.reuse_clean_frames_across_passes is True
    assert config.verification.selective_cross_pass_decode is True
    assert config.verification.prefetch_authoritative_audits is True
    assert config.verification.authoritative_prefetch_frames == 12
    assert config.verification.authoritative_prefetch_max_bytes == 134_217_728
    assert config.tracking.use_apple_vision_acceleration is True
    assert config.tracking.authoritative_audit_interval_ms == 350
    assert config.tracking.reuse_authoritative_frames is True
    assert config.tracking.maximum_flow_points == 40
    assert config.tracking.optical_flow_workers == 4
    assert config.tracking.parallel_flow_min_frame_pixels == 1_000_000
    assert config.tracking.flow_pyramid_levels == 3
    assert config.output.preset == "veryfast"
    assert config.ocr.text_detection_limit_side_length == 832


def test_output_config_accepts_only_named_x264_presets() -> None:
    assert OutputConfig(preset="medium").preset == "medium"

    with pytest.raises(ValidationError):
        OutputConfig(preset="not-a-real-preset")  # type: ignore[arg-type]


def test_tracking_authoritative_audit_interval_is_bounded() -> None:
    with pytest.raises(ValidationError):
        TrackingConfig(authoritative_audit_interval_ms=0)


def test_verification_scan_interval_cannot_exceed_forced_rescan_interval() -> None:
    with pytest.raises(ValidationError, match="minimum_scan_interval_ms"):
        VerificationConfig(
            minimum_scan_interval_ms=501,
            reuse_max_interval_ms=500,
        )


@pytest.mark.parametrize(
    ("contents", "location"),
    [
        ("remote_ocr_key: do-not-echo", "remote_ocr_key"),
        ("redaction:\n  blur:\n    secret_value: do-not-echo", "secret_value"),
        ("verification:\n  max_passes: 99", "verification.max_passes"),
    ],
)
def test_invalid_fields_report_only_locations(
    tmp_path: Path, contents: str, location: str
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigurationError) as caught:
        load_app_config(path)

    assert location in str(caught.value)
    assert "do-not-echo" not in str(caught.value)


def test_malformed_yaml_does_not_echo_source_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("secret_value: [do-not-echo", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="could not be parsed") as caught:
        load_app_config(path)

    assert "do-not-echo" not in str(caught.value)


def test_rejects_excessive_yaml_alias_expansion(tmp_path: Path) -> None:
    path = tmp_path / "aliases.yaml"
    aliases = "\n".join(f"field_{index}: *shared" for index in range(40))
    path.write_text(f"shared: &shared do-not-echo\n{aliases}\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="could not be parsed") as caught:
        load_app_config(path)

    assert "do-not-echo" not in str(caught.value)


def test_rejects_non_yaml_non_mapping_and_oversized_files(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "config.json"
    wrong_extension.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=".yaml or .yml"):
        load_app_config(wrong_extension)

    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="root must be a mapping"):
        load_app_config(sequence)

    oversized = tmp_path / "oversized.yaml"
    oversized.write_bytes(b"x" * (MAX_CONFIG_BYTES + 1))
    with pytest.raises(ConfigurationError, match="1 MiB"):
        load_app_config(oversized)


def test_runtime_config_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        BlurConfig.model_validate({"gaussian_sigma": 8, "unknown": "value"})
