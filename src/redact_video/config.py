"""Validated runtime configuration for the redaction engine."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _default_ocr_cpu_threads() -> int:
    """Use available local cores without overcommitting a single OCR job."""

    return min(8, max(1, os.cpu_count() or 4))


class BlurMode(str, Enum):
    PRIVACY_BLUR = "privacy_blur"


class RedactionPolicy(str, Enum):
    CARD_DETAILS = "card_details"


class ProcessingMode(str, Enum):
    STANDARD = "standard"
    SAFE = "safe"
    FAST = "fast"


class BlurConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: BlurMode = BlurMode.PRIVACY_BLUR
    gaussian_sigma: float = Field(default=8.0, gt=0)
    min_padding_px: int = Field(default=6, ge=0)
    padding_ratio: float = Field(default=0.004, ge=0)
    pixel_block_size: int = Field(default=12, ge=2)

    def padding_for(self, frame_width: int, frame_height: int) -> int:
        return max(
            self.min_padding_px,
            round(min(frame_width, frame_height) * self.padding_ratio),
        )


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    codec: str = "libx264"
    crf: int = Field(default=18, ge=0, le=51)
    preset: Literal[
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ] = "veryfast"
    pixel_format: str = "yuv420p"
    preserve_audio: bool = True
    audio_bitrate: str = "192k"

    @field_validator("codec")
    @classmethod
    def normalize_codec(cls, value: str) -> str:
        if value in {"h264", "libx264"}:
            return "libx264"
        raise ValueError("V1 output codec must be h264 or libx264")


class MediaLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_dimension: int = Field(default=16, gt=0)
    max_dimension: int = Field(default=8192, gt=0)
    max_pixels: int = Field(default=35_000_000, gt=0)


class OcrConfig(BaseModel):
    """Local OCR settings optimized for English screen recordings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str = Field(default="en", min_length=2)
    ocr_version: str = "PP-OCRv6"
    device: str = "cpu"
    text_detection_model_name: str | None = "PP-OCRv6_tiny_det"
    text_recognition_model_name: str | None = "PP-OCRv6_tiny_rec"
    text_detection_model_dir: Path | None = None
    text_recognition_model_dir: Path | None = None
    minimum_confidence: float = Field(default=0.20, ge=0, le=1)
    max_input_dimension: int | None = Field(default=1920, ge=640, le=8192)
    apple_vision_max_input_dimension: int = Field(default=1440, ge=640, le=8192)
    apple_vision_recognition_level: Literal["accurate", "fast"] = "fast"
    apple_vision_full_resolution_fallback: bool = False
    text_detection_limit_side_length: int = Field(default=832, ge=320, le=4000)
    full_resolution_fallback: bool = True
    inference_engine: (
        Literal[
            "paddle",
            "paddle_static",
            "paddle_dynamic",
            "onnxruntime",
        ]
        | None
    ) = "onnxruntime"
    cpu_threads: int = Field(
        default_factory=_default_ocr_cpu_threads,
        ge=1,
        le=64,
    )
    enable_mkldnn: bool = True


class DetectionThresholds(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    card_number: float = Field(default=0.72, ge=0, le=1)
    expiry: float = Field(default=0.70, ge=0, le=1)
    cvv: float = Field(default=0.68, ge=0, le=1)
    saved_card: float = Field(default=0.70, ge=0, le=1)


class DetectionWeights(BaseModel):
    """Centralized evidence weights used by card detectors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic_label: float = Field(default=0.35, ge=0, le=1)
    payment_context: float = Field(default=0.20, ge=0, le=1)
    numeric_structure: float = Field(default=0.15, ge=0, le=1)
    luhn: float = Field(default=0.15, ge=0, le=1)
    field_geometry: float = Field(default=0.10, ge=0, le=1)
    neighboring_field: float = Field(default=0.05, ge=0, le=1)


class DetectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    thresholds: DetectionThresholds = Field(default_factory=DetectionThresholds)
    weights: DetectionWeights = Field(default_factory=DetectionWeights)
    fuzzy_label_threshold: float = Field(default=0.76, ge=0, le=1)
    minimum_payment_context: float = Field(default=0.50, ge=0, le=1)
    semantic_anchor_base_confidence: float = Field(default=0.76, ge=0, le=1)
    unlabeled_luhn_confidence: float = Field(default=0.74, ge=0, le=1)
    payment_luhn_confidence: float = Field(default=0.82, ge=0, le=1)
    blur_saved_card: bool = True
    max_ocr_span_tokens: int = Field(default=6, ge=1, le=12)
    scan_fps: float = Field(default=8.0, gt=0, le=120)
    safe_scan_fps: float = Field(default=12.0, gt=0, le=120)


class TrackingConfig(BaseModel):
    """Frame-to-frame tracking and privacy-retention settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sticky_duration_ms: int = Field(default=350, ge=0)
    reacquire_interval_ms: int = Field(default=500, gt=0)
    use_apple_vision_acceleration: bool = True
    overlap_source_ocr_and_tracking: bool = True
    authoritative_audit_interval_ms: int = Field(default=350, ge=1, le=60_000)
    reuse_authoritative_frames: bool = True
    authoritative_reuse_pixel_delta: int = Field(default=6, ge=0, le=32)
    authoritative_reuse_max_changed_pixels: int = Field(
        default=64,
        ge=0,
        le=10_000,
    )
    authoritative_reuse_max_changed_fraction: float = Field(
        default=0.00002,
        ge=0,
        le=0.01,
    )
    maximum_flow_points: int = Field(default=40, ge=4, le=500)
    optical_flow_workers: int = Field(default=4, ge=1, le=16)
    parallel_flow_min_frame_pixels: int = Field(
        default=1_000_000,
        ge=1,
        le=100_000_000,
    )
    minimum_flow_points: int = Field(default=3, ge=2, le=100)
    flow_quality_level: float = Field(default=0.01, gt=0, le=1)
    minimum_feature_distance_px: float = Field(default=4.0, gt=0)
    feature_margin_ratio: float = Field(default=0.50, ge=0, le=3)
    flow_window_size: int = Field(default=21, ge=5, le=101)
    flow_pyramid_levels: int = Field(default=3, ge=0, le=8)
    maximum_flow_error: float = Field(default=30.0, gt=0)
    maximum_forward_backward_error: float = Field(default=2.0, gt=0)
    minimum_flow_confidence: float = Field(default=0.45, ge=0, le=1)
    velocity_smoothing: float = Field(default=0.65, ge=0, le=1)
    template_search_radius_ratio: float = Field(default=1.0, ge=0.1, le=5)
    minimum_template_search_px: int = Field(default=24, ge=1)
    minimum_template_confidence: float = Field(default=0.72, ge=0, le=1)
    scene_change_size: int = Field(default=64, ge=16, le=256)
    scene_pixel_delta: int = Field(default=28, ge=1, le=255)
    scene_change_fraction: float = Field(default=0.38, ge=0, le=1)
    detection_match_iou: float = Field(default=0.20, ge=0, le=1)
    detection_match_distance_ratio: float = Field(default=1.25, gt=0, le=10)

    @model_validator(mode="after")
    def validate_flow_point_bounds(self) -> Self:
        if self.maximum_flow_points < self.minimum_flow_points:
            raise ValueError("maximum_flow_points cannot be below minimum_flow_points")
        return self


class VerificationConfig(BaseModel):
    """Rendered-output verification and automatic repair settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    max_passes: int = Field(default=3, ge=1, le=10)
    threshold_scale: float = Field(default=0.90, gt=0, le=1)
    reuse_clean_frames: bool = True
    reuse_clean_frames_across_passes: bool = True
    selective_cross_pass_decode: bool = True
    reuse_guard_max_dimension: int = Field(default=96, ge=32, le=512)
    reuse_pixel_delta: int = Field(default=12, ge=0, le=32)
    reuse_max_changed_pixels: int = Field(default=64, ge=0, le=10_000)
    reuse_max_changed_fraction: float = Field(default=0.005, ge=0, le=0.25)
    minimum_scan_interval_ms: int = Field(default=100, ge=1, le=60_000)
    immediate_scene_change_fraction: float = Field(default=0.20, ge=0, le=1)
    reuse_max_interval_ms: int = Field(default=500, ge=1, le=60_000)
    use_apple_vision_acceleration: bool = True
    authoritative_audit_interval_ms: int = Field(default=500, ge=1, le=60_000)
    prefetch_authoritative_audits: bool = True
    authoritative_prefetch_frames: int = Field(default=12, ge=1, le=120)
    authoritative_prefetch_max_bytes: int = Field(
        default=128 * 1024 * 1024,
        ge=1024 * 1024,
        le=4 * 1024 * 1024 * 1024,
    )
    repair_pre_roll_ms: int = Field(default=350, ge=0, le=60_000)
    repair_post_roll_ms: int = Field(default=500, ge=0, le=60_000)
    repair_padding_px: int = Field(default=12, ge=0, le=1024)

    @model_validator(mode="after")
    def validate_scan_intervals(self) -> Self:
        if self.minimum_scan_interval_ms > self.reuse_max_interval_ms:
            raise ValueError(
                "minimum_scan_interval_ms cannot exceed reuse_max_interval_ms"
            )
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: RedactionPolicy = RedactionPolicy.CARD_DETAILS

    ffmpeg_binary: str = Field(
        default_factory=lambda: os.getenv("REDACT_VIDEO_FFMPEG", "ffmpeg")
    )
    ffprobe_binary: str = Field(
        default_factory=lambda: os.getenv("REDACT_VIDEO_FFPROBE", "ffprobe")
    )
    blur: BlurConfig = Field(default_factory=BlurConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    media_limits: MediaLimits = Field(default_factory=MediaLimits)
