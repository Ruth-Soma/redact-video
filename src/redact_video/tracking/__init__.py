"""Frame-oriented sensitive-region tracking."""

from redact_video.tracking.optical_flow import MotionEstimate, OpticalFlowEstimator
from redact_video.tracking.scene_change import SceneChangeDetector, SceneChangeResult
from redact_video.tracking.template_match import TemplateMatch, TemplateMatcher
from redact_video.tracking.tracker import (
    AdaptiveTrackingEngine,
    DetectionProvider,
    RegionTracker,
)

__all__ = [
    "AdaptiveTrackingEngine",
    "DetectionProvider",
    "MotionEstimate",
    "OpticalFlowEstimator",
    "RegionTracker",
    "SceneChangeDetector",
    "SceneChangeResult",
    "TemplateMatch",
    "TemplateMatcher",
]
