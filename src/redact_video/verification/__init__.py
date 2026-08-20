"""Rendered-output verification and automatic repair."""

from redact_video.verification.models import (
    RepairWindow,
    VerificationLeak,
    VerificationPassResult,
    VerificationSummary,
)
from redact_video.verification.repair import RepairEngine, build_repair_windows
from redact_video.verification.verifier import SafetyVerifier

__all__ = [
    "RepairEngine",
    "RepairWindow",
    "SafetyVerifier",
    "VerificationLeak",
    "VerificationPassResult",
    "VerificationSummary",
    "build_repair_windows",
]
