"""Manual-region parsing used to validate the Phase 1 media pipeline."""

from __future__ import annotations

from pydantic import ValidationError

from redact_video.exceptions import ConfigurationError
from redact_video.models import BoundingBox


def parse_manual_box(value: str) -> BoundingBox:
    """Parse an ``x,y,width,height`` rectangle without echoing its contents."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ConfigurationError("Manual boxes must use x,y,width,height")
    try:
        coordinates = [int(part) for part in parts]
        return BoundingBox(
            x=coordinates[0],
            y=coordinates[1],
            width=coordinates[2],
            height=coordinates[3],
        )
    except (ValueError, ValidationError) as exc:
        raise ConfigurationError(
            "Manual-box coordinates must be non-negative integers with positive size"
        ) from exc
