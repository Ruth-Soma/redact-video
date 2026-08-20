"""Golden-frame privacy coverage and over-redaction metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from redact_video.config import BlurConfig
from redact_video.models import BoundingBox
from redact_video.redaction.timeline import MaskTimeline
from tests.synthetic.generator import SyntheticScenario


@dataclass(frozen=True, slots=True)
class GoldenMaskMetrics:
    sensitive_pixels: int
    covered_sensitive_pixels: int
    missed_sensitive_pixels: int
    mask_pixels: int
    overmask_pixels: int
    masked_visible_pixels: int

    @property
    def coverage(self) -> float:
        if self.sensitive_pixels == 0:
            return 1.0
        return self.covered_sensitive_pixels / self.sensitive_pixels

    @property
    def overmask_ratio(self) -> float:
        if self.sensitive_pixels == 0:
            return float(self.overmask_pixels > 0)
        return self.overmask_pixels / self.sensitive_pixels


def _paint(mask: np.ndarray, bbox: BoundingBox) -> None:
    mask[bbox.y : bbox.bottom, bbox.x : bbox.right] = True


def measure_mask_timeline(
    scenario: SyntheticScenario,
    timeline: MaskTimeline,
    *,
    blur_config: BlurConfig | None = None,
) -> GoldenMaskMetrics:
    """Compare the rendered mask footprint with per-frame ground truth."""

    config = blur_config or BlurConfig()
    padding = config.padding_for(scenario.width, scenario.height)
    sensitive_pixels = 0
    covered_sensitive_pixels = 0
    missed_sensitive_pixels = 0
    mask_pixels = 0
    overmask_pixels = 0
    masked_visible_pixels = 0

    for truth in scenario.frames:
        sensitive = np.zeros((scenario.height, scenario.width), dtype=np.bool_)
        visible = np.zeros_like(sensitive)
        rendered_mask = np.zeros_like(sensitive)
        for region in truth.sensitive_regions:
            _paint(sensitive, region.bbox)
        for bbox in truth.visible_regions:
            _paint(visible, bbox)
        for bbox in timeline.boxes_for(truth.frame_index):
            padded = bbox.expanded(padding, padding).clipped(
                scenario.width,
                scenario.height,
            )
            if padded is not None:
                _paint(rendered_mask, padded)

        sensitive_count = int(np.count_nonzero(sensitive))
        covered_count = int(np.count_nonzero(sensitive & rendered_mask))
        sensitive_pixels += sensitive_count
        covered_sensitive_pixels += covered_count
        missed_sensitive_pixels += sensitive_count - covered_count
        mask_pixels += int(np.count_nonzero(rendered_mask))
        overmask_pixels += int(np.count_nonzero(rendered_mask & ~sensitive))
        masked_visible_pixels += int(np.count_nonzero(rendered_mask & visible))

    return GoldenMaskMetrics(
        sensitive_pixels=sensitive_pixels,
        covered_sensitive_pixels=covered_sensitive_pixels,
        missed_sensitive_pixels=missed_sensitive_pixels,
        mask_pixels=mask_pixels,
        overmask_pixels=overmask_pixels,
        masked_visible_pixels=masked_visible_pixels,
    )
