from __future__ import annotations

from pathlib import Path

import pytest

from redact_video.media.decoder import (
    SelectedFrameDecoder,
    _selection_chunks,
    _selection_expression,
)


def test_selection_expression_coalesces_contiguous_frame_ranges() -> None:
    assert _selection_expression((1, 2, 3, 8, 10, 11)) == (
        "between(n\\,1\\,3)+eq(n\\,8)+between(n\\,10\\,11)"
    )


def test_selection_chunks_bound_expression_terms_without_splitting_ranges() -> None:
    indices = (0, 1, 4, 8, 12, 13, 20, 24)

    assert _selection_chunks(indices, max_terms=3) == (
        (0, 1, 4, 8),
        (12, 13, 20, 24),
    )


@pytest.mark.parametrize("frame_indices", [(), (2, 1), (1, 1), (-1,)])
def test_selected_decoder_requires_unique_ordered_indices(
    frame_indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        SelectedFrameDecoder(
            Path("unused.mp4"),
            30,
            frame_indices=frame_indices,
        )
